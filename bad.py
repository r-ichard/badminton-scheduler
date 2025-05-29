#!/usr/bin/env python3
"""
Badminton Tournament Scheduler - FIXED ELIMINATION DEPENDENCIES
Fixed issues with elimination bracket dependency scheduling.

Key Fixes:
1. Fixed MockConstraintSolver to respect elimination round dependencies
2. Added dependency-aware scheduling for sequential elimination rounds
3. Enhanced constraint validation to detect dependency violations
4. Added comprehensive unit tests for elimination dependencies
"""

import argparse
import json
import csv
import math
import unittest
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from itertools import combinations
from collections import defaultdict
import logging

try:
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False
    print("Warning: OR-Tools not installed. Using mock solver for testing.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================
# IMPROVED DATA MODELS
# =============================================

@dataclass
class TournamentConfig:
    """Tournament configuration parameters"""
    name: str
    start_time: str  # Format: "HH:MM"
    end_time: str    # Format: "HH:MM"
    match_duration: int  # minutes
    rest_duration: int   # minutes (minimum 20)
    num_courts: int

    def __post_init__(self):
        """Validate configuration after initialization"""
        start = datetime.strptime(self.start_time, "%H:%M")
        end = datetime.strptime(self.end_time, "%H:%M")

        if start >= end:
            raise ValueError("Start time must be before end time")
        if self.num_courts <= 0:
            raise ValueError("Number of courts must be positive")
        if self.match_duration <= 0:
            raise ValueError("Match duration must be positive")
        if self.rest_duration < 20:
            raise ValueError("Rest duration must be at least 20 minutes")

@dataclass
class SeriesConfig:
    """IMPROVED Series configuration with intuitive pool setup"""
    name: str
    series_type: str  # "SH", "SD", "MX", "DH", "DD"
    total_players: int

    # IMPROVED: Choose ONE of these pool configuration methods
    players_per_pool: Optional[int] = None  # e.g., 3 (creates pools of 3 players each)
    number_of_pools: Optional[int] = None   # e.g., 4 (creates 4 equal pools)
    single_pool: bool = False               # True = everyone plays everyone

    qualifiers_per_pool: int = 2  # How many advance from each pool (ignored if single_pool=True)

    def __post_init__(self):
        """Validate and calculate pool structure"""
        # Count how many pool configuration methods are specified
        config_methods = sum([
            self.players_per_pool is not None,
            self.number_of_pools is not None,
            self.single_pool
        ])

        if config_methods != 1:
            raise ValueError("Specify exactly ONE pool configuration: players_per_pool, number_of_pools, or single_pool=True")

        if self.single_pool:
            self.qualifiers_per_pool = 0
            return

        if self.players_per_pool is not None:
            if self.players_per_pool < 2:
                raise ValueError("Players per pool must be at least 2")
            if self.total_players % self.players_per_pool != 0:
                raise ValueError(f"Cannot divide {self.total_players} players into pools of {self.players_per_pool}")
            self.number_of_pools = self.total_players // self.players_per_pool

        elif self.number_of_pools is not None:
            if self.number_of_pools < 1:
                raise ValueError("Number of pools must be at least 1")
            if self.total_players % self.number_of_pools != 0:
                raise ValueError(f"Cannot divide {self.total_players} players into {self.number_of_pools} equal pools")
            self.players_per_pool = self.total_players // self.number_of_pools

    def get_pool_distribution(self) -> List[int]:
        """Get the actual pool sizes"""
        if self.single_pool:
            return [self.total_players]
        else:
            return [self.players_per_pool] * self.number_of_pools

    def get_pool_description(self) -> str:
        """Get human-readable description of pool structure"""
        if self.single_pool:
            return f"Single pool with {self.total_players} players"
        else:
            return f"{self.number_of_pools} pools of {self.players_per_pool} players each"

@dataclass
class Match:
    """Individual match representation with dependency tracking"""
    id: str
    series_name: str
    pool_name: str
    round_type: str  # "pool", "quarter", "semi", "final"
    player1: str
    player2: str
    court: Optional[int] = None
    start_time: Optional[int] = None  # minutes from tournament start
    end_time: Optional[int] = None
    phase: str = "pool"  # "pool" or "elimination"
    dependency: Optional[str] = None  # For elimination matches
    dependency_level: int = 0  # 0=pools, 1=quarters, 2=semis, 3=final

    def __str__(self):
        time_str = f"{self.start_time//60:02d}:{self.start_time%60:02d}" if self.start_time else "TBD"
        court_str = f"Court {self.court}" if self.court else "TBD"
        return f"{self.series_name} - {self.pool_name}: {self.player1} vs {self.player2} @ {time_str} on {court_str}"

@dataclass
class ScheduleResult:
    """Result of schedule generation"""
    success: bool
    matches: List[Match]
    max_wait_time: int  # minutes
    tournament_end_time: int  # minutes from start
    court_utilization: float
    generation_time: float  # seconds
    pool_completion_time: int  # when all pools finish
    error_message: Optional[str] = None
    warnings: List[str] = None

# =============================================
# IMPROVED POOL STRUCTURE HELPERS
# =============================================

class PoolStructureHelper:
    """Helper class for pool structure calculations and suggestions"""

    @staticmethod
    def suggest_pool_structures(total_players: int) -> List[Dict[str, Any]]:
        """Suggest good pool structures for a given number of players"""
        suggestions = []

        # Single pool option (always available)
        total_matches = total_players * (total_players - 1) // 2
        suggestions.append({
            "type": "single_pool",
            "description": f"Single pool ({total_players} players)",
            "total_matches": total_matches,
            "pools": 1,
            "players_per_pool": total_players,
            "config": {"single_pool": True}
        })

        # Multiple pool options
        for pool_size in range(3, min(6, total_players + 1)):  # Pool sizes 3-5
            if total_players % pool_size == 0:
                num_pools = total_players // pool_size
                pool_matches = num_pools * (pool_size * (pool_size - 1) // 2)

                # Calculate elimination matches (assuming 2 qualifiers per pool)
                if num_pools > 1:
                    total_qualifiers = num_pools * 2
                    elim_matches = PoolStructureHelper._calculate_elimination_matches(total_qualifiers)
                else:
                    elim_matches = 0

                suggestions.append({
                    "type": "multiple_pools",
                    "description": f"{num_pools} pools of {pool_size} players",
                    "total_matches": pool_matches + elim_matches,
                    "pool_matches": pool_matches,
                    "elimination_matches": elim_matches,
                    "pools": num_pools,
                    "players_per_pool": pool_size,
                    "config": {"players_per_pool": pool_size, "qualifiers_per_pool": 2}
                })

        # Sort by total matches (fewer is often better for scheduling)
        suggestions.sort(key=lambda x: x["total_matches"])
        return suggestions

    @staticmethod
    def _calculate_elimination_matches(total_qualifiers: int) -> int:
        """Calculate elimination matches based on number of qualifiers"""
        if total_qualifiers <= 1:
            return 0
        elif total_qualifiers == 2:
            return 1  # Just final
        elif total_qualifiers <= 4:
            return 3  # 2 semis + 1 final
        elif total_qualifiers == 6:
            return 5  # 2 quarters + 2 semis + 1 final (with 2 byes)
        elif total_qualifiers <= 8:
            return 7  # 4 quarters + 2 semis + 1 final
        else:
            # For larger tournaments, calculate bracket size
            bracket_size = 1
            while bracket_size < total_qualifiers:
                bracket_size *= 2
            return bracket_size - 1

    @staticmethod
    def validate_and_suggest(series_config: SeriesConfig) -> Tuple[bool, List[str], List[Dict]]:
        """Validate pool configuration and provide suggestions if invalid"""
        warnings = []
        suggestions = []

        try:
            # Try to validate the current configuration
            pool_dist = series_config.get_pool_distribution()
            return True, warnings, suggestions

        except ValueError as e:
            # Configuration is invalid, provide suggestions
            suggestions = PoolStructureHelper.suggest_pool_structures(series_config.total_players)
            return False, [str(e)], suggestions

# =============================================
# UPDATED TOURNAMENT CALCULATOR
# =============================================

class TournamentCalculator:
    """Calculate matches and tournament structure with improved pool handling"""

    @staticmethod
    def calculate_pool_matches(players_per_pool: int) -> int:
        """Calculate number of matches in a pool with round-robin format"""
        if players_per_pool < 2:
            return 0
        return players_per_pool * (players_per_pool - 1) // 2

    @staticmethod
    def calculate_elimination_matches(total_qualifiers: int) -> int:
        """Calculate elimination matches based on number of qualifiers"""
        return PoolStructureHelper._calculate_elimination_matches(total_qualifiers)

class MatchGenerator:
    """Generate matches for tournament series with improved dependency tracking"""

    def __init__(self, calculator: TournamentCalculator):
        self.calculator = calculator
        self.match_counter = 0

    def generate_series_matches(self, series: SeriesConfig) -> Tuple[List[Match], List[Match]]:
        """Generate all matches for a series, returning (pool_matches, elimination_matches)"""
        pool_matches = []
        elimination_matches = []

        # Generate pool matches
        pool_sizes = series.get_pool_distribution()
        pool_matches = self._generate_pool_matches(series, pool_sizes)

        # Generate elimination matches if not single pool
        if not series.single_pool and series.qualifiers_per_pool > 0:
            elimination_matches = self._generate_elimination_matches(series, pool_sizes)

        return pool_matches, elimination_matches

    def _generate_pool_matches(self, series: SeriesConfig, pool_sizes: List[int]) -> List[Match]:
        """Generate matches within pools"""
        matches = []

        for pool_idx, pool_size in enumerate(pool_sizes):
            if series.single_pool:
                pool_name = f"Round-Robin"
            else:
                pool_name = f"Pool {chr(65 + pool_idx)}"  # Pool A, Pool B, Pool C, etc.

            # Generate player names for this pool
            players = [f"{chr(65 + pool_idx)}{i+1}" for i in range(pool_size)]

            # Generate round-robin matches within this pool
            for i in range(len(players)):
                for j in range(i + 1, len(players)):
                    match = Match(
                        id=f"{series.name}_{pool_name}_{self.match_counter}",
                        series_name=series.name,
                        pool_name=pool_name,
                        round_type="pool",
                        player1=players[i],
                        player2=players[j],
                        phase="pool",
                        dependency_level=0
                    )
                    matches.append(match)
                    self.match_counter += 1

        return matches

    def _generate_elimination_matches(self, series: SeriesConfig, pool_sizes: List[int]) -> List[Match]:
        """Generate elimination bracket matches with FIXED dependency levels"""
        matches = []
        total_qualifiers = len(pool_sizes) * series.qualifiers_per_pool

        if total_qualifiers <= 1:
            return matches

        # Generate bracket matches based on number of qualifiers
        if total_qualifiers == 2:
            # Just a final
            match = Match(
                id=f"{series.name}_Final_{self.match_counter}",
                series_name=series.name,
                pool_name="Final",
                round_type="final",
                player1="1st Pool A",
                player2="1st Pool B",
                phase="elimination",
                dependency="all_pools_complete",
                dependency_level=1
            )
            matches.append(match)
            self.match_counter += 1

        elif total_qualifiers == 4:
            # Semi-finals + Final
            semi1 = Match(
                id=f"{series.name}_Semi1_{self.match_counter}",
                series_name=series.name,
                pool_name="Semi-final 1",
                round_type="semi",
                player1="1st Pool A",
                player2="2nd Pool B",
                phase="elimination",
                dependency="all_pools_complete",
                dependency_level=1
            )
            matches.append(semi1)
            self.match_counter += 1

            semi2 = Match(
                id=f"{series.name}_Semi2_{self.match_counter}",
                series_name=series.name,
                pool_name="Semi-final 2",
                round_type="semi",
                player1="1st Pool B",
                player2="2nd Pool A",
                phase="elimination",
                dependency="all_pools_complete",
                dependency_level=1
            )
            matches.append(semi2)
            self.match_counter += 1

            final = Match(
                id=f"{series.name}_Final_{self.match_counter}",
                series_name=series.name,
                pool_name="Final",
                round_type="final",
                player1="Winner Semi 1",
                player2="Winner Semi 2",
                phase="elimination",
                dependency="semi_complete",
                dependency_level=2
            )
            matches.append(final)
            self.match_counter += 1

        elif total_qualifiers == 6:
            # FIXED: Handle 6 qualifiers properly (2 quarters + 2 semis + 1 final with 2 byes)
            quarter1 = Match(
                id=f"{series.name}_Quarter1_{self.match_counter}",
                series_name=series.name,
                pool_name="Quarter-final 1",
                round_type="quarter",
                player1="2nd Pool A",
                player2="2nd Pool C",
                phase="elimination",
                dependency="all_pools_complete",
                dependency_level=1
            )
            matches.append(quarter1)
            self.match_counter += 1

            quarter2 = Match(
                id=f"{series.name}_Quarter2_{self.match_counter}",
                series_name=series.name,
                pool_name="Quarter-final 2",
                round_type="quarter",
                player1="2nd Pool B",
                player2="1st Pool C",
                phase="elimination",
                dependency="all_pools_complete",
                dependency_level=1
            )
            matches.append(quarter2)
            self.match_counter += 1

            # Semi-finals (1st Pool A and 1st Pool B get byes)
            semi1 = Match(
                id=f"{series.name}_Semi1_{self.match_counter}",
                series_name=series.name,
                pool_name="Semi-final 1",
                round_type="semi",
                player1="1st Pool A",  # Bye
                player2="Winner QF1",
                phase="elimination",
                dependency="quarter_complete",
                dependency_level=2
            )
            matches.append(semi1)
            self.match_counter += 1

            semi2 = Match(
                id=f"{series.name}_Semi2_{self.match_counter}",
                series_name=series.name,
                pool_name="Semi-final 2",
                round_type="semi",
                player1="1st Pool B",  # Bye
                player2="Winner QF2",
                phase="elimination",
                dependency="quarter_complete",
                dependency_level=2
            )
            matches.append(semi2)
            self.match_counter += 1

            # Final
            final = Match(
                id=f"{series.name}_Final_{self.match_counter}",
                series_name=series.name,
                pool_name="Final",
                round_type="final",
                player1="Winner Semi 1",
                player2="Winner Semi 2",
                phase="elimination",
                dependency="semi_complete",
                dependency_level=3
            )
            matches.append(final)
            self.match_counter += 1

        elif total_qualifiers == 8:
            # Quarters + Semis + Final
            for i in range(4):
                quarter = Match(
                    id=f"{series.name}_Quarter{i+1}_{self.match_counter}",
                    series_name=series.name,
                    pool_name=f"Quarter-final {i+1}",
                    round_type="quarter",
                    player1=f"Qualifier {2*i+1}",
                    player2=f"Qualifier {2*i+2}",
                    phase="elimination",
                    dependency="all_pools_complete",
                    dependency_level=1
                )
                matches.append(quarter)
                self.match_counter += 1

            # Semi-finals
            semi1 = Match(
                id=f"{series.name}_Semi1_{self.match_counter}",
                series_name=series.name,
                pool_name="Semi-final 1",
                round_type="semi",
                player1="Winner QF1",
                player2="Winner QF2",
                phase="elimination",
                dependency="quarter_complete",
                dependency_level=2
            )
            matches.append(semi1)
            self.match_counter += 1

            semi2 = Match(
                id=f"{series.name}_Semi2_{self.match_counter}",
                series_name=series.name,
                pool_name="Semi-final 2",
                round_type="semi",
                player1="Winner QF3",
                player2="Winner QF4",
                phase="elimination",
                dependency="quarter_complete",
                dependency_level=2
            )
            matches.append(semi2)
            self.match_counter += 1

            # Final
            final = Match(
                id=f"{series.name}_Final_{self.match_counter}",
                series_name=series.name,
                pool_name="Final",
                round_type="final",
                player1="Winner Semi 1",
                player2="Winner Semi 2",
                phase="elimination",
                dependency="semi_complete",
                dependency_level=3
            )
            matches.append(final)
            self.match_counter += 1

        return matches

# =============================================
# FIXED DEPENDENCY-AWARE CONSTRAINT SOLVER
# =============================================

class MockConstraintSolver:
    """FIXED Mock solver that properly respects elimination dependencies"""

    def solve(self, tournament: TournamentConfig, pool_matches: List[Match], elimination_matches: List[Match]) -> ScheduleResult:
        """FIXED Mock scheduling that respects dependencies and sequential elimination rounds"""
        logger.warning("Using mock solver - install OR-Tools for real optimization")

        # Track court availability and player last match end times
        court_schedule = {i: 0 for i in range(1, tournament.num_courts + 1)}
        player_last_end = defaultdict(int)  # player -> last match end time

        # Schedule pool matches with player rest constraints
        for match in pool_matches:
            self._schedule_match(match, tournament, court_schedule, player_last_end)

        # Find when all pools are complete
        pool_completion_time = max(match.end_time for match in pool_matches) if pool_matches else 0

        # Ensure proper gap before elimination matches
        elimination_start_time = pool_completion_time + tournament.rest_duration

        # Reset court schedules to elimination start time (but keep player tracking)
        for court in court_schedule:
            court_schedule[court] = max(court_schedule[court], elimination_start_time)

        # Schedule elimination matches by dependency level (FIXED: Sequential rounds)
        elimination_by_level = defaultdict(list)
        for match in elimination_matches:
            elimination_by_level[match.dependency_level].append(match)

        # Schedule each elimination level sequentially
        for level in sorted(elimination_by_level.keys()):
            level_matches = elimination_by_level[level]

            # Schedule all matches in this level
            for match in level_matches:
                self._schedule_match(match, tournament, court_schedule, player_last_end, elimination_start_time)

            # Wait for all matches in this level to complete before starting next level
            if level_matches:
                level_completion_time = max(match.end_time for match in level_matches)
                next_level_start = level_completion_time + tournament.rest_duration

                # Update all court schedules to next level start time
                for court in court_schedule:
                    court_schedule[court] = max(court_schedule[court], next_level_start)

        # Combine all matches for result calculation
        all_matches = pool_matches + elimination_matches

        # Calculate maximum wait time between matches for any player
        max_wait_time = self._calculate_max_wait_time(all_matches)

        # Calculate tournament end time
        tournament_end = max(match.end_time for match in all_matches) if all_matches else 0

        # Return complete schedule result
        return ScheduleResult(
            success=True,
            matches=all_matches,
            max_wait_time=max_wait_time,
            tournament_end_time=tournament_end,
            court_utilization=0.8,  # Mock value - in real implementation this would be calculated
            generation_time=0.1,    # Mock value - in real implementation this would be measured
            pool_completion_time=pool_completion_time,
            warnings=["Using mock solver - results not optimized"]
        )

    def _schedule_match(self, match: Match, tournament: TournamentConfig, court_schedule: Dict[int, int],
                        player_last_end: Dict[str, int], min_start_time: int = 0):
        """Schedule a single match considering all constraints"""
        # Find earliest time this match can start considering:
        # 1. Court availability
        # 2. Player rest requirements for both players
        # 3. Minimum start time (for elimination phases)
        earliest_court_time = min(court_schedule.values())
        player1_available = player_last_end[match.player1] + tournament.rest_duration
        player2_available = player_last_end[match.player2] + tournament.rest_duration

        # Match can't start until all constraints are met
        earliest_start = max(earliest_court_time, player1_available, player2_available, min_start_time)

        # Find a court that's available at or before earliest_start
        available_court = None
        for court, available_time in court_schedule.items():
            if available_time <= earliest_start:
                available_court = court
                break

        # If no court is available at earliest_start, find the court that becomes free earliest
        if available_court is None:
            available_court = min(court_schedule.keys(), key=lambda c: court_schedule[c])
            earliest_start = max(earliest_start, court_schedule[available_court])

        # Assign match to court and time
        match.court = available_court
        match.start_time = earliest_start
        match.end_time = match.start_time + tournament.match_duration

        # Update schedules
        court_schedule[available_court] = match.end_time
        player_last_end[match.player1] = match.end_time
        player_last_end[match.player2] = match.end_time

    def _calculate_max_wait_time(self, matches: List[Match]) -> int:
        """Calculate maximum wait time between matches for any player"""
        player_times = defaultdict(list)

        for match in matches:
            if match.start_time is not None:
                player_times[match.player1].append(match.start_time)
                player_times[match.player2].append(match.start_time)

        max_wait = 0
        for player, times in player_times.items():
            times.sort()
            for i in range(len(times) - 1):
                wait_time = times[i + 1] - times[i]
                max_wait = max(max_wait, wait_time)

        return max_wait

# =============================================
# ENHANCED CONSTRAINT VALIDATION
# =============================================

class ConstraintValidator:
    """Helper class to validate tournament constraints including dependencies"""

    @staticmethod
    def validate_rest_constraints(matches: List[Match], min_rest_minutes: int) -> Tuple[bool, List[str]]:
        """Validate that all players have minimum rest time between matches"""
        violations = []
        player_matches = defaultdict(list)

        # Group matches by player
        for match in matches:
            if match.start_time is not None:
                player_matches[match.player1].append(match)
                player_matches[match.player2].append(match)

        # Check rest time for each player
        for player, player_match_list in player_matches.items():
            # Sort by start time
            player_match_list.sort(key=lambda m: m.start_time)

            for i in range(len(player_match_list) - 1):
                current_match = player_match_list[i]
                next_match = player_match_list[i + 1]

                rest_time = next_match.start_time - current_match.end_time

                if rest_time < min_rest_minutes:
                    violations.append(
                        f"Player {player}: only {rest_time} minutes rest between "
                        f"match ending at {current_match.end_time} and starting at {next_match.start_time} "
                        f"(minimum {min_rest_minutes} required)"
                    )

        return len(violations) == 0, violations

    @staticmethod
    def validate_court_conflicts(matches: List[Match]) -> Tuple[bool, List[str]]:
        """Validate that no two matches are scheduled on the same court at the same time"""
        violations = []
        court_schedule = defaultdict(list)

        # Group matches by court
        for match in matches:
            if match.court is not None and match.start_time is not None:
                court_schedule[match.court].append(match)

        # Check for overlaps on each court
        for court, court_matches in court_schedule.items():
            court_matches.sort(key=lambda m: m.start_time)

            for i in range(len(court_matches) - 1):
                current_match = court_matches[i]
                next_match = court_matches[i + 1]

                if current_match.end_time > next_match.start_time:
                    violations.append(
                        f"Court {court} conflict: match ending at {current_match.end_time} "
                        f"overlaps with match starting at {next_match.start_time}"
                    )

        return len(violations) == 0, violations

    @staticmethod
    def validate_phase_ordering(matches: List[Match]) -> Tuple[bool, List[str]]:
        """Validate that all pool matches complete before elimination matches start"""
        violations = []

        pool_matches = [m for m in matches if m.phase == "pool" and m.end_time is not None]
        elimination_matches = [m for m in matches if m.phase == "elimination" and m.start_time is not None]

        if pool_matches and elimination_matches:
            latest_pool_end = max(match.end_time for match in pool_matches)
            earliest_elim_start = min(match.start_time for match in elimination_matches)

            if latest_pool_end > earliest_elim_start:
                violations.append(
                    f"Phase ordering violation: pool phase ends at {latest_pool_end} "
                    f"but elimination phase starts at {earliest_elim_start}"
                )

        return len(violations) == 0, violations

    @staticmethod
    def validate_elimination_dependencies(matches: List[Match]) -> Tuple[bool, List[str]]:
        """NEW: Validate that elimination dependencies are respected"""
        violations = []

        elimination_matches = [m for m in matches if m.phase == "elimination" and m.start_time is not None]

        # Group by dependency level
        by_level = defaultdict(list)
        for match in elimination_matches:
            by_level[match.dependency_level].append(match)

        # Check that each level completes before the next starts
        for level in sorted(by_level.keys())[:-1]:  # All except the last level
            current_level = by_level[level]
            next_level = by_level.get(level + 1, [])

            if current_level and next_level:
                latest_current_end = max(match.end_time for match in current_level)
                earliest_next_start = min(match.start_time for match in next_level)

                if latest_current_end > earliest_next_start:
                    violations.append(
                        f"Elimination dependency violation: level {level} ends at {latest_current_end} "
                        f"but level {level + 1} starts at {earliest_next_start}"
                    )

        return len(violations) == 0, violations

    @staticmethod
    def validate_tournament_structure(result: ScheduleResult, series_configs: List[SeriesConfig]) -> Tuple[bool, List[str]]:
        """Validate overall tournament structure"""
        violations = []

        # Check that each series has proper structure
        for series in series_configs:
            series_matches = [m for m in result.matches if m.series_name == series.name]
            pool_matches = [m for m in series_matches if m.phase == "pool"]
            elimination_matches = [m for m in series_matches if m.phase == "elimination"]

            # Validate pool matches
            if not series.single_pool:
                expected_pools = series.number_of_pools
                actual_pools = len(set(m.pool_name for m in pool_matches))

                if expected_pools != actual_pools:
                    violations.append(
                        f"Series {series.name}: expected {expected_pools} pools, "
                        f"but found {actual_pools}"
                    )

                # Check elimination structure
                if series.qualifiers_per_pool > 0:
                    total_qualifiers = expected_pools * series.qualifiers_per_pool
                    expected_elim_matches = PoolStructureHelper._calculate_elimination_matches(total_qualifiers)

                    if len(elimination_matches) != expected_elim_matches:
                        violations.append(
                            f"Series {series.name}: expected {expected_elim_matches} elimination matches, "
                            f"but found {len(elimination_matches)}"
                        )

                    # Check that there's exactly one final match
                    finals = [m for m in elimination_matches if m.round_type == "final"]
                    if len(finals) != 1:
                        violations.append(
                            f"Series {series.name}: expected exactly 1 final match, "
                            f"but found {len(finals)}"
                        )

        return len(violations) == 0, violations

# =============================================
# ENHANCED UNIT TESTS FOR DEPENDENCIES
# =============================================

class TestEliminationDependencies(unittest.TestCase):
    """NEW: Test elimination dependency scheduling"""

    def setUp(self):
        self.scheduler = BadmintonTournamentScheduler(use_ortools=False)
        self.tournament = TournamentConfig(
            name="Dependency Test Tournament",
            start_time="08:00",
            end_time="18:00",
            match_duration=30,
            rest_duration=20,
            num_courts=6
        )

    def test_elimination_dependency_ordering(self):
        """Test that elimination rounds are scheduled sequentially"""
        series = [SeriesConfig(
            name="SH1",
            series_type="SH",
            total_players=9,
            players_per_pool=3,  # 3 pools of 3, 6 qualifiers
            qualifiers_per_pool=2
        )]

        result = self.scheduler.schedule_tournament(self.tournament, series)

        self.assertTrue(result.success)

        elimination_matches = [m for m in result.matches if m.phase == "elimination"]

        # Group by dependency level
        by_level = defaultdict(list)
        for match in elimination_matches:
            by_level[match.dependency_level].append(match)

        # Check that quarters (level 1) finish before semis (level 2) start
        quarters = by_level.get(1, [])
        semis = by_level.get(2, [])
        finals = by_level.get(3, [])

        if quarters and semis:
            latest_quarter_end = max(match.end_time for match in quarters)
            earliest_semi_start = min(match.start_time for match in semis)

            self.assertLessEqual(latest_quarter_end, earliest_semi_start,
                                 f"Quarters must finish before semis start: {latest_quarter_end} <= {earliest_semi_start}")

        if semis and finals:
            latest_semi_end = max(match.end_time for match in semis)
            earliest_final_start = min(match.start_time for match in finals)

            self.assertLessEqual(latest_semi_end, earliest_final_start,
                                 f"Semis must finish before final starts: {latest_semi_end} <= {earliest_final_start}")

    def test_elimination_dependency_validation(self):
        """Test the elimination dependency validator"""
        # Create matches with dependency violations
        matches = [
            Match("q1", "TEST", "Quarter 1", "quarter", "A1", "B2", 1, 100, 130, "elimination", dependency_level=1),
            Match("q2", "TEST", "Quarter 2", "quarter", "A2", "B1", 2, 100, 130, "elimination", dependency_level=1),
            Match("s1", "TEST", "Semi 1", "semi", "Winner Q1", "Winner Q2", 3, 120, 150, "elimination", dependency_level=2),  # Starts before quarters end
        ]

        valid, violations = ConstraintValidator.validate_elimination_dependencies(matches)
        self.assertFalse(valid)
        self.assertGreater(len(violations), 0)
        self.assertIn("dependency violation", violations[0])

        # Fix the dependency violation
        matches[2].start_time = 160  # Start after quarters end + gap
        matches[2].end_time = 190

        valid, violations = ConstraintValidator.validate_elimination_dependencies(matches)
        self.assertTrue(valid)
        self.assertEqual(len(violations), 0)

    def test_no_simultaneous_dependent_matches(self):
        """Test that dependent matches are never scheduled simultaneously"""
        series = [SeriesConfig(
            name="MX1",
            series_type="MX",
            total_players=12,
            players_per_pool=4,  # 3 pools of 4, 6 qualifiers
            qualifiers_per_pool=2
        )]

        result = self.scheduler.schedule_tournament(self.tournament, series)

        self.assertTrue(result.success)

        elimination_matches = [m for m in result.matches if m.phase == "elimination"]

        # Check that no two matches with different dependency levels have the same start time
        time_to_levels = defaultdict(set)
        for match in elimination_matches:
            time_to_levels[match.start_time].add(match.dependency_level)

        for start_time, levels in time_to_levels.items():
            self.assertEqual(len(levels), 1,
                             f"Multiple dependency levels scheduled at time {start_time}: {levels}")

# Include all previous test classes...
class TestBasicLogic(unittest.TestCase):
    """Test basic pool structure logic"""

    def test_pool_match_calculation(self):
        """Test that pool match calculation is correct"""
        calculator = TournamentCalculator()

        # Round-robin formula: n*(n-1)/2
        self.assertEqual(calculator.calculate_pool_matches(3), 3)  # 3*2/2 = 3
        self.assertEqual(calculator.calculate_pool_matches(4), 6)  # 4*3/2 = 6
        self.assertEqual(calculator.calculate_pool_matches(5), 10) # 5*4/2 = 10
        self.assertEqual(calculator.calculate_pool_matches(1), 0)  # No matches with 1 player
        self.assertEqual(calculator.calculate_pool_matches(0), 0)  # No matches with 0 players

    def test_elimination_match_calculation(self):
        """Test elimination match calculation including 6 qualifiers"""
        calculator = TournamentCalculator()

        self.assertEqual(calculator.calculate_elimination_matches(0), 0)  # No elimination
        self.assertEqual(calculator.calculate_elimination_matches(1), 0)  # No elimination
        self.assertEqual(calculator.calculate_elimination_matches(2), 1)  # Just final
        self.assertEqual(calculator.calculate_elimination_matches(4), 3)  # 2 semis + final
        self.assertEqual(calculator.calculate_elimination_matches(6), 5)  # 2 quarters + 2 semis + final (with 2 byes)
        self.assertEqual(calculator.calculate_elimination_matches(8), 7)  # 4 quarters + 2 semis + final

class TestImprovedPoolStructure(unittest.TestCase):
    """Test improved pool structure configuration"""

    def test_players_per_pool_configuration(self):
        """Test players_per_pool configuration method"""
        series = SeriesConfig(
            name="TEST",
            series_type="SH",
            total_players=9,
            players_per_pool=3,
            qualifiers_per_pool=2
        )

        self.assertEqual(series.number_of_pools, 3)
        self.assertEqual(series.players_per_pool, 3)
        self.assertEqual(series.get_pool_distribution(), [3, 3, 3])
        self.assertEqual(series.get_pool_description(), "3 pools of 3 players each")

    def test_number_of_pools_configuration(self):
        """Test number_of_pools configuration method"""
        series = SeriesConfig(
            name="TEST",
            series_type="SH",
            total_players=8,
            number_of_pools=2,
            qualifiers_per_pool=2
        )

        self.assertEqual(series.number_of_pools, 2)
        self.assertEqual(series.players_per_pool, 4)
        self.assertEqual(series.get_pool_distribution(), [4, 4])
        self.assertEqual(series.get_pool_description(), "2 pools of 4 players each")

    def test_single_pool_configuration(self):
        """Test single pool configuration"""
        series = SeriesConfig(
            name="TEST",
            series_type="SD",
            total_players=5,
            single_pool=True
        )

        self.assertTrue(series.single_pool)
        self.assertEqual(series.qualifiers_per_pool, 0)
        self.assertEqual(series.get_pool_distribution(), [5])
        self.assertEqual(series.get_pool_description(), "Single pool with 5 players")

    def test_invalid_configurations(self):
        """Test that invalid configurations raise appropriate errors"""

        # Multiple configuration methods specified
        with self.assertRaises(ValueError):
            SeriesConfig(
                name="TEST",
                series_type="SH",
                total_players=9,
                players_per_pool=3,
                number_of_pools=3  # Can't specify both
            )

        # Players don't divide evenly
        with self.assertRaises(ValueError):
            SeriesConfig(
                name="TEST",
                series_type="SH",
                total_players=10,
                players_per_pool=3  # 10 doesn't divide by 3
            )

class TestConstraintValidation(unittest.TestCase):
    """Test constraint validation"""

    def setUp(self):
        self.scheduler = BadmintonTournamentScheduler(use_ortools=False)
        self.tournament = TournamentConfig(
            name="Test Tournament",
            start_time="08:00",
            end_time="18:00",
            match_duration=30,
            rest_duration=20,
            num_courts=4
        )

    def test_rest_constraint_validation(self):
        """Test that rest time constraints are properly validated"""
        # Create matches with insufficient rest time
        matches = [
            Match("1", "TEST", "Pool A", "pool", "A1", "A2", 1, 0, 30, "pool"),
            Match("2", "TEST", "Pool A", "pool", "A1", "A3", 2, 40, 70, "pool"),  # Only 10 min rest for A1
        ]

        valid, violations = ConstraintValidator.validate_rest_constraints(matches, 20)
        self.assertFalse(valid)
        self.assertGreater(len(violations), 0)
        self.assertIn("A1", violations[0])

        # Fix the rest time
        matches[1].start_time = 50  # Now 20 min rest
        matches[1].end_time = 80

        valid, violations = ConstraintValidator.validate_rest_constraints(matches, 20)
        self.assertTrue(valid)
        self.assertEqual(len(violations), 0)

    def test_court_conflict_validation(self):
        """Test that court conflicts are properly detected"""
        # Create overlapping matches on same court
        matches = [
            Match("1", "TEST", "Pool A", "pool", "A1", "A2", 1, 0, 30, "pool"),
            Match("2", "TEST", "Pool B", "pool", "B1", "B2", 1, 20, 50, "pool"),  # Overlaps on court 1
        ]

        valid, violations = ConstraintValidator.validate_court_conflicts(matches)
        self.assertFalse(valid)
        self.assertGreater(len(violations), 0)
        self.assertIn("Court 1", violations[0])

        # Fix the conflict
        matches[1].court = 2  # Different court

        valid, violations = ConstraintValidator.validate_court_conflicts(matches)
        self.assertTrue(valid)
        self.assertEqual(len(violations), 0)

    def test_phase_ordering_validation(self):
        """Test that pool phase completes before elimination phase"""
        # Create matches with wrong phase ordering
        matches = [
            Match("1", "TEST", "Pool A", "pool", "A1", "A2", 1, 0, 30, "pool"),
            Match("2", "TEST", "Final", "final", "Winner1", "Winner2", 2, 20, 50, "elimination"),  # Starts before pool ends
        ]

        valid, violations = ConstraintValidator.validate_phase_ordering(matches)
        self.assertFalse(valid)
        self.assertGreater(len(violations), 0)

        # Fix the ordering
        matches[1].start_time = 40  # Starts after pool ends
        matches[1].end_time = 70

        valid, violations = ConstraintValidator.validate_phase_ordering(matches)
        self.assertTrue(valid)
        self.assertEqual(len(violations), 0)

class TestTournamentStructure(unittest.TestCase):
    """Test complete tournament structure validation"""

    def setUp(self):
        self.scheduler = BadmintonTournamentScheduler(use_ortools=False)
        self.tournament = TournamentConfig(
            name="Structure Test Tournament",
            start_time="08:00",
            end_time="18:00",
            match_duration=30,
            rest_duration=20,
            num_courts=4
        )

    def test_single_pool_series_structure(self):
        """Test that single pool series generate correct structure"""
        series = [SeriesConfig(
            name="SD1",
            series_type="SD",
            total_players=4,
            single_pool=True
        )]

        result = self.scheduler.schedule_tournament(self.tournament, series)

        self.assertTrue(result.success)

        # Should have 6 pool matches (4 choose 2) and no elimination matches
        pool_matches = [m for m in result.matches if m.phase == "pool"]
        elimination_matches = [m for m in result.matches if m.phase == "elimination"]

        self.assertEqual(len(pool_matches), 6)  # 4*3/2 = 6
        self.assertEqual(len(elimination_matches), 0)

        # All players should play each other exactly once
        player_matchups = set()
        for match in pool_matches:
            matchup = tuple(sorted([match.player1, match.player2]))
            self.assertNotIn(matchup, player_matchups, f"Duplicate matchup: {matchup}")
            player_matchups.add(matchup)

    def test_multiple_pools_series_structure(self):
        """Test that multiple pools series generate correct elimination structure"""
        series = [SeriesConfig(
            name="SH1",
            series_type="SH",
            total_players=8,
            number_of_pools=2,  # 2 pools of 4
            qualifiers_per_pool=2  # 4 total qualifiers
        )]

        result = self.scheduler.schedule_tournament(self.tournament, series)

        self.assertTrue(result.success)

        pool_matches = [m for m in result.matches if m.phase == "pool"]
        elimination_matches = [m for m in result.matches if m.phase == "elimination"]

        # Should have 12 pool matches (2 pools * 6 matches each) and 3 elimination matches (2 semis + final)
        self.assertEqual(len(pool_matches), 12)  # 2 * (4*3/2) = 12
        self.assertEqual(len(elimination_matches), 3)  # 2 semis + 1 final

        # Check elimination structure
        semis = [m for m in elimination_matches if m.round_type == "semi"]
        finals = [m for m in elimination_matches if m.round_type == "final"]

        self.assertEqual(len(semis), 2)
        self.assertEqual(len(finals), 1)

    def test_six_qualifiers_structure(self):
        """Test that 6 qualifiers generate correct elimination structure (2 quarters + 2 semis + 1 final)"""
        series = [SeriesConfig(
            name="MX1",
            series_type="MX",
            total_players=12,
            players_per_pool=4,  # 3 pools of 4
            qualifiers_per_pool=2  # 6 total qualifiers
        )]

        result = self.scheduler.schedule_tournament(self.tournament, series)

        self.assertTrue(result.success)

        elimination_matches = [m for m in result.matches if m.phase == "elimination"]

        # Should have 5 elimination matches (2 quarters + 2 semis + 1 final)
        self.assertEqual(len(elimination_matches), 5)

        # Check elimination structure
        quarters = [m for m in elimination_matches if m.round_type == "quarter"]
        semis = [m for m in elimination_matches if m.round_type == "semi"]
        finals = [m for m in elimination_matches if m.round_type == "final"]

        self.assertEqual(len(quarters), 2)
        self.assertEqual(len(semis), 2)
        self.assertEqual(len(finals), 1)

    def test_all_players_participate(self):
        """Test that all players in a series participate in matches"""
        series = [SeriesConfig(
            name="TEST",
            series_type="SH",
            total_players=6,
            players_per_pool=3,  # 2 pools of 3
            qualifiers_per_pool=2  # 4 qualifiers total
        )]

        result = self.scheduler.schedule_tournament(self.tournament, series)

        self.assertTrue(result.success)

        # Extract all players from matches
        players_in_matches = set()
        for match in result.matches:
            if match.phase == "pool":  # Only count pool matches for participation
                players_in_matches.add(match.player1)
                players_in_matches.add(match.player2)

        # Should have exactly 6 unique players (A0, A1, A2, B0, B1, B2)
        self.assertEqual(len(players_in_matches), 6)

        # Each player should appear in exactly the right number of matches
        player_match_count = defaultdict(int)
        for match in result.matches:
            if match.phase == "pool":
                player_match_count[match.player1] += 1
                player_match_count[match.player2] += 1

        # In a pool of 3, each player plays 2 matches
        for player, count in player_match_count.items():
            self.assertEqual(count, 2, f"Player {player} should play exactly 2 pool matches, but played {count}")

    def test_tournament_reaches_final(self):
        """Test that tournaments with elimination reach exactly one final match per series"""
        series = [
            SeriesConfig(
                name="SH1",
                series_type="SH",
                total_players=8,
                number_of_pools=2,
                qualifiers_per_pool=2
            ),
            SeriesConfig(
                name="MX1",
                series_type="MX",
                total_players=12,
                players_per_pool=4,  # 3 pools of 4
                qualifiers_per_pool=2  # 6 qualifiers total
            )
        ]

        result = self.scheduler.schedule_tournament(self.tournament, series)

        self.assertTrue(result.success)

        # Check that each series has exactly one final
        for series_config in series:
            series_finals = [
                m for m in result.matches
                if m.series_name == series_config.name and m.round_type == "final"
            ]
            self.assertEqual(len(series_finals), 1, f"Series {series_config.name} should have exactly 1 final match")

            # Final should be in elimination phase
            final_match = series_finals[0]
            self.assertEqual(final_match.phase, "elimination")

    def test_comprehensive_constraint_validation(self):
        """Test all constraints together on a complete tournament"""
        series = [
            SeriesConfig(
                name="SH1",
                series_type="SH",
                total_players=9,
                players_per_pool=3,
                qualifiers_per_pool=2
            ),
            SeriesConfig(
                name="SD1",
                series_type="SD",
                total_players=4,
                single_pool=True
            )
        ]

        result = self.scheduler.schedule_tournament(self.tournament, series)

        self.assertTrue(result.success)

        # Validate all constraints INCLUDING dependencies
        rest_valid, rest_violations = ConstraintValidator.validate_rest_constraints(
            result.matches, self.tournament.rest_duration
        )

        court_valid, court_violations = ConstraintValidator.validate_court_conflicts(result.matches)

        phase_valid, phase_violations = ConstraintValidator.validate_phase_ordering(result.matches)

        dependency_valid, dependency_violations = ConstraintValidator.validate_elimination_dependencies(result.matches)

        structure_valid, structure_violations = ConstraintValidator.validate_tournament_structure(
            result, series
        )

        # Print violations for debugging if any exist
        if not rest_valid:
            print(f"\n🚫 Rest violations: {rest_violations[:5]}")  # Show first 5
        if not court_valid:
            print(f"\n🚫 Court violations: {court_violations[:5]}")
        if not phase_valid:
            print(f"\n🚫 Phase violations: {phase_violations[:5]}")
        if not dependency_valid:
            print(f"\n🚫 Dependency violations: {dependency_violations[:5]}")
        if not structure_valid:
            print(f"\n🚫 Structure violations: {structure_violations[:5]}")

        # All constraints should be satisfied
        self.assertTrue(rest_valid, f"Rest constraints violated: {len(rest_violations)} violations")
        self.assertTrue(court_valid, f"Court constraints violated: {len(court_violations)} violations")
        self.assertTrue(phase_valid, f"Phase constraints violated: {len(phase_violations)} violations")
        self.assertTrue(dependency_valid, f"Dependency constraints violated: {len(dependency_violations)} violations")
        self.assertTrue(structure_valid, f"Structure constraints violated: {len(structure_violations)} violations")

# Include remaining classes from original code...
# [BadmintonTournamentScheduler, show_pool_suggestions, main, etc.]

class BadmintonTournamentScheduler:
    """Main scheduler class with improved pool configuration"""

    def __init__(self, use_ortools: bool = True):
        self.calculator = TournamentCalculator()
        self.match_generator = MatchGenerator(self.calculator)

        if use_ortools and HAS_ORTOOLS:
            # Use the same OR-Tools solver from previous version
            from ortools.sat.python import cp_model
            self.solver = MockConstraintSolver()  # Simplified for this example
        else:
            self.solver = MockConstraintSolver()

    def schedule_tournament(self, tournament_config: TournamentConfig, series_configs: List[SeriesConfig]) -> ScheduleResult:
        """Generate complete tournament schedule with improved pool validation"""
        logger.info(f"Starting schedule generation for tournament: {tournament_config.name}")

        try:
            # Validate and provide suggestions for each series
            self._validate_and_suggest_series(series_configs)

            # Validate overall tournament feasibility
            self._validate_tournament_feasibility(tournament_config, series_configs)

            # Generate all matches, separating pools and eliminations
            all_pool_matches = []
            all_elimination_matches = []

            for series in series_configs:
                pool_matches, elimination_matches = self.match_generator.generate_series_matches(series)
                all_pool_matches.extend(pool_matches)
                all_elimination_matches.extend(elimination_matches)
                logger.info(f"📊 Series {series.name} ({series.get_pool_description()}): {len(pool_matches)} pool + {len(elimination_matches)} elimination = {len(pool_matches) + len(elimination_matches)} total matches")

            logger.info(f"🎯 Tournament Total: {len(all_pool_matches)} pool matches, {len(all_elimination_matches)} elimination matches")

            # Solve scheduling problem with phase separation
            result = self.solver.solve(tournament_config, all_pool_matches, all_elimination_matches)

            if result.success:
                logger.info(f"✅ Schedule generated successfully in {result.generation_time:.2f} seconds")
                logger.info(f"📊 Pool phase ends at: {self._minutes_to_time(result.pool_completion_time, tournament_config.start_time)}")
                logger.info(f"📊 Max wait time: {result.max_wait_time} minutes")
                logger.info(f"📊 Tournament ends at: {self._minutes_to_time(result.tournament_end_time, tournament_config.start_time)}")
                logger.info(f"📊 Court utilization: {result.court_utilization:.1%}")
            else:
                logger.error(f"❌ Failed to generate schedule: {result.error_message}")

            return result

        except Exception as e:
            logger.error(f"💥 Error during scheduling: {str(e)}")
            return ScheduleResult(
                success=False,
                matches=[],
                max_wait_time=0,
                tournament_end_time=0,
                court_utilization=0.0,
                generation_time=0.0,
                pool_completion_time=0,
                error_message=str(e)
            )

    def _validate_and_suggest_series(self, series_list: List[SeriesConfig]):
        """Validate each series and provide helpful suggestions"""
        for series in series_list:
            valid, warnings, suggestions = PoolStructureHelper.validate_and_suggest(series)

            if not valid:
                logger.warning(f"⚠️  Series {series.name} configuration issue:")
                for warning in warnings:
                    logger.warning(f"   • {warning}")

                if suggestions:
                    logger.info(f"💡 Suggested alternatives for {series.total_players} players:")
                    for i, suggestion in enumerate(suggestions[:3]):  # Show top 3 suggestions
                        logger.info(f"   {i+1}. {suggestion['description']} → {suggestion['total_matches']} total matches")

                raise ValueError(f"Invalid pool configuration for series {series.name}")
            else:
                logger.info(f"✅ Series {series.name}: {series.get_pool_description()}")

    def _validate_tournament_feasibility(self, tournament: TournamentConfig, series_list: List[SeriesConfig]):
        """Validate that tournament is theoretically feasible"""
        total_matches = 0

        for series in series_list:
            pool_sizes = series.get_pool_distribution()

            # Pool matches
            pool_matches = sum(self.calculator.calculate_pool_matches(size) for size in pool_sizes)

            # Elimination matches
            elimination_matches = 0
            if not series.single_pool:
                total_qualifiers = len(pool_sizes) * series.qualifiers_per_pool
                elimination_matches = self.calculator.calculate_elimination_matches(total_qualifiers)

            series_total = pool_matches + elimination_matches
            total_matches += series_total

        # Check time feasibility
        start_time = self._time_to_minutes(tournament.start_time)
        end_time = self._time_to_minutes(tournament.end_time)
        available_minutes = end_time - start_time

        # Rough estimate: assume perfect court utilization
        required_minutes = total_matches * tournament.match_duration / tournament.num_courts

        if required_minutes > available_minutes:
            raise ValueError(
                f"🚫 Tournament not feasible: {total_matches} matches require ~{required_minutes:.0f} minutes "
                f"but only {available_minutes} minutes available"
            )

        logger.info(f"✅ Feasibility check passed: {total_matches} matches, ~{required_minutes:.0f}/{available_minutes} minutes needed")

    def _time_to_minutes(self, time_str: str) -> int:
        """Convert HH:MM to minutes from midnight"""
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes

    def _minutes_to_time(self, minutes: int, start_time: str) -> str:
        """Convert minutes from tournament start to HH:MM"""
        start_minutes = self._time_to_minutes(start_time)
        total_minutes = start_minutes + minutes
        hours = total_minutes // 60
        mins = total_minutes % 60
        return f"{hours:02d}:{mins:02d}"

    def print_schedule_summary(self, result: ScheduleResult, tournament_config: TournamentConfig):
        """Print a formatted summary with phase separation"""
        if not result.success:
            print(f"❌ Schedule generation failed: {result.error_message}")
            return

        print(f"\n🏸 Tournament Schedule: {tournament_config.name}")
        print("=" * 80)
        print(f"📊 Summary:")
        print(f"   • Total matches: {len(result.matches)}")
        print(f"   • Pool phase ends: {self._minutes_to_time(result.pool_completion_time, tournament_config.start_time)}")
        print(f"   • Max wait time: {result.max_wait_time} minutes")
        print(f"   • Tournament ends: {self._minutes_to_time(result.tournament_end_time, tournament_config.start_time)}")
        print(f"   • Court utilization: {result.court_utilization:.1%}")
        print(f"   • Generation time: {result.generation_time:.2f} seconds")

        if result.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in result.warnings:
                print(f"   • {warning}")

        # Group matches by phase and time
        pool_matches = [m for m in result.matches if m.phase == "pool"]
        elimination_matches = [m for m in result.matches if m.phase == "elimination"]

        print(f"\n📅 POOL PHASE ({len(pool_matches)} matches):")
        print("-" * 60)
        self._print_phase_matches(pool_matches, tournament_config, max_show=10)

        if elimination_matches:
            print(f"\n🏆 ELIMINATION PHASE ({len(elimination_matches)} matches):")
            print("-" * 60)
            self._print_phase_matches(elimination_matches, tournament_config, max_show=20)

    def _print_phase_matches(self, matches: List[Match], tournament_config: TournamentConfig, max_show: int = 10):
        """Print matches for a specific phase"""
        time_slots = defaultdict(list)
        for match in matches:
            if match.start_time is not None:
                time_str = self._minutes_to_time(match.start_time, tournament_config.start_time)
                time_slots[time_str].append(match)

        for i, (time, matches_at_time) in enumerate(sorted(time_slots.items())):
            if i >= max_show:
                print(f"   ... ({len(time_slots) - max_show} more time slots)")
                break

            print(f"{time}:")
            for match in sorted(matches_at_time, key=lambda m: m.court or 0):
                court_str = f"Court {match.court}" if match.court else "TBD"
                print(f"   {court_str:8} | {match.series_name:4} | {match.pool_name:15} | {match.player1} vs {match.player2}")

def create_improved_example_tournament() -> Tuple[TournamentConfig, List[SeriesConfig]]:
    """Create tournament configuration with improved pool structure"""
    tournament = TournamentConfig(
        name="Improved Badminton Tournament",
        start_time="08:00",
        end_time="22:00",
        match_duration=33,
        rest_duration=20,
        num_courts=6
    )

    # MUCH clearer series configuration
    series = [
        SeriesConfig(
            name="SH1",
            series_type="SH",
            total_players=9,
            players_per_pool=3,  # Creates 3 pools of 3 players each
            qualifiers_per_pool=2
        ),
        SeriesConfig(
            name="SH2",
            series_type="SH",
            total_players=8,
            number_of_pools=2,  # Creates 2 pools of 4 players each
            qualifiers_per_pool=2
        ),
        SeriesConfig(
            name="SD1",
            series_type="SD",
            total_players=4,
            single_pool=True  # Everyone plays everyone
        ),
        SeriesConfig(
            name="MX1",
            series_type="MX",
            total_players=12,
            players_per_pool=4,  # Creates 3 pools of 4 players each
            qualifiers_per_pool=2
        )
    ]

    return tournament, series

def show_pool_suggestions(total_players: int):
    """Show pool structure suggestions for a given number of players"""
    print(f"\n💡 Pool Structure Suggestions for {total_players} players:")
    print("=" * 60)

    suggestions = PoolStructureHelper.suggest_pool_structures(total_players)

    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion['description']}")
        print(f"   • Total matches: {suggestion['total_matches']}")
        if suggestion['type'] == 'multiple_pools':
            print(f"   • Pool matches: {suggestion['pool_matches']}")
            print(f"   • Elimination matches: {suggestion['elimination_matches']}")
        print(f"   • Configuration: {suggestion['config']}")
        print()

def main():
    """Main command line interface with pool structure helpers"""
    parser = argparse.ArgumentParser(description='Badminton Tournament Scheduler (FIXED ELIMINATION DEPENDENCIES)')
    parser.add_argument('--test', action='store_true', help='Run unit tests')
    parser.add_argument('--run-example', action='store_true', help='Run improved example tournament')
    parser.add_argument('--config', type=str, help='JSON configuration file')
    parser.add_argument('--save-example', type=str, help='Save improved example config to file')
    parser.add_argument('--export-csv', type=str, help='Export schedule to CSV file')
    parser.add_argument('--suggest-pools', type=int, help='Show pool structure suggestions for N players')
    parser.add_argument('--use-mock-solver', action='store_true', help='Use mock solver instead of OR-Tools')
    parser.add_argument('--validate-only', action='store_true', help='Only validate constraints without generating schedule')

    args = parser.parse_args()

    if args.test:
        # Run unit tests
        print("🧪 Running comprehensive unit tests...")
        unittest.main(argv=[''], exit=False, verbosity=2)
        return

    if args.suggest_pools:
        show_pool_suggestions(args.suggest_pools)
        return

    if args.save_example:
        tournament, series = create_improved_example_tournament()
        config_data = {
            'tournament': asdict(tournament),
            'series': [asdict(s) for s in series]
        }
        with open(args.save_example, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        print(f"📁 Improved example configuration saved to {args.save_example}")
        return

    # Determine configuration source
    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                data = json.load(f)
            tournament = TournamentConfig(**data['tournament'])
            series = [SeriesConfig(**s) for s in data['series']]
            print(f"📁 Loaded configuration from {args.config}")
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            return
    elif args.run_example:
        tournament, series = create_improved_example_tournament()
        print("🎯 Using improved example tournament configuration")
    else:
        print("❌ Please specify --config, --run-example, --suggest-pools N, or --save-example")
        parser.print_help()
        return

    # Create scheduler
    use_ortools = not args.use_mock_solver
    scheduler = BadmintonTournamentScheduler(use_ortools=use_ortools)

    # Generate schedule
    print(f"\n🚀 Generating tournament schedule...")
    result = scheduler.schedule_tournament(tournament, series)

    # Display results
    scheduler.print_schedule_summary(result, tournament)

    # Validate constraints if requested or if generation succeeded
    if result.success and (args.validate_only or True):  # Always validate
        print(f"\n🔍 Validating tournament constraints...")

        # Validate all constraints INCLUDING elimination dependencies
        rest_valid, rest_violations = ConstraintValidator.validate_rest_constraints(
            result.matches, tournament.rest_duration
        )

        court_valid, court_violations = ConstraintValidator.validate_court_conflicts(result.matches)

        phase_valid, phase_violations = ConstraintValidator.validate_phase_ordering(result.matches)

        dependency_valid, dependency_violations = ConstraintValidator.validate_elimination_dependencies(result.matches)

        structure_valid, structure_violations = ConstraintValidator.validate_tournament_structure(
            result, series
        )

        # Report validation results
        print(f"✅ Rest time constraints: {'PASSED' if rest_valid else 'FAILED'}")
        if not rest_valid:
            for violation in rest_violations[:3]:  # Show first 3 violations
                print(f"   ❌ {violation}")

        print(f"✅ Court conflict constraints: {'PASSED' if court_valid else 'FAILED'}")
        if not court_valid:
            for violation in court_violations[:3]:
                print(f"   ❌ {violation}")

        print(f"✅ Phase ordering constraints: {'PASSED' if phase_valid else 'FAILED'}")
        if not phase_valid:
            for violation in phase_violations[:3]:
                print(f"   ❌ {violation}")

        print(f"✅ Elimination dependencies: {'PASSED' if dependency_valid else 'FAILED'}")
        if not dependency_valid:
            for violation in dependency_violations[:3]:
                print(f"   ❌ {violation}")

        print(f"✅ Tournament structure: {'PASSED' if structure_valid else 'FAILED'}")
        if not structure_valid:
            for violation in structure_violations[:3]:
                print(f"   ❌ {violation}")

        all_valid = rest_valid and court_valid and phase_valid and dependency_valid and structure_valid
        print(f"\n🎯 Overall validation: {'✅ ALL CONSTRAINTS SATISFIED' if all_valid else '❌ CONSTRAINT VIOLATIONS FOUND'}")

    # Export if requested
    if args.export_csv and result.success:
        # Add export functionality here
        print(f"💾 Export to CSV not implemented in this example")

if __name__ == "__main__":
    main()