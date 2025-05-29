"""Core scheduling logic including match generation and OR-Tools solver."""

import time as time_module
import logging
from typing import List, Tuple, Dict, Any
from collections import defaultdict
from itertools import combinations

try:
    from ortools.sat.python import cp_model

    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False

from models import TournamentConfig, SeriesConfig, Match, ScheduleResult

logger = logging.getLogger(__name__)


class PoolStructureHelper:
    """Helper class for pool structure calculations and suggestions"""

    @staticmethod
    def suggest_pool_structures(total_players: int) -> List[Dict[str, Any]]:
        """Suggest good pool structures for a given number of players"""
        suggestions = []

        # Single pool option (always available)
        total_matches = total_players * (total_players - 1) // 2
        suggestions.append(
            {
                "type": "single_pool",
                "description": f"Single pool ({total_players} players)",
                "total_matches": total_matches,
                "pools": 1,
                "players_per_pool": total_players,
                "config": {"single_pool": True},
            }
        )

        # Multiple pool options
        for pool_size in range(3, min(6, total_players + 1)):  # Pool sizes 3-5
            if total_players % pool_size == 0:
                num_pools = total_players // pool_size
                pool_matches = num_pools * (pool_size * (pool_size - 1) // 2)

                # Calculate elimination matches (assuming 2 qualifiers per pool)
                if num_pools > 1:
                    total_qualifiers = num_pools * 2
                    elim_matches = PoolStructureHelper._calculate_elimination_matches(
                        total_qualifiers
                    )
                else:
                    elim_matches = 0

                suggestions.append(
                    {
                        "type": "multiple_pools",
                        "description": f"{num_pools} pools of {pool_size} players",
                        "total_matches": pool_matches + elim_matches,
                        "pool_matches": pool_matches,
                        "elimination_matches": elim_matches,
                        "pools": num_pools,
                        "players_per_pool": pool_size,
                        "config": {
                            "players_per_pool": pool_size,
                            "qualifiers_per_pool": 2,
                        },
                    }
                )

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
    def validate_and_suggest(
        series_config: SeriesConfig,
    ) -> Tuple[bool, List[str], List[Dict]]:
        """Validate pool configuration and provide suggestions if invalid"""
        warnings = []
        suggestions = []

        try:
            # Try to validate the current configuration
            pool_dist = series_config.get_pool_distribution()
            return True, warnings, suggestions

        except ValueError as e:
            # Configuration is invalid, provide suggestions
            suggestions = PoolStructureHelper.suggest_pool_structures(
                series_config.total_players
            )
            return False, [str(e)], suggestions


class TournamentCalculator:
    """Calculate matches and tournament structure"""

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
    """Generate matches for tournament series with dependency tracking"""

    def __init__(self, calculator: TournamentCalculator):
        self.calculator = calculator
        self.match_counter = 0

    def generate_series_matches(
        self, series: SeriesConfig
    ) -> Tuple[List[Match], List[Match]]:
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

    def _generate_pool_matches(
        self, series: SeriesConfig, pool_sizes: List[int]
    ) -> List[Match]:
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
                        dependency_level=0,
                    )
                    matches.append(match)
                    self.match_counter += 1

        return matches

    def _generate_elimination_matches(
        self, series: SeriesConfig, pool_sizes: List[int]
    ) -> List[Match]:
        """Generate elimination bracket matches with dependency levels"""
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
                dependency_level=1,
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
                dependency_level=1,
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
                dependency_level=1,
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
                dependency_level=2,
            )
            matches.append(final)
            self.match_counter += 1

        elif total_qualifiers == 6:
            # Handle 6 qualifiers properly (2 quarters + 2 semis + 1 final with 2 byes)
            quarter1 = Match(
                id=f"{series.name}_Quarter1_{self.match_counter}",
                series_name=series.name,
                pool_name="Quarter-final 1",
                round_type="quarter",
                player1="2nd Pool A",
                player2="2nd Pool C",
                phase="elimination",
                dependency="all_pools_complete",
                dependency_level=1,
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
                dependency_level=1,
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
                dependency_level=2,
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
                dependency_level=2,
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
                dependency_level=3,
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
                    dependency_level=1,
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
                dependency_level=2,
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
                dependency_level=2,
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
                dependency_level=3,
            )
            matches.append(final)
            self.match_counter += 1

        return matches


class ORToolsConstraintSolver:
    """OR-Tools based constraint solver for tournament scheduling"""

    def solve(
        self,
        tournament: TournamentConfig,
        pool_matches: List[Match],
        elimination_matches: List[Match],
    ) -> ScheduleResult:
        """Solve scheduling problem using OR-Tools CP-SAT solver"""
        start_time = time_module.time()

        try:
            # Convert time to minutes
            start_minutes = self._time_to_minutes(tournament.start_time)
            end_minutes = self._time_to_minutes(tournament.end_time)

            # Combine all matches
            all_matches = pool_matches + elimination_matches

            # Create CP model
            model = cp_model.CpModel()

            # Create time slots (every match_duration minutes)
            time_slots = list(
                range(
                    start_minutes,
                    end_minutes - tournament.match_duration + 1,
                    tournament.match_duration,
                )
            )

            # Variables for each match
            match_vars = []
            for idx, match in enumerate(all_matches):
                time_var = model.NewIntVarFromDomain(
                    cp_model.Domain.FromValues(time_slots), f"time_{idx}"
                )
                court_var = model.NewIntVar(1, tournament.num_courts, f"court_{idx}")
                match_vars.append((time_var, court_var))

            # Map players to their matches
            player_to_matches = defaultdict(list)
            for idx, match in enumerate(all_matches):
                player_to_matches[match.player1].append(idx)
                player_to_matches[match.player2].append(idx)

            # Constraint 1: Rest time between matches for same player
            for player, match_indices in player_to_matches.items():
                for i, j in combinations(match_indices, 2):
                    time_i, _ = match_vars[i]
                    time_j, _ = match_vars[j]

                    # Either match i finishes before match j starts (with rest) or vice versa
                    rest1 = model.NewBoolVar(f"rest_{player}_{i}_{j}_1")
                    rest2 = model.NewBoolVar(f"rest_{player}_{i}_{j}_2")

                    model.Add(
                        time_i + tournament.match_duration + tournament.rest_duration
                        <= time_j
                    ).OnlyEnforceIf(rest1)
                    model.Add(
                        time_j + tournament.match_duration + tournament.rest_duration
                        <= time_i
                    ).OnlyEnforceIf(rest2)
                    model.AddBoolOr([rest1, rest2])

            # Constraint 2: No court conflicts (no two matches on same court at same time)
            for i in range(len(all_matches)):
                for j in range(i + 1, len(all_matches)):
                    time_i, court_i = match_vars[i]
                    time_j, court_j = match_vars[j]

                    # If same court, then different times
                    same_court = model.NewBoolVar(f"same_court_{i}_{j}")
                    model.Add(court_i == court_j).OnlyEnforceIf(same_court)
                    model.Add(court_i != court_j).OnlyEnforceIf(same_court.Not())
                    model.Add(time_i != time_j).OnlyEnforceIf(same_court)

            # Constraint 3: Phase ordering - all pools before elimination
            pool_indices = [
                i for i, match in enumerate(all_matches) if match.phase == "pool"
            ]
            elimination_indices = [
                i for i, match in enumerate(all_matches) if match.phase == "elimination"
            ]

            if pool_indices and elimination_indices:
                # All pool matches must finish before any elimination match starts
                for pool_idx in pool_indices:
                    for elim_idx in elimination_indices:
                        time_pool, _ = match_vars[pool_idx]
                        time_elim, _ = match_vars[elim_idx]
                        model.Add(
                            time_pool
                            + tournament.match_duration
                            + tournament.rest_duration
                            <= time_elim
                        )

            # Constraint 4: Elimination dependency ordering
            elimination_by_level = defaultdict(list)
            for idx in elimination_indices:
                level = all_matches[idx].dependency_level
                elimination_by_level[level].append(idx)

            # Each level must complete before next level starts
            for level in sorted(elimination_by_level.keys())[:-1]:
                current_level_indices = elimination_by_level[level]
                next_level_indices = elimination_by_level.get(level + 1, [])

                for curr_idx in current_level_indices:
                    for next_idx in next_level_indices:
                        time_curr, _ = match_vars[curr_idx]
                        time_next, _ = match_vars[next_idx]
                        model.Add(
                            time_curr
                            + tournament.match_duration
                            + tournament.rest_duration
                            <= time_next
                        )

            # Objective: Minimize tournament end time and wait times
            # Calculate tournament end time
            end_times = []
            for time_var, _ in match_vars:
                end_time = model.NewIntVar(
                    start_minutes, end_minutes, f"end_{time_var.Name()}"
                )
                model.Add(end_time == time_var + tournament.match_duration)
                end_times.append(end_time)

            max_end_time = model.NewIntVar(start_minutes, end_minutes, "max_end_time")
            model.AddMaxEquality(max_end_time, end_times)

            # Calculate wait times
            wait_times = []
            for player, match_indices in player_to_matches.items():
                if len(match_indices) > 1:
                    for i, j in combinations(sorted(match_indices), 2):
                        time_i, _ = match_vars[i]
                        time_j, _ = match_vars[j]

                        wait_time = model.NewIntVar(
                            0, end_minutes - start_minutes, f"wait_{player}_{i}_{j}"
                        )
                        model.Add(
                            wait_time == time_j - time_i - tournament.match_duration
                        )
                        wait_times.append(wait_time)

            max_wait_time = model.NewIntVar(
                0, end_minutes - start_minutes, "max_wait_time"
            )
            if wait_times:
                model.AddMaxEquality(max_wait_time, wait_times)
            else:
                model.Add(max_wait_time == 0)

            # Multi-objective: minimize both tournament duration and max wait time
            # Weight tournament end time more heavily to prefer shorter tournaments
            model.Minimize(max_end_time + 10 * max_wait_time)

            # Solve
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 300.0  # 5 minute timeout
            status = solver.Solve(model)

            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                # Extract solution
                for idx, (time_var, court_var) in enumerate(match_vars):
                    match = all_matches[idx]
                    match.start_time = (
                        solver.Value(time_var) - start_minutes
                    )  # Minutes from tournament start
                    match.end_time = match.start_time + tournament.match_duration
                    match.court = solver.Value(court_var)

                # Calculate metrics
                pool_completion = max(
                    (match.end_time for match in pool_matches), default=0
                )
                tournament_end = max(
                    (match.end_time for match in all_matches), default=0
                )
                max_wait = solver.Value(max_wait_time) if wait_times else 0

                # Calculate court utilization
                total_match_time = len(all_matches) * tournament.match_duration
                total_available_time = tournament.num_courts * tournament_end
                court_utilization = (
                    total_match_time / total_available_time
                    if total_available_time > 0
                    else 0
                )

                generation_time = time_module.time() - start_time

                return ScheduleResult(
                    success=True,
                    matches=all_matches,
                    max_wait_time=max_wait,
                    tournament_end_time=tournament_end,
                    court_utilization=court_utilization,
                    generation_time=generation_time,
                    pool_completion_time=pool_completion,
                    warnings=[],
                )
            else:
                return ScheduleResult(
                    success=False,
                    matches=[],
                    max_wait_time=0,
                    tournament_end_time=0,
                    court_utilization=0.0,
                    generation_time=time_module.time() - start_time,
                    pool_completion_time=0,
                    error_message=f"OR-Tools solver failed with status: {solver.StatusName(status)}",
                )

        except Exception as e:
            return ScheduleResult(
                success=False,
                matches=[],
                max_wait_time=0,
                tournament_end_time=0,
                court_utilization=0.0,
                generation_time=time_module.time() - start_time,
                pool_completion_time=0,
                error_message=f"Solver error: {str(e)}",
            )

    def _time_to_minutes(self, time_str: str) -> int:
        """Convert HH:MM to minutes from midnight"""
        hours, minutes = map(int, time_str.split(":"))
        return hours * 60 + minutes


class BadmintonTournamentScheduler:
    """Main scheduler class using OR-Tools"""

    def __init__(self):
        """Initialize scheduler with OR-Tools solver"""
        if not HAS_ORTOOLS:
            raise ImportError("OR-Tools is required for tournament scheduling")

        self.calculator = TournamentCalculator()
        self.match_generator = MatchGenerator(self.calculator)
        self.solver = ORToolsConstraintSolver()

    def schedule_tournament(
        self, tournament_config: TournamentConfig, series_configs: List[SeriesConfig]
    ) -> ScheduleResult:
        """Generate complete tournament schedule"""
        logger.info(
            f"Starting schedule generation for tournament: {tournament_config.name}"
        )

        try:
            # Validate and provide suggestions for each series
            self._validate_and_suggest_series(series_configs)

            # Validate overall tournament feasibility
            self._validate_tournament_feasibility(tournament_config, series_configs)

            # Generate all matches, separating pools and eliminations
            all_pool_matches = []
            all_elimination_matches = []

            for series in series_configs:
                pool_matches, elimination_matches = (
                    self.match_generator.generate_series_matches(series)
                )
                all_pool_matches.extend(pool_matches)
                all_elimination_matches.extend(elimination_matches)
                logger.info(
                    f"📊 Series {series.name} ({series.get_pool_description()}): {len(pool_matches)} pool + {len(elimination_matches)} elimination = {len(pool_matches) + len(elimination_matches)} total matches"
                )

            logger.info(
                f"🎯 Tournament Total: {len(all_pool_matches)} pool matches, {len(all_elimination_matches)} elimination matches"
            )

            # Solve scheduling problem
            result = self.solver.solve(
                tournament_config, all_pool_matches, all_elimination_matches
            )

            if result.success:
                logger.info(
                    f"✅ Schedule generated successfully in {result.generation_time:.2f} seconds"
                )
                logger.info(
                    f"📊 Pool phase ends at: {self._minutes_to_time(result.pool_completion_time, tournament_config.start_time)}"
                )
                logger.info(f"📊 Max wait time: {result.max_wait_time} minutes")
                logger.info(
                    f"📊 Tournament ends at: {self._minutes_to_time(result.tournament_end_time, tournament_config.start_time)}"
                )
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
                error_message=str(e),
            )

    def _validate_and_suggest_series(self, series_list: List[SeriesConfig]):
        """Validate each series and provide helpful suggestions"""
        for series in series_list:
            valid, warnings, suggestions = PoolStructureHelper.validate_and_suggest(
                series
            )

            if not valid:
                logger.warning(f"⚠️  Series {series.name} configuration issue:")
                for warning in warnings:
                    logger.warning(f"   • {warning}")

                if suggestions:
                    logger.info(
                        f"💡 Suggested alternatives for {series.total_players} players:"
                    )
                    for i, suggestion in enumerate(
                        suggestions[:3]
                    ):  # Show top 3 suggestions
                        logger.info(
                            f"   {i+1}. {suggestion['description']} → {suggestion['total_matches']} total matches"
                        )

                raise ValueError(f"Invalid pool configuration for series {series.name}")
            else:
                logger.info(f"✅ Series {series.name}: {series.get_pool_description()}")

    def _validate_tournament_feasibility(
        self, tournament: TournamentConfig, series_list: List[SeriesConfig]
    ):
        """Validate that tournament is theoretically feasible"""
        total_matches = 0

        for series in series_list:
            pool_sizes = series.get_pool_distribution()

            # Pool matches
            pool_matches = sum(
                self.calculator.calculate_pool_matches(size) for size in pool_sizes
            )

            # Elimination matches
            elimination_matches = 0
            if not series.single_pool:
                total_qualifiers = len(pool_sizes) * series.qualifiers_per_pool
                elimination_matches = self.calculator.calculate_elimination_matches(
                    total_qualifiers
                )

            series_total = pool_matches + elimination_matches
            total_matches += series_total

        # Check time feasibility with more realistic estimates
        start_time = self._time_to_minutes(tournament.start_time)
        end_time = self._time_to_minutes(tournament.end_time)
        available_minutes = end_time - start_time

        # More realistic estimate: account for scheduling inefficiencies
        # Assume 70% efficiency due to rest periods and dependencies
        required_minutes = (
            total_matches * tournament.match_duration / tournament.num_courts
        ) / 0.7

        if required_minutes > available_minutes:
            raise ValueError(
                f"🚫 Tournament not feasible: {total_matches} matches require ~{required_minutes:.0f} minutes "
                f"but only {available_minutes} minutes available"
            )

        logger.info(
            f"✅ Feasibility check passed: {total_matches} matches, ~{required_minutes:.0f}/{available_minutes} minutes needed"
        )

    def _time_to_minutes(self, time_str: str) -> int:
        """Convert HH:MM to minutes from midnight"""
        hours, minutes = map(int, time_str.split(":"))
        return hours * 60 + minutes

    def _minutes_to_time(self, minutes: int, start_time: str) -> str:
        """Convert minutes from tournament start to HH:MM"""
        start_minutes = self._time_to_minutes(start_time)
        total_minutes = start_minutes + minutes
        hours = total_minutes // 60
        mins = total_minutes % 60
        return f"{hours:02d}:{mins:02d}"

    def print_schedule_summary(
        self, result: ScheduleResult, tournament_config: TournamentConfig
    ):
        """Print a formatted summary with phase separation"""
        if not result.success:
            print(f"❌ Schedule generation failed: {result.error_message}")
            return

        print(f"\n🏸 Tournament Schedule: {tournament_config.name}")
        print("=" * 80)
        print(f"📊 Summary:")
        print(f"   • Total matches: {len(result.matches)}")
        print(
            f"   • Pool phase ends: {self._minutes_to_time(result.pool_completion_time, tournament_config.start_time)}"
        )
        print(f"   • Max wait time: {result.max_wait_time} minutes")
        print(
            f"   • Tournament ends: {self._minutes_to_time(result.tournament_end_time, tournament_config.start_time)}"
        )
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
            self._print_phase_matches(
                elimination_matches, tournament_config, max_show=20
            )

    def _print_phase_matches(
        self,
        matches: List[Match],
        tournament_config: TournamentConfig,
        max_show: int = 10,
    ):
        """Print matches for a specific phase"""
        time_slots = defaultdict(list)
        for match in matches:
            if match.start_time is not None:
                time_str = self._minutes_to_time(
                    match.start_time, tournament_config.start_time
                )
                time_slots[time_str].append(match)

        for i, (time, matches_at_time) in enumerate(sorted(time_slots.items())):
            if i >= max_show:
                print(f"   ... ({len(time_slots) - max_show} more time slots)")
                break

            print(f"{time}:")
            for match in sorted(matches_at_time, key=lambda m: m.court or 0):
                court_str = f"Court {match.court}" if match.court else "TBD"
                print(
                    f"   {court_str:8} | {match.series_name:4} | {match.pool_name:15} | {match.player1} vs {match.player2}"
                )
