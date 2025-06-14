"""Constraint validation for tournament schedules."""

from typing import List, Tuple
from collections import defaultdict

from src.models import (
    Match,
    ScheduleResult,
    SeriesConfig,
    SinglePoolConfig,
    PlayerPerPoolConfig,
    NumPoolsConfig,
)
from src.scheduling import PoolStructureHelper


class ConstraintValidator:
    """Helper class to validate tournament constraints"""

    @staticmethod
    def validate_rest_constraints(
        matches: List[Match], min_rest_minutes: int
    ) -> Tuple[bool, List[str]]:
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
        """Validate that pool matches complete before elimination matches start within each series"""
        violations = []

        # Group matches by series for series-specific validation
        series_matches = defaultdict(lambda: {"pool": [], "elimination": []})

        for match in matches:
            if match.phase == "pool" and match.end_time is not None:
                series_matches[match.series_name]["pool"].append(match)
            elif match.phase == "elimination" and match.start_time is not None:
                series_matches[match.series_name]["elimination"].append(match)

        # Check phase ordering within each series
        for series_name, matches_by_phase in series_matches.items():
            pool_matches = matches_by_phase["pool"]
            elimination_matches = matches_by_phase["elimination"]

            if pool_matches and elimination_matches:
                latest_pool_end = max(match.end_time for match in pool_matches)
                earliest_elim_start = min(
                    match.start_time for match in elimination_matches
                )

                if latest_pool_end > earliest_elim_start:
                    violations.append(
                        f"Phase ordering violation in {series_name}: pool phase ends at {latest_pool_end} "
                        f"but elimination phase starts at {earliest_elim_start}"
                    )

        return len(violations) == 0, violations

    @staticmethod
    def validate_elimination_dependencies(
        matches: List[Match],
    ) -> Tuple[bool, List[str]]:
        """Validate that elimination dependencies are respected"""
        violations = []

        elimination_matches = [
            m for m in matches if m.phase == "elimination" and m.start_time is not None
        ]

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
    def validate_tournament_structure(
        result: ScheduleResult, series_configs: List[SeriesConfig]
    ) -> Tuple[bool, List[str]]:
        """Validate overall tournament structure"""
        violations = []

        # Check that each series has proper structure
        for series in series_configs:
            series_matches = [m for m in result.matches if m.series_name == series.name]
            pool_matches = [m for m in series_matches if m.phase == "pool"]
            elimination_matches = [
                m for m in series_matches if m.phase == "elimination"
            ]

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
                    expected_elim_matches = (
                        PoolStructureHelper._calculate_elimination_matches(
                            total_qualifiers
                        )
                    )

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

    @staticmethod
    def validate_all_constraints(
        result: ScheduleResult, tournament_config, series_configs: List[SeriesConfig]
    ) -> Tuple[bool, List[str]]:
        """Validate all constraints at once"""
        all_violations = []

        # Validate all constraint types
        rest_valid, rest_violations = ConstraintValidator.validate_rest_constraints(
            result.matches, tournament_config.rest_duration
        )
        court_valid, court_violations = ConstraintValidator.validate_court_conflicts(
            result.matches
        )
        phase_valid, phase_violations = ConstraintValidator.validate_phase_ordering(
            result.matches
        )
        dependency_valid, dependency_violations = (
            ConstraintValidator.validate_elimination_dependencies(result.matches)
        )
        structure_valid, structure_violations = (
            ConstraintValidator.validate_tournament_structure(result, series_configs)
        )

        # Collect all violations
        all_violations.extend(rest_violations)
        all_violations.extend(court_violations)
        all_violations.extend(phase_violations)
        all_violations.extend(dependency_violations)
        all_violations.extend(structure_violations)

        overall_valid = (
            rest_valid
            and court_valid
            and phase_valid
            and dependency_valid
            and structure_valid
        )

        return overall_valid, all_violations
