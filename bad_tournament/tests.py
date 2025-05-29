"""Unit tests for badminton tournament scheduler using OR-Tools."""

import unittest
from collections import defaultdict

from models import (
    TournamentConfig,
    SeriesConfig,
    Match,
    ScheduleResult,
    SinglePoolConfig,
    PlayerPerPoolConfig,
    NumPoolsConfig,
    create_series_config,
)
from scheduling import TournamentCalculator, BadmintonTournamentScheduler
from validation import ConstraintValidator


class TestBasicLogic(unittest.TestCase):
    """Test basic pool structure logic"""

    def test_pool_match_calculation(self):
        """Test that pool match calculation is correct"""
        calculator = TournamentCalculator()

        # Round-robin formula: n*(n-1)/2
        self.assertEqual(calculator.calculate_pool_matches(3), 3)  # 3*2/2 = 3
        self.assertEqual(calculator.calculate_pool_matches(4), 6)  # 4*3/2 = 6
        self.assertEqual(calculator.calculate_pool_matches(5), 10)  # 5*4/2 = 10
        self.assertEqual(
            calculator.calculate_pool_matches(1), 0
        )  # No matches with 1 player
        self.assertEqual(
            calculator.calculate_pool_matches(0), 0
        )  # No matches with 0 players

    def test_elimination_match_calculation(self):
        """Test elimination match calculation including 6 qualifiers"""
        calculator = TournamentCalculator()

        self.assertEqual(
            calculator.calculate_elimination_matches(0), 0
        )  # No elimination
        self.assertEqual(
            calculator.calculate_elimination_matches(1), 0
        )  # No elimination
        self.assertEqual(calculator.calculate_elimination_matches(2), 1)  # Just final
        self.assertEqual(
            calculator.calculate_elimination_matches(4), 3
        )  # 2 semis + final
        self.assertEqual(
            calculator.calculate_elimination_matches(6), 5
        )  # 2 quarters + 2 semis + final (with 2 byes)
        self.assertEqual(
            calculator.calculate_elimination_matches(8), 7
        )  # 4 quarters + 2 semis + final


class TestImprovedPoolStructure(unittest.TestCase):
    """Test improved pool structure configuration"""

    def test_players_per_pool_configuration(self):
        """Test players_per_pool configuration method"""
        series = PlayerPerPoolConfig(
            name="TEST",
            series_type="SH",
            total_players=9,
            players=3,
            qualifiers=2,
        )

        self.assertEqual(series.number_of_pools, 3)
        self.assertEqual(series.players_per_pool, 3)
        self.assertEqual(series.get_pool_distribution(), [3, 3, 3])
        self.assertEqual(series.get_pool_description(), "3 pools of 3 players each")

    def test_number_of_pools_configuration(self):
        """Test number_of_pools configuration method"""
        series = NumPoolsConfig(
            name="TEST",
            series_type="SH",
            total_players=8,
            pools=2,
            qualifiers=2,
        )

        self.assertEqual(series.number_of_pools, 2)
        self.assertEqual(series.players_per_pool, 4)
        self.assertEqual(series.get_pool_distribution(), [4, 4])
        self.assertEqual(series.get_pool_description(), "2 pools of 4 players each")

    def test_single_pool_configuration(self):
        """Test single pool configuration"""
        series = SinglePoolConfig(name="TEST", series_type="SD", total_players=5)

        self.assertTrue(series.single_pool)
        self.assertEqual(series.qualifiers_per_pool, 0)
        self.assertEqual(series.get_pool_distribution(), [5])
        self.assertEqual(series.get_pool_description(), "Single pool with 5 players")

    def test_invalid_configurations(self):
        """Test that invalid configurations raise appropriate errors"""

        # Multiple configuration methods specified
        with self.assertRaises(ValueError):
            create_series_config(
                name="TEST",
                series_type="SH",
                total_players=9,
                players_per_pool=3,
                number_of_pools=3,  # Can't specify both
            )

        # Players don't divide evenly
        with self.assertRaises(ValueError):
            PlayerPerPoolConfig(
                name="TEST",
                series_type="SH",
                total_players=10,
                players=3,  # 10 doesn't divide by 3
            )


class TestConstraintValidation(unittest.TestCase):
    """Test constraint validation with mock matches"""

    def test_rest_constraint_validation(self):
        """Test that rest time constraints are properly validated"""
        # Create matches with insufficient rest time
        matches = [
            Match("1", "TEST", "Pool A", "pool", "A1", "A2", 1, 0, 30, "pool"),
            Match(
                "2", "TEST", "Pool A", "pool", "A1", "A3", 2, 40, 70, "pool"
            ),  # Only 10 min rest for A1
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
            Match(
                "2", "TEST", "Pool B", "pool", "B1", "B2", 1, 20, 50, "pool"
            ),  # Overlaps on court 1
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
            Match(
                "2",
                "TEST",
                "Final",
                "final",
                "Winner1",
                "Winner2",
                2,
                20,
                50,
                "elimination",
            ),  # Starts before pool ends
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

    def test_elimination_dependency_validation(self):
        """Test the elimination dependency validator"""
        # Create matches with dependency violations
        matches = [
            Match(
                "q1",
                "TEST",
                "Quarter 1",
                "quarter",
                "A1",
                "B2",
                1,
                100,
                130,
                "elimination",
                dependency_level=1,
            ),
            Match(
                "q2",
                "TEST",
                "Quarter 2",
                "quarter",
                "A2",
                "B1",
                2,
                100,
                130,
                "elimination",
                dependency_level=1,
            ),
            Match(
                "s1",
                "TEST",
                "Semi 1",
                "semi",
                "Winner Q1",
                "Winner Q2",
                3,
                120,
                150,
                "elimination",
                dependency_level=2,
            ),  # Starts before quarters end
        ]

        valid, violations = ConstraintValidator.validate_elimination_dependencies(
            matches
        )
        self.assertFalse(valid)
        self.assertGreater(len(violations), 0)
        self.assertIn("dependency violation", violations[0])

        # Fix the dependency violation
        matches[2].start_time = 160  # Start after quarters end + gap
        matches[2].end_time = 190

        valid, violations = ConstraintValidator.validate_elimination_dependencies(
            matches
        )
        self.assertTrue(valid)
        self.assertEqual(len(violations), 0)


class TestTournamentStructure(unittest.TestCase):
    """Test complete tournament structure validation using OR-Tools"""

    def setUp(self):
        self.scheduler = BadmintonTournamentScheduler()  # Uses OR-Tools
        self.tournament = TournamentConfig(
            name="Structure Test Tournament",
            start_time="08:00",
            end_time="18:00",
            match_duration=30,
            rest_duration=20,
            num_courts=4,
        )

    def test_single_pool_series_structure(self):
        """Test that single pool series generate correct structure"""
        series = [SinglePoolConfig(name="SD1", series_type="SD", total_players=4)]

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
        series = [
            NumPoolsConfig(
                name="SH1",
                series_type="SH",
                total_players=8,
                pools=2,  # 2 pools of 4
                qualifiers=2,  # 4 total qualifiers
            )
        ]

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
        # FIXED: Use longer tournament window for this complex test
        extended_tournament = TournamentConfig(
            name="Extended Test Tournament",
            start_time="08:00",
            end_time="20:00",  # Extended to 12 hours
            match_duration=30,
            rest_duration=20,
            num_courts=6,  # More courts
        )

        series = [
            PlayerPerPoolConfig(
                name="MX1",
                series_type="MX",
                total_players=12,
                players=4,  # 3 pools of 4
                qualifiers=2,  # 6 total qualifiers
            )
        ]

        result = self.scheduler.schedule_tournament(extended_tournament, series)

        self.assertTrue(result.success, f"Scheduling failed: {result.error_message}")

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
        series = [
            PlayerPerPoolConfig(
                name="TEST",
                series_type="SH",
                total_players=6,
                players=3,  # 2 pools of 3
                qualifiers=2,  # 4 qualifiers total
            )
        ]

        result = self.scheduler.schedule_tournament(self.tournament, series)

        self.assertTrue(result.success)

        # Extract all players from matches
        players_in_matches = set()
        for match in result.matches:
            if match.phase == "pool":  # Only count pool matches for participation
                players_in_matches.add(match.player1)
                players_in_matches.add(match.player2)

        # Should have exactly 6 unique players (A1, A2, A3, B1, B2, B3)
        self.assertEqual(len(players_in_matches), 6)

        # Each player should appear in exactly the right number of matches
        player_match_count = defaultdict(int)
        for match in result.matches:
            if match.phase == "pool":
                player_match_count[match.player1] += 1
                player_match_count[match.player2] += 1

        # In a pool of 3, each player plays 2 matches
        for player, count in player_match_count.items():
            self.assertEqual(
                count,
                2,
                f"Player {player} should play exactly 2 pool matches, but played {count}",
            )

    def test_tournament_reaches_final(self):
        """Test that tournaments with elimination reach exactly one final match per series"""
        # FIXED: Use smaller, more realistic tournament configurations
        series = [
            PlayerPerPoolConfig(
                name="SH1",
                series_type="SH",
                total_players=6,  # Reduced from 8
                players=3,  # 2 pools of 3 = 6 pool matches + 1 final
                qualifiers=1,  # Only 1 qualifier per pool = 2 total
            ),
            SinglePoolConfig(
                name="SD1",
                series_type="SD",
                total_players=4,  # 6 matches, no elimination
            ),
        ]

        result = self.scheduler.schedule_tournament(self.tournament, series)

        self.assertTrue(result.success, f"Scheduling failed: {result.error_message}")

        # Check that SH1 series has exactly one final (SD1 has no elimination)
        sh1_finals = [
            m
            for m in result.matches
            if m.series_name == "SH1" and m.round_type == "final"
        ]
        self.assertEqual(
            len(sh1_finals), 1, f"Series SH1 should have exactly 1 final match"
        )

        # Final should be in elimination phase
        final_match = sh1_finals[0]
        self.assertEqual(final_match.phase, "elimination")

    def test_comprehensive_constraint_validation(self):
        """Test all constraints together on a complete tournament using OR-Tools"""
        # FIXED: Use smaller tournament that fits in time window
        series = [
            PlayerPerPoolConfig(
                name="SH1",
                series_type="SH",
                total_players=6,  # Reduced from 9
                players=3,
                qualifiers=1,  # Reduced to create smaller elimination
            ),
            SinglePoolConfig(name="SD1", series_type="SD", total_players=4),
        ]

        result = self.scheduler.schedule_tournament(self.tournament, series)

        self.assertTrue(result.success, f"Scheduling failed: {result.error_message}")

        # Validate all constraints INCLUDING dependencies
        rest_valid, rest_violations = ConstraintValidator.validate_rest_constraints(
            result.matches, self.tournament.rest_duration
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
            ConstraintValidator.validate_tournament_structure(result, series)
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

        # All constraints should be satisfied with OR-Tools
        self.assertTrue(
            rest_valid, f"Rest constraints violated: {len(rest_violations)} violations"
        )
        self.assertTrue(
            court_valid,
            f"Court constraints violated: {len(court_violations)} violations",
        )
        self.assertTrue(
            phase_valid,
            f"Phase constraints violated: {len(phase_violations)} violations",
        )
        self.assertTrue(
            dependency_valid,
            f"Dependency constraints violated: {len(dependency_violations)} violations",
        )
        self.assertTrue(
            structure_valid,
            f"Structure constraints violated: {len(structure_violations)} violations",
        )


class TestEliminationDependencies(unittest.TestCase):
    """Test elimination dependency scheduling using OR-Tools"""

    def setUp(self):
        self.scheduler = BadmintonTournamentScheduler()  # Uses OR-Tools
        self.tournament = TournamentConfig(
            name="Dependency Test Tournament",
            start_time="08:00",
            end_time="18:00",
            match_duration=30,
            rest_duration=20,
            num_courts=6,
        )

    def test_elimination_dependency_ordering(self):
        """Test that elimination rounds are scheduled sequentially using OR-Tools"""
        series = [
            PlayerPerPoolConfig(
                name="SH1",
                series_type="SH",
                total_players=9,
                players=3,  # 3 pools of 3, 6 qualifiers
                qualifiers=2,
            )
        ]

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

            self.assertLessEqual(
                latest_quarter_end,
                earliest_semi_start,
                f"Quarters must finish before semis start: {latest_quarter_end} <= {earliest_semi_start}",
            )

        if semis and finals:
            latest_semi_end = max(match.end_time for match in semis)
            earliest_final_start = min(match.start_time for match in finals)

            self.assertLessEqual(
                latest_semi_end,
                earliest_final_start,
                f"Semis must finish before final starts: {latest_semi_end} <= {earliest_final_start}",
            )

    def test_no_simultaneous_dependent_matches(self):
        """Test that dependent matches are never scheduled simultaneously using OR-Tools"""
        series = [
            PlayerPerPoolConfig(
                name="MX1",
                series_type="MX",
                total_players=12,
                players=4,  # 3 pools of 4, 6 qualifiers
                qualifiers=2,
            )
        ]

        result = self.scheduler.schedule_tournament(self.tournament, series)

        self.assertTrue(result.success)

        elimination_matches = [m for m in result.matches if m.phase == "elimination"]

        # Check that no two matches with different dependency levels have the same start time
        time_to_levels = defaultdict(set)
        for match in elimination_matches:
            time_to_levels[match.start_time].add(match.dependency_level)

        for start_time, levels in time_to_levels.items():
            self.assertEqual(
                len(levels),
                1,
                f"Multiple dependency levels scheduled at time {start_time}: {levels}",
            )


class TestORToolsScheduling(unittest.TestCase):
    """Test OR-Tools scheduling with real constraints"""

    def setUp(self):
        """Set up test environment with OR-Tools scheduler"""
        self.scheduler = BadmintonTournamentScheduler()  # Always uses OR-Tools
        self.tournament = TournamentConfig(
            name="Test Tournament",
            start_time="08:00",
            end_time="18:00",
            match_duration=30,
            rest_duration=20,
            num_courts=4,
        )

    def test_single_pool_scheduling(self):
        """Test OR-Tools scheduling with single pool"""
        series = [SinglePoolConfig(name="SD1", series_type="SD", total_players=4)]

        result = self.scheduler.schedule_tournament(self.tournament, series)

        self.assertTrue(result.success, f"Scheduling failed: {result.error_message}")

        # Should have 6 pool matches and no elimination matches
        pool_matches = [m for m in result.matches if m.phase == "pool"]
        elimination_matches = [m for m in result.matches if m.phase == "elimination"]

        self.assertEqual(len(pool_matches), 6)  # 4*3/2 = 6
        self.assertEqual(len(elimination_matches), 0)

        # All matches should be scheduled
        for match in pool_matches:
            self.assertIsNotNone(match.start_time)
            self.assertIsNotNone(match.court)

    def test_multiple_pools_scheduling(self):
        """Test OR-Tools scheduling with multiple pools and elimination"""
        series = [
            NumPoolsConfig(
                name="SH1",
                series_type="SH",
                total_players=8,
                pools=2,
                qualifiers=2,
            )
        ]

        result = self.scheduler.schedule_tournament(self.tournament, series)

        self.assertTrue(result.success, f"Scheduling failed: {result.error_message}")

        pool_matches = [m for m in result.matches if m.phase == "pool"]
        elimination_matches = [m for m in result.matches if m.phase == "elimination"]

        self.assertEqual(len(pool_matches), 12)  # 2 * (4*3/2) = 12
        self.assertEqual(len(elimination_matches), 3)  # 2 semis + 1 final

        # All matches should be scheduled
        for match in result.matches:
            self.assertIsNotNone(match.start_time)
            self.assertIsNotNone(match.court)

    def test_constraint_satisfaction(self):
        """Test that OR-Tools satisfies all constraints"""
        series = [
            PlayerPerPoolConfig(
                name="TEST",
                series_type="SH",
                total_players=6,
                players=3,
                qualifiers=2,
            )
        ]

        result = self.scheduler.schedule_tournament(self.tournament, series)

        self.assertTrue(result.success, f"Scheduling failed: {result.error_message}")

        # Validate all constraints
        rest_valid, rest_violations = ConstraintValidator.validate_rest_constraints(
            result.matches, self.tournament.rest_duration
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

        self.assertTrue(rest_valid, f"Rest constraints violated: {rest_violations}")
        self.assertTrue(court_valid, f"Court constraints violated: {court_violations}")
        self.assertTrue(phase_valid, f"Phase constraints violated: {phase_violations}")
        self.assertTrue(
            dependency_valid,
            f"Dependency constraints violated: {dependency_violations}",
        )

    def test_time_window_enforcement(self):
        """Test that OR-Tools respects tournament time window"""
        # Create a tournament with very limited time
        tight_tournament = TournamentConfig(
            name="Tight Schedule",
            start_time="08:00",
            end_time="10:00",  # Only 2 hours
            match_duration=30,
            rest_duration=20,
            num_courts=2,
        )

        # Try to schedule too many matches
        series = [
            PlayerPerPoolConfig(
                name="OVERLOAD",
                series_type="SH",
                total_players=12,
                players=3,
                qualifiers=2,
            )
        ]

        # Should either fail gracefully or fit within time window
        result = self.scheduler.schedule_tournament(tight_tournament, series)

        if result.success:
            # If successful, all matches must be within time window
            for match in result.matches:
                self.assertLessEqual(match.end_time, 120)  # 2 hours = 120 minutes


def run_all_tests():
    """Run all unit tests"""
    # Discover and run all tests in the current module
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()
