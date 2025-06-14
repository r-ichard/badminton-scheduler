"""Command line interface and example configurations."""

import argparse
import json
import logging
from dataclasses import asdict
from typing import Tuple, List, cast, Dict, Any

from src.models import (
    TournamentConfig,
    SeriesConfig,
    SinglePoolConfig,
    PlayerPerPoolConfig,
    NumPoolsConfig,
    BaseSeriesConfig,
    create_series_config,
)
from src.scheduling import BadmintonTournamentScheduler, PoolStructureHelper
from src.validation import ConstraintValidator
def run_all_tests():
    """Run all unit tests with proper import handling"""
    import sys
    import os
    import unittest
    
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Import and run tests
    try:
        from tests.test_tournament import (
            TestBasicLogic, TestImprovedPoolStructure, TestConstraintValidation,
            TestTournamentStructure, TestEliminationDependencies, TestORToolsScheduling
        )
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add all test classes
        for test_class in [TestBasicLogic, TestImprovedPoolStructure, TestConstraintValidation,
                          TestTournamentStructure, TestEliminationDependencies, TestORToolsScheduling]:
            suite.addTests(loader.loadTestsFromTestCase(test_class))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result.wasSuccessful()
    except ImportError as e:
        print(f"❌ Error importing tests: {e}")
        return False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_improved_example_tournament() -> Tuple[TournamentConfig, List[SeriesConfig]]:
    """Create tournament configuration with improved pool structure and feasibility constraints"""
    import random

    tournament = TournamentConfig(
        name="OR-Tools Badminton Tournament",
        start_time="08:00",
        end_time="22:00",  # 14 hours = 840 minutes
        match_duration=33,
        rest_duration=20,
        num_courts=6,
    )

    # Calculate available time and rough capacity
    available_minutes = 14 * 60  # 840 minutes
    court_capacity = (
        tournament.num_courts * available_minutes // tournament.match_duration
    )
    # With 70% efficiency due to rest periods and dependencies
    realistic_capacity = int(court_capacity * 0.7)  # ~106 matches max

    # Generate randomized but feasible series configurations
    random.seed(42)  # For reproducible results

    series = []
    total_matches = 0

    # Series 1: Small SH series
    sh1_players = random.choice([6, 9])  # 6 or 9 players
    if sh1_players == 6:
        sh1_config = PlayerPerPoolConfig(
            name="SH1",
            series_type="SH",
            total_players=sh1_players,
            players=3,  # 2 pools of 3
            qualifiers=2,
        )
        sh1_matches = 6 + 3  # 6 pool + 3 elimination (2 semis + 1 final)
    else:  # 9 players
        sh1_config = PlayerPerPoolConfig(
            name="SH1",
            series_type="SH",
            total_players=sh1_players,
            players=3,  # 3 pools of 3
            qualifiers=2,
        )
        sh1_matches = 9 + 5  # 9 pool + 5 elimination (2 quarters + 2 semis + 1 final)

    series.append(sh1_config)
    total_matches += sh1_matches

    # Series 2: Medium SH series (only if we have capacity)
    if total_matches + 15 < realistic_capacity:
        sh2_players = random.choice([8, 12])
        if sh2_players == 8:
            sh2_config = NumPoolsConfig(
                name="SH2",
                series_type="SH",
                total_players=sh2_players,
                pools=2,  # 2 pools of 4
                qualifiers=2,
            )
            sh2_matches = 12 + 3  # 12 pool + 3 elimination
        else:  # 12 players
            sh2_config = PlayerPerPoolConfig(
                name="SH2",
                series_type="SH",
                total_players=sh2_players,
                players=4,  # 3 pools of 4
                qualifiers=1,  # Reduced to keep feasible
            )
            sh2_matches = 18 + 1  # 18 pool + 1 final

        series.append(sh2_config)
        total_matches += sh2_matches

    # Series 3: Single pool SD series (always feasible)
    if total_matches + 10 < realistic_capacity:
        sd1_players = random.choice([4, 5, 6])
        sd1_config = SinglePoolConfig(
            name="SD1", series_type="SD", total_players=sd1_players
        )
        sd1_matches = sd1_players * (sd1_players - 1) // 2  # Round-robin

        series.append(sd1_config)
        total_matches += sd1_matches

    # Series 4: MX series (only if significant capacity remains)
    if total_matches + 20 < realistic_capacity:
        mx1_players = random.choice([8, 12])
        if mx1_players == 8:
            mx1_config = NumPoolsConfig(
                name="MX1",
                series_type="MX",
                total_players=mx1_players,
                pools=2,  # 2 pools of 4
                qualifiers=1,  # Reduced to keep feasible
            )
            mx1_matches = 12 + 1  # 12 pool + 1 final
        else:  # 12 players
            mx1_config = PlayerPerPoolConfig(
                name="MX1",
                series_type="MX",
                total_players=mx1_players,
                players=6,  # 2 pools of 6
                qualifiers=1,  # Just a final
            )
            mx1_matches = 30 + 1  # 30 pool + 1 final

        series.append(mx1_config)
        total_matches += mx1_matches

    # Fallback: If no series were added due to capacity constraints
    if not series:
        # Create minimal feasible tournament
        series = [
            SinglePoolConfig(name="SD1", series_type="SD", total_players=4),
            PlayerPerPoolConfig(
                name="SH1",
                series_type="SH",
                total_players=6,
                players=3,
                qualifiers=2,
            ),
        ]

    # Log the configuration for debugging
    print(
        f"🎯 Generated tournament with {len(series)} series, estimated {total_matches} matches"
    )
    print(
        f"📊 Capacity: {realistic_capacity} matches, Usage: {total_matches/realistic_capacity:.1%}"
    )

    return tournament, series


def create_feasible_test_tournament() -> Tuple[TournamentConfig, List[SeriesConfig]]:
    """Create a smaller, guaranteed feasible tournament for testing"""
    tournament = TournamentConfig(
        name="Test Tournament",
        start_time="08:00",
        end_time="18:00",  # 10 hours
        match_duration=30,
        rest_duration=20,
        num_courts=4,
    )

    # Small, guaranteed feasible series
    series = [
        SinglePoolConfig(name="SD1", series_type="SD", total_players=4),  # 6 matches
        PlayerPerPoolConfig(
            name="SH1",
            series_type="SH",
            total_players=6,
            players=3,  # 2 pools of 3
            qualifiers=1,  # Only 1 final match
        ),
    ]

    return tournament, series


def validate_tournament_feasibility(
    tournament: TournamentConfig, series: List[SeriesConfig]
) -> Tuple[bool, str]:
    """Validate that a tournament configuration is theoretically feasible"""
    total_matches = 0

    for s in series:
        pool_sizes = s.get_pool_distribution()

        # Pool matches
        pool_matches = sum((size * (size - 1) // 2) for size in pool_sizes)

        # Elimination matches
        elim_matches = 0
        if not s.single_pool and s.qualifiers_per_pool > 0:
            total_qualifiers = len(pool_sizes) * s.qualifiers_per_pool
            if total_qualifiers == 2:
                elim_matches = 1
            elif total_qualifiers <= 4:
                elim_matches = 3
            elif total_qualifiers == 6:
                elim_matches = 5
            elif total_qualifiers <= 8:
                elim_matches = 7

        total_matches += pool_matches + elim_matches

    # Check feasibility
    available_minutes = (
        int(tournament.end_time.split(":")[0])
        - int(tournament.start_time.split(":")[0])
    ) * 60

    # Realistic capacity with 60% efficiency (more conservative)
    realistic_capacity = int(
        (tournament.num_courts * available_minutes / tournament.match_duration) * 0.6
    )

    if total_matches > realistic_capacity:
        return (
            False,
            f"Too many matches: {total_matches} > {realistic_capacity} capacity",
        )

    return True, f"Feasible: {total_matches}/{realistic_capacity} matches"


def show_pool_suggestions(total_players: int):
    """Show pool structure suggestions for a given number of players"""
    print(f"\n💡 Pool Structure Suggestions for {total_players} players:")
    print("=" * 60)

    suggestions = PoolStructureHelper.suggest_pool_structures(total_players)

    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion['description']}")
        print(f"   • Total matches: {suggestion['total_matches']}")
        if suggestion["type"] == "multiple_pools":
            print(f"   • Pool matches: {suggestion['pool_matches']}")
            print(f"   • Elimination matches: {suggestion['elimination_matches']}")
        print(f"   • Configuration: {suggestion['config']}")
        print()


def main():
    """Main command line interface using OR-Tools"""
    parser = argparse.ArgumentParser(
        description="Badminton Tournament Scheduler (OR-Tools Only)"
    )
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument(
        "--run-example", action="store_true", help="Run example tournament"
    )
    parser.add_argument("--config", type=str, help="JSON configuration file")
    parser.add_argument("--save-example", type=str, help="Save example config to file")
    parser.add_argument("--export-csv", type=str, help="Export schedule to CSV file")
    parser.add_argument(
        "--suggest-pools",
        type=int,
        help="Show pool structure suggestions for N players",
    )

    args = parser.parse_args()

    if args.test:
        print("🧪 Running OR-Tools unit tests...")
        run_all_tests()
        return

    if args.suggest_pools:
        show_pool_suggestions(args.suggest_pools)
        return

    if args.save_example:

        def serialize_series_config(config: SeriesConfig) -> Dict[str, Any]:
            """Serialize a SeriesConfig object to a dictionary for JSON export"""
            result = {
                "name": config.name,
                "series_type": config.series_type,
                "total_players": config.total_players,
            }

            # Add configuration-specific fields
            if isinstance(config, SinglePoolConfig):
                result["single_pool"] = True
            elif isinstance(config, PlayerPerPoolConfig):
                result["players_per_pool"] = config.players
                result["qualifiers_per_pool"] = config.qualifiers
            elif isinstance(config, NumPoolsConfig):
                result["number_of_pools"] = config.pools
                result["qualifiers_per_pool"] = config.qualifiers

            return result

        tournament, series = create_improved_example_tournament()
        config_data = {
            "tournament": asdict(tournament),
            "series": [serialize_series_config(s) for s in series],
        }
        with open(args.save_example, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        print(f"📁 Example configuration saved to {args.save_example}")
        return

    # Determine configuration source
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                data = json.load(f)
            tournament = TournamentConfig(**data["tournament"])

            # Convert legacy SeriesConfig format to new discriminated union types
            series = []
            for s in data["series"]:
                # Extract common parameters
                name = s["name"]
                series_type = s["series_type"]
                total_players = s["total_players"]

                # Determine which type to create based on configuration
                if s.get("single_pool", False):
                    series.append(
                        SinglePoolConfig(
                            name=name,
                            series_type=series_type,
                            total_players=total_players,
                        )
                    )
                elif "players_per_pool" in s and s["players_per_pool"] is not None:
                    series.append(
                        PlayerPerPoolConfig(
                            name=name,
                            series_type=series_type,
                            total_players=total_players,
                            players=s["players_per_pool"],
                            qualifiers=s.get("qualifiers_per_pool", 2),
                        )
                    )
                elif "number_of_pools" in s and s["number_of_pools"] is not None:
                    series.append(
                        NumPoolsConfig(
                            name=name,
                            series_type=series_type,
                            total_players=total_players,
                            pools=s["number_of_pools"],
                            qualifiers=s.get("qualifiers_per_pool", 2),
                        )
                    )

            print(f"📁 Loaded configuration from {args.config}")
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            return
    elif args.run_example:
        tournament, series = create_improved_example_tournament()

        # Validate feasibility
        is_feasible, message = validate_tournament_feasibility(tournament, series)
        print(f"🎯 Using OR-Tools example tournament configuration")
        print(f"📊 Feasibility check: {message}")

        if not is_feasible:
            print(
                "⚠️  Configuration may be challenging - switching to guaranteed feasible version"
            )
            tournament, series = create_feasible_test_tournament()
    else:
        print(
            "❌ Please specify --config, --run-example, --suggest-pools N, or --save-example"
        )
        parser.print_help()
        return

    # Create scheduler (always uses OR-Tools)
    try:
        scheduler = BadmintonTournamentScheduler()
    except ImportError as e:
        print(f"❌ {e}")
        return

    # Generate schedule
    print(f"\n🚀 Generating tournament schedule using OR-Tools...")
    result = scheduler.schedule_tournament(tournament, series)

    # Display results
    scheduler.print_schedule_summary(result, tournament)

    # Always validate constraints
    if result.success:
        print(f"\n🔍 Validating tournament constraints...")

        # Validate all constraints
        rest_valid, rest_violations = ConstraintValidator.validate_rest_constraints(
            result.matches, tournament.rest_duration
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

        # Report validation results
        print(f"✅ Rest time constraints: {'PASSED' if rest_valid else 'FAILED'}")
        if not rest_valid:
            for violation in rest_violations[:3]:
                print(f"   ❌ {violation}")

        print(f"✅ Court conflict constraints: {'PASSED' if court_valid else 'FAILED'}")
        if not court_valid:
            for violation in court_violations[:3]:
                print(f"   ❌ {violation}")

        print(f"✅ Phase ordering constraints: {'PASSED' if phase_valid else 'FAILED'}")
        if not phase_valid:
            for violation in phase_violations[:3]:
                print(f"   ❌ {violation}")

        print(
            f"✅ Elimination dependencies: {'PASSED' if dependency_valid else 'FAILED'}"
        )
        if not dependency_valid:
            for violation in dependency_violations[:3]:
                print(f"   ❌ {violation}")

        print(f"✅ Tournament structure: {'PASSED' if structure_valid else 'FAILED'}")
        if not structure_valid:
            for violation in structure_violations[:3]:
                print(f"   ❌ {violation}")

        all_valid = (
            rest_valid
            and court_valid
            and phase_valid
            and dependency_valid
            and structure_valid
        )
        print(
            f"\n🎯 Overall validation: {'✅ ALL CONSTRAINTS SATISFIED' if all_valid else '❌ CONSTRAINT VIOLATIONS FOUND'}"
        )

    # Export if requested
    if args.export_csv and result.success:
        print(f"💾 CSV export functionality can be added here")
