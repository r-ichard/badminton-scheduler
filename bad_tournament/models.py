"""Data models for badminton tournament scheduling."""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Literal
from datetime import datetime


@dataclass
class TournamentConfig:
    """Tournament configuration parameters"""

    name: str
    start_time: str  # Format: "HH:MM"
    end_time: str  # Format: "HH:MM"
    match_duration: int  # minutes
    rest_duration: int  # minutes (minimum 20)
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
class BaseSeriesConfig:
    """Base series configuration with common fields"""

    name: str
    series_type: str  # "SH", "SD", "MX", "DH", "DD"
    total_players: int

    def get_pool_distribution(self) -> List[int]:
        """Get the actual pool sizes (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement get_pool_distribution")

    def get_pool_description(self) -> str:
        """Get human-readable description of pool structure (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement get_pool_description")

    @property
    def single_pool(self) -> bool:
        """Compatibility property for existing code"""
        return isinstance(self, SinglePoolConfig)

    @property
    def number_of_pools(self) -> Optional[int]:
        """Compatibility property for existing code"""
        if isinstance(self, PlayerPerPoolConfig):
            return self.total_players // self.players_per_pool
        elif isinstance(self, NumPoolsConfig):
            return self.pools
        return None

    @property
    def players_per_pool(self) -> Optional[int]:
        """Compatibility property for existing code"""
        if isinstance(self, PlayerPerPoolConfig):
            return self.players
        elif isinstance(self, NumPoolsConfig):
            return self.total_players // self.pools
        return None

    @property
    def qualifiers_per_pool(self) -> int:
        """Compatibility property for existing code"""
        if isinstance(self, SinglePoolConfig):
            return 0
        elif isinstance(self, PlayerPerPoolConfig):
            return self.qualifiers
        elif isinstance(self, NumPoolsConfig):
            return self.qualifiers
        return 0


@dataclass
class SinglePoolConfig(BaseSeriesConfig):
    """Series configuration with a single pool (everyone plays everyone)"""

    config_type: Literal["single_pool"] = "single_pool"

    def __post_init__(self):
        """Validate single pool configuration"""
        if self.total_players < 2:
            raise ValueError("Single pool must have at least 2 players")

    def get_pool_distribution(self) -> List[int]:
        """Get the actual pool sizes"""
        return [self.total_players]

    def get_pool_description(self) -> str:
        """Get human-readable description of pool structure"""
        return f"Single pool with {self.total_players} players"


@dataclass
class PlayerPerPoolConfig(BaseSeriesConfig):
    """Series configuration with a fixed number of players per pool"""

    players: int  # Number of players per pool
    qualifiers: int = 2  # How many advance from each pool
    config_type: Literal["players_per_pool"] = "players_per_pool"

    def __post_init__(self):
        """Validate players per pool configuration"""
        if self.players < 2:
            raise ValueError("Players per pool must be at least 2")
        if self.total_players % self.players != 0:
            raise ValueError(
                f"Cannot divide {self.total_players} players into pools of {self.players}"
            )
        if self.qualifiers < 0 or self.qualifiers > self.players:
            raise ValueError(
                f"Qualifiers per pool must be between 0 and {self.players}"
            )

    def get_pool_distribution(self) -> List[int]:
        """Get the actual pool sizes"""
        num_pools = self.total_players // self.players
        return [self.players] * num_pools

    def get_pool_description(self) -> str:
        """Get human-readable description of pool structure"""
        num_pools = self.total_players // self.players
        return f"{num_pools} pools of {self.players} players each"


@dataclass
class NumPoolsConfig(BaseSeriesConfig):
    """Series configuration with a fixed number of pools"""

    pools: int  # Number of pools
    qualifiers: int = 2  # How many advance from each pool
    config_type: Literal["number_of_pools"] = "number_of_pools"

    def __post_init__(self):
        """Validate number of pools configuration"""
        if self.pools < 1:
            raise ValueError("Number of pools must be at least 1")
        if self.total_players % self.pools != 0:
            raise ValueError(
                f"Cannot divide {self.total_players} players into {self.pools} equal pools"
            )
        players_per_pool = self.total_players // self.pools
        if self.qualifiers < 0 or self.qualifiers > players_per_pool:
            raise ValueError(
                f"Qualifiers per pool must be between 0 and {players_per_pool}"
            )

    def get_pool_distribution(self) -> List[int]:
        """Get the actual pool sizes"""
        players_per_pool = self.total_players // self.pools
        return [players_per_pool] * self.pools

    def get_pool_description(self) -> str:
        """Get human-readable description of pool structure"""
        players_per_pool = self.total_players // self.pools
        return f"{self.pools} pools of {players_per_pool} players each"


# Type alias for the union of all series configuration types
SeriesConfig = Union[SinglePoolConfig, PlayerPerPoolConfig, NumPoolsConfig]


def create_series_config(
    name: str,
    series_type: str,
    total_players: int,
    players_per_pool: Optional[int] = None,
    number_of_pools: Optional[int] = None,
    single_pool: bool = False,
    qualifiers_per_pool: int = 2,
) -> SeriesConfig:
    """Factory function to create the appropriate SeriesConfig subclass based on the parameters provided"""
    config_count = sum(
        [players_per_pool is not None, number_of_pools is not None, single_pool]
    )
    if config_count != 1:
        raise ValueError(
            "Specify exactly ONE pool configuration: players_per_pool, number_of_pools, or single_pool=True"
        )

    if single_pool:
        return SinglePoolConfig(
            name=name, series_type=series_type, total_players=total_players
        )
    elif players_per_pool is not None:
        return PlayerPerPoolConfig(
            name=name,
            series_type=series_type,
            total_players=total_players,
            players=players_per_pool,
            qualifiers=qualifiers_per_pool,
        )
    elif number_of_pools is not None:
        return NumPoolsConfig(
            name=name,
            series_type=series_type,
            total_players=total_players,
            pools=number_of_pools,
            qualifiers=qualifiers_per_pool,
        )

    # This should never happen due to the config_count check above
    raise ValueError("Invalid configuration")


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
        time_str = (
            f"{self.start_time//60:02d}:{self.start_time%60:02d}"
            if self.start_time
            else "TBD"
        )
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
