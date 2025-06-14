# Tournament Configuration JSON Format Specification

This document describes the expected JSON format for badminton tournament configurations based on the tournament scheduling system's code logic and test requirements.

## Root Structure

The JSON file must contain two main sections:

```json
{
  "tournament": { ... },
  "series": [ ... ]
}
```

## Tournament Configuration (`tournament`)

The tournament section defines the overall tournament parameters.

**Required Fields:**

- `name` (string): Tournament name
- `start_time` (string): Tournament start time in "HH:MM" format (24-hour)
- `end_time` (string): Tournament end time in "HH:MM" format (24-hour)
- `match_duration` (integer): Duration of each match in minutes (must be > 0)
- `rest_duration` (integer): Minimum rest time between matches in minutes (must be ≥ 20)
- `num_courts` (integer): Number of available courts (must be > 0)

**Validation Rules:**
- `start_time` must be before `end_time`
- `match_duration` must be positive
- `rest_duration` must be at least 20 minutes
- `num_courts` must be positive

**Example:**
```json
{
  "tournament": {
    "name": "GRAND PRIX VILLE D'OULLINS 2022",
    "start_time": "08:00",
    "end_time": "19:00",
    "match_duration": 30,
    "rest_duration": 20,
    "num_courts": 30
  }
}
```

## Series Configuration (`series`)

The series section is an array of series configurations. Each series represents a tournament category (e.g., Men's Singles, Women's Doubles).

### Common Fields (All Series Types)

**Required Fields:**
- `name` (string): Unique series identifier (e.g., "SH1", "DD2")
- `series_type` (string): Type of series - valid values: "SH", "SD", "MX", "DH", "DD"
- `total_players` (integer): Total number of players/teams in the series (must be ≥ 2 for single pool, ≥ 4 for multiple pools)

### Pool Configuration Methods

Each series must specify exactly **ONE** of the following pool configuration methods:

#### Method 1: Single Pool (`single_pool`)

For series where all players play in one pool (round-robin).

**Required Field:**
- `single_pool` (boolean): Must be `true`

**Restrictions:**
- No qualifiers advance (everyone plays everyone)
- Minimum 2 players required

**Example:**
```json
{
  "name": "SH9",
  "series_type": "SH",
  "total_players": 3,
  "single_pool": true
}
```

#### Method 2: Players Per Pool (`players_per_pool`)

Specify the number of players in each pool.

**Required Fields:**
- `players_per_pool` (integer): Number of players per pool (must be ≥ 2)
- `qualifiers_per_pool` (integer): Number of players who advance from each pool (0 ≤ qualifiers ≤ players_per_pool)

**Validation Rules:**
- `total_players` must be evenly divisible by `players_per_pool`
- `qualifiers_per_pool` cannot exceed `players_per_pool`

**Example:**
```json
{
  "name": "SH1",
  "series_type": "SH",
  "total_players": 12,
  "players_per_pool": 3,
  "qualifiers_per_pool": 2
}
```

#### Method 3: Number of Pools (`number_of_pools`)

Specify the total number of pools.

**Required Fields:**
- `number_of_pools` (integer): Total number of pools (must be ≥ 1)
- `qualifiers_per_pool` (integer): Number of players who advance from each pool

**Alternative Field Name:**
- `pools` (integer): Can be used instead of `number_of_pools` (as seen in test.json)

**Validation Rules:**
- `total_players` must be evenly divisible by `number_of_pools`
- All pools will have equal size
- `qualifiers_per_pool` cannot exceed players per pool

**Example:**
```json
{
  "name": "SH10",
  "series_type": "SH",
  "total_players": 8,
  "number_of_pools": 2,
  "qualifiers_per_pool": 2
}
```

**Alternative format:**
```json
{
  "name": "MX1",
  "series_type": "MX",
  "total_players": 12,
  "pools": 2,
  "qualifiers_per_pool": 1
}
```

## Series Types

The following series types are supported:

- **SH**: Singles Men/Men's Singles
- **SD**: Singles Women/Women's Singles  
- **DH**: Doubles Men/Men's Doubles
- **DD**: Doubles Women/Women's Doubles
- **MX**: Mixed Doubles

## Elimination Structure

Based on the number of qualifiers from pools, the system automatically generates elimination rounds:

- **2 qualifiers**: 1 final match
- **4 qualifiers**: 2 semi-finals + 1 final
- **6 qualifiers**: 2 quarter-finals + 2 semi-finals + 1 final (with 2 byes)
- **8 qualifiers**: 4 quarter-finals + 2 semi-finals + 1 final

## Validation Rules Summary

### Tournament Level:
- Start time must be before end time
- Match duration must be positive
- Rest duration must be at least 20 minutes
- Number of courts must be positive

### Series Level:
- Each series must have a unique name
- Exactly one pool configuration method must be specified
- Total players must be compatible with chosen pool structure
- Qualifiers per pool cannot exceed players per pool

### Scheduling Constraints:
- Players must have minimum rest time between matches
- No two matches can be scheduled on the same court simultaneously
- Pool matches must complete before elimination matches start (per series)
- Elimination rounds must be scheduled sequentially (quarters → semis → finals)
- All matches must fit within tournament time window

## Complete Example

```json
{
  "tournament": {
    "name": "Badminton Championship 2024",
    "start_time": "08:00",
    "end_time": "18:00",
    "match_duration": 30,
    "rest_duration": 20,
    "num_courts": 8
  },
  "series": [
    {
      "name": "SH1",
      "series_type": "SH",
      "total_players": 12,
      "players_per_pool": 3,
      "qualifiers_per_pool": 2
    },
    {
      "name": "SD1",
      "series_type": "SD",
      "total_players": 8,
      "number_of_pools": 2,
      "qualifiers_per_pool": 2
    },
    {
      "name": "MX1",
      "series_type": "MX",
      "total_players": 4,
      "single_pool": true
    }
  ]
}
```

## Error Conditions

The system will reject configurations that:

1. Specify multiple pool configuration methods
2. Have players that don't divide evenly into pools
3. Have more qualifiers than players per pool
4. Have invalid time ranges
5. Have insufficient tournament duration for all matches
6. Have invalid series types
7. Have duplicate series names

This specification ensures that tournament configurations are valid and can be successfully processed by the scheduling system.