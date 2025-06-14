# 🏸 Badminton Tournament Scheduler

NB : This is a work in progress, some logic and scheduling can be improved. 


A sophisticated tournament scheduling system for badminton competitions using **Google OR-Tools** constraint programming. This system efficiently schedules complex tournament structures with multiple series, pool play, and elimination rounds while satisfying all tournament constraints.

## ✨ Features

- **🔄 Multiple Tournament Formats**: Support for single pools, multiple pools with elimination brackets
- **⚡ Advanced Constraint Satisfaction**: Uses Google OR-Tools CP-SAT solver for optimal scheduling
- **🎯 Multiple Series Support**: Handle different categories (Singles, Doubles, Mixed) simultaneously
- **🏊 Flexible Pool Structures**: Configure pools by players per pool or number of pools
- **✅ Comprehensive Validation**: Ensures all tournament rules and constraints are satisfied
- **⏰ Time Window Management**: Respects tournament start/end times and minimum rest periods
- **🏟️ Court Optimization**: Maximizes court utilization while avoiding conflicts

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OR-Tools library

### Installation

```bash
# Clone the repository
https://github.com/r-ichard/badminton-scheduler
cd badminton-scheduler

# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
python main.py --test

# Generate and run an example tournament
python main.py --run-example
```

## 📋 Usage

### Command Line Interface

```bash
# Run unit tests
python main.py --test

# Run example tournament
python main.py --run-example

# Use custom configuration file
python main.py --config tournament.json

# Save example configuration
python main.py --save-example example.json

# Get pool structure suggestions
python main.py --suggest-pools 12

# Export schedule to CSV
python main.py --export-csv schedule.csv
```

### Configuration Format

Tournaments are defined using JSON configuration files. See [`docs/JSON_FORMAT_SPECIFICATION.md`](docs/JSON_FORMAT_SPECIFICATION.md) for complete documentation.

#### Basic Example

```json
{
  "tournament": {
    "name": "Club Championship 2024",
    "start_time": "08:00",
    "end_time": "18:00",
    "match_duration": 30,
    "rest_duration": 20,
    "num_courts": 4
  },
  "series": [
    {
      "name": "SH1",
      "series_type": "SH",
      "total_players": 8,
      "number_of_pools": 2,
      "qualifiers_per_pool": 2
    }
  ]
}
```

## 🏗️ Tournament Structure

### Series Types
- **SH**: Men's Singles
- **SD**: Women's Singles  
- **DH**: Men's Doubles
- **DD**: Women's Doubles
- **MX**: Mixed Doubles

### Pool Configuration Methods

1. **Single Pool**: All players in one round-robin pool
2. **Players Per Pool**: Specify number of players in each pool
3. **Number of Pools**: Specify total number of pools

### Elimination Brackets

The system automatically generates appropriate elimination structures:
- **2 qualifiers**: Final only
- **4 qualifiers**: Semi-finals + Final
- **6 qualifiers**: Quarter-finals + Semi-finals + Final (with 2 byes)
- **8 qualifiers**: Quarter-finals + Semi-finals + Final

## ⚙️ Technical Details

### Constraint Satisfaction

The scheduler uses Google OR-Tools CP-SAT solver to satisfy multiple constraints:

1. **Rest Time Constraints**: Players must have minimum rest between matches
2. **Court Conflict Prevention**: No two matches on same court simultaneously  
3. **Phase Ordering**: Pool play completes before elimination rounds
4. **Elimination Dependencies**: Tournament rounds proceed sequentially
5. **Time Window Enforcement**: All matches within tournament hours

### Architecture

```
src/
├── models.py          # Data models and tournament configuration
├── scheduling.py      # Core scheduling logic and OR-Tools solver
├── validation.py      # Constraint validation and verification
├── cli.py            # Command-line interface and example configurations
└── main.py           # Entry point

tests/
└── test_tournament.py # Comprehensive unit test suite

examples/
├── tournament_example.json
└── bacothon16.json

docs/
├── JSON_FORMAT_SPECIFICATION.md
└── DEVELOPMENT.md
```

## 🧪 Testing

Run the complete test suite:

```bash
python main.py --test
```

Tests cover:
- Pool structure calculations
- Constraint validation
- Tournament generation
- OR-Tools solver integration
- Elimination bracket logic

## 📈 Performance

The OR-Tools implementation provides:
- **⚡ Optimal Solutions**: Mathematically proven optimal schedules when possible
- **🚀 Fast Solving**: Typically solves tournaments in seconds
- **📊 Scalability**: Handles tournaments with 100+ matches
- **🔧 Flexibility**: Adapts to various tournament formats and constraints

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Format code
black src/ tests/

# Run tests
python main.py --test
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google OR-Tools**: Constraint programming framework
- **Badminton Community**: Tournament format specifications and requirements

## 📚 Documentation

- [`docs/JSON_FORMAT_SPECIFICATION.md`](docs/JSON_FORMAT_SPECIFICATION.md): Complete configuration format documentation
- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md): Development guidelines and code style



---

**Made with ❤️ for the badminton community**