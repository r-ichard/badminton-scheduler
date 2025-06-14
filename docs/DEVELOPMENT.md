# Development Guide

This document provides guidance for developers contributing to the Badminton Tournament Scheduler project.

## Build/Test Commands

- **Run all tests**: `python main.py --test`
- **Run example tournament**: `python main.py --run-example`
- **Run with config file**: `python main.py --config FILE.json`
- **Save example config**: `python main.py --save-example FILE.json`
- **Export schedule to CSV**: `python main.py --export-csv FILE.csv`
- **Show pool structure options**: `python main.py --suggest-pools N`
- **Format code**: `black src/ tests/`

## Code Style Guidelines

### Python Code Style

- Use **Black formatter** (v24.4.2) for consistent code style
- Use **4-space indentation** (enforced by Black)
- Use **snake_case** for variables and functions
- Use **CamelCase** for classes
- Use **triple double quotes** for docstrings
- Use **type hints** with Python typing module

### Data Models

- Use **dataclasses** for data models
- Validate input in `__post_init__` methods
- Raise `ValueError` with descriptive messages for invalid inputs
- Use `is` operator for None checks, not `==`

### Testing

- Use **unittest framework** for comprehensive unit tests
- Test all public methods and edge cases
- Mock external dependencies when necessary
- Ensure tests are deterministic and reproducible

### Logging

- Use Python's `logging` module for informational output
- Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)
- Include contextual information in log messages

## Architecture Overview

### Core Components

1. **`models.py`**: Data structures and configuration classes
2. **`scheduling.py`**: Tournament scheduling logic with OR-Tools
3. **`validation.py`**: Constraint validation and verification
4. **`cli.py`**: Command-line interface and user interactions
5. **`main.py`**: Application entry point

### Key Design Patterns

- **Dataclass Pattern**: For immutable data structures
- **Strategy Pattern**: For different pool configuration methods
- **Factory Pattern**: For creating series configurations
- **Constraint Programming**: Using OR-Tools for optimization

## Development Workflow

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/badminton-tournament-scheduler.git
cd badminton-tournament-scheduler

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests to verify setup
python main.py --test
```

### Making Changes

1. **Create a feature branch**: `git checkout -b feature/your-feature`
2. **Make your changes** following the coding standards
3. **Add/update tests** for new functionality
4. **Format code**: `black src/ tests/`
5. **Run tests**: `python main.py --test`
6. **Commit changes**: `git commit -m "Add feature description"`
7. **Push branch**: `git push origin feature/your-feature`
8. **Create Pull Request**

### Testing Guidelines

- Write tests for all new functionality
- Ensure tests pass before submitting PR
- Test edge cases and error conditions
- Use descriptive test names that explain what is being tested

Example test structure:
```python
def test_pool_structure_calculation_with_valid_input(self):
    """Test that pool structure calculation works correctly with valid input"""
    # Arrange
    config = PlayerPerPoolConfig(...)
    
    # Act
    result = config.get_pool_distribution()
    
    # Assert
    self.assertEqual(result, expected_result)
```

## OR-Tools Integration

### Constraint Programming Concepts

The scheduler uses Google OR-Tools CP-SAT solver with these key concepts:

- **Variables**: Represent match start times, court assignments
- **Constraints**: Encode tournament rules and requirements
- **Objective**: Minimize total tournament time or maximize satisfaction
- **Solver**: Finds optimal or feasible solutions

### Adding New Constraints

When adding constraints to the scheduler:

1. Define the constraint clearly in comments
2. Use appropriate OR-Tools constraint methods
3. Test the constraint with edge cases
4. Document the constraint's purpose

Example:
```python
# Constraint: Players must have minimum rest between matches
for player in players:
    for match1, match2 in consecutive_matches[player]:
        model.Add(match2_start >= match1_end + rest_duration)
```

## Common Development Tasks

### Adding New Series Types

1. Update the `series_type` validation in `models.py`
2. Add documentation to README
3. Create test cases for the new series type

### Modifying Pool Structure Logic

1. Update the relevant configuration class in `models.py`
2. Modify the `get_pool_distribution()` method
3. Update validation logic in `validation.py`
4. Add comprehensive tests

### Extending CLI Functionality

1. Add new command-line arguments in `cli.py`
2. Implement the functionality
3. Update help text and documentation
4. Add integration tests

## Performance Considerations

- OR-Tools solver performance depends on problem size and constraints
- Large tournaments (100+ matches) may take longer to solve
- Consider time limits for solver when adding new constraints
- Profile code when optimizing performance

## Debugging Tips

### Common Issues

1. **Import Errors**: Ensure all relative imports use `src.` prefix
2. **OR-Tools Not Found**: Install with `pip install ortools`
3. **Solver Timeouts**: Increase time limit or simplify constraints
4. **Validation Failures**: Check input data format and constraints

### Debugging OR-Tools Issues

```python
# Enable solver logging
solver.parameters.log_search_progress = True

# Check solver status
status = solver.Solve(model)
if status == cp_model.OPTIMAL:
    print("Solution found!")
elif status == cp_model.FEASIBLE:
    print("Feasible solution found!")
else:
    print("No solution found")
```

## Contributing Guidelines

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New functionality is tested
- [ ] Documentation is updated
- [ ] No breaking changes without discussion
- [ ] Commit messages are descriptive

### Pull Request Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
```

## Support and Questions

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Code Review**: All changes require review before merging

---

*This document is maintained by the development team and updated as the project evolves.*