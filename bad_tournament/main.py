#!/usr/bin/env python3
"""
Badminton Tournament Scheduler - OR-Tools Only Implementation
Refactored into modular structure for better maintainability.

Entry point for the badminton tournament scheduler.
"""

try:
    from ortools.sat.python import cp_model

    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False
    print(
        "❌ OR-Tools is required for tournament scheduling. Install with: pip install ortools"
    )
    exit(1)

if __name__ == "__main__":
    from cli import main

    main()
