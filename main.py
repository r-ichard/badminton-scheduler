#!/usr/bin/env python3
"""
Badminton Tournament Scheduler
Entry point for the tournament scheduling system.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False
    print("❌ OR-Tools is required for tournament scheduling.")
    print("Install with: pip install ortools")
    sys.exit(1)

if __name__ == "__main__":
    from src.cli import main
    main()