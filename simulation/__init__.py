"""
Fleet Shift Analyzer - Historical Simulation Module

This module provides tools for backtesting shift analysis algorithms
on historical data to validate predictions and alert generation.
"""

from .simulation_engine import SimulationEngine, SimulationPoint, SimulationResult

__all__ = ['SimulationEngine', 'SimulationPoint', 'SimulationResult']