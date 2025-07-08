# Fleet Shift Analysis - Historical Simulation Module

This module provides tools for backtesting the shift analysis algorithms on historical data to validate predictions and alert generation accuracy.

## Overview

The simulation framework allows you to:
- Run the shift analysis algorithms on historical data as if in real-time
- Track when alerts would have been generated
- Compare predictions against actual outcomes
- Calculate accuracy metrics (precision, recall, F1 score)
- Visualize results to identify patterns and areas for improvement

## Components

### 1. `simulation_engine.py`
Core simulation engine that:
- Processes historical data in chronological order
- Runs analysis at configurable time intervals
- Tracks predictions and actual outcomes
- Calculates performance metrics

### 2. `run_simulation.py`
Command-line interface for running simulations with options for:
- Time period selection
- Analysis frequency (hourly, half-hourly, etc.)
- Vehicle selection (all or specific vehicles)
- Shift configuration parameters

### 3. `generate_charts.py`
Generates charts from simulation results:
- Alert timeline charts
- Accuracy metrics
- Vehicle performance comparisons
- Confusion matrices

## Caching API Responses

To speed up debugging and development, the simulation engine supports caching of API responses. This is especially useful when:
- Running multiple simulations on the same date range
- Debugging simulation logic without waiting for API calls
- Testing different algorithm parameters on the same data

### Enabling Cache

```bash
# Use default cache directory (./simulation_cache)
python simulation/run_simulation.py --fleet-id 104002 --start-date 2025-06-17 --end-date 2025-06-17

# Specify custom cache directory
python simulation/run_simulation.py --fleet-id 104002 --start-date 2025-06-17 --end-date 2025-06-17 \
    --cache-dir ./my_cache_folder

# Disable cache (force API calls)
python simulation/run_simulation.py --fleet-id 104002 --start-date 2025-06-17 --end-date 2025-06-17 \
    --no-cache
```

### Cache Behavior

- **First run**: Fetches data from API and saves to cache
- **Subsequent runs**: Loads data from cache files (instant)
- **Cache key**: Based on fleet ID, vehicle ID, start date, and end date
- **Cache format**: JSON files with descriptive names
- **Cache location**: Default is `./simulation_cache/`

### Managing Cache

```bash
# Clear all cache
rm -rf ./simulation_cache/

# Clear cache for specific date range
rm ./simulation_cache/fleet_104002_vehicle_*_20250617_*.json

# View cache size
du -sh ./simulation_cache/
```

### Cache File Naming

Cache files are named with this pattern:
```
fleet_{fleet_id}_vehicle_{vehicle_id}_{start_date}_{start_time}_to_{end_date}_{end_time}.json
```

Example:
```
fleet_104002_vehicle_12345_20250616_000000_to_20250618_000000.json
```

## Usage

### Running a Simulation

```bash
# Simulate a full week with hourly analysis
python simulation/run_simulation.py \
    --fleet-id 104002 \
    --start-date 2025-06-10 \
    --end-date 2025-06-17 \
    --timezone Australia/Perth \
    --interval-hours 1.0 \
    --output-dir ./simulation_results/week_sim

# Simulate a single day with 30-minute intervals
python simulation/run_simulation.py \
    --fleet-id 104002 \
    --start-date "2025-06-17 06:00:00" \
    --end-date "2025-06-17 18:00:00" \
    --timezone Australia/Perth \
    --interval-hours 0.5 \
    --output-dir ./simulation_results/day_sim

# Simulate specific vehicles only
python simulation/run_simulation.py \
    --fleet-id 104002 \
    --start-date 2025-06-15 \
    --end-date 2025-06-16 \
    --vehicles 12345 \
    --vehicles 67890 \
    --output-dir ./simulation_results/specific_vehicles
```

### Generating Charts

```bash
# Generate charts from simulation results
python simulation/generate_charts.py \
    --simulation-id sim_20250617_143022 \
    --results-dir ./simulation_results/week_sim \
    --timezone Australia/Perth
```

## Key Concepts

### No Future Data Leakage
At each simulation time point, the system only uses data that would have been available at that moment in time. This ensures realistic testing of the prediction algorithms.

### Performance Metrics

- **Accuracy**: Overall correctness of predictions
- **Precision**: Of the alerts generated, how many were correct?
- **Recall**: Of the actual problems, how many were caught?
- **F1 Score**: Harmonic mean of precision and recall

### Confusion Matrix

|                    | No Alert (Predicted) | Alert (Predicted) |
|--------------------|---------------------|------------------|
| **Success (Actual)** | True Negative       | False Positive   |
| **Problem (Actual)** | False Negative      | True Positive    |

## Output Files

Each simulation creates:
- `{simulation_id}_results.json`: Complete results with metrics
- `{simulation_id}_timeline.csv`: Detailed timeline of all analysis points
- `{simulation_id}_*.png`: Various visualization charts

## Configuration Parameters

### Simulation Parameters
- `--interval-hours`: How often to run analysis (default: 1.0)
- `--vehicles`: Specific vehicle IDs to simulate (optional)

### Shift Parameters (inherited from main system)
- `--shift-hours`: Duration of a shift (default: 12)
- `--target-trips`: Target trips per shift (default: 4)
- `--buffer-minutes`: Buffer time before shift end (default: 30)
- `--default-trip-duration`: Default trip duration when no history (default: 45)

## Best Practices

1. **Start with shorter periods**: Test on a few days before running week-long simulations
2. **Use appropriate intervals**: Hourly is usually sufficient; use shorter intervals only when needed
3. **Compare configurations**: Run multiple simulations with different parameters to find optimal settings
4. **Review false positives**: High false positive rates indicate overly sensitive alert thresholds
5. **Check timing**: Look at when alerts are generated - too early may annoy users, too late may be useless

## Interpreting Results

### Good Performance Indicators
- High accuracy (>85%)
- Balanced precision and recall
- Alerts generated with sufficient time to act
- Consistent performance across vehicles

### Warning Signs
- Very high false positive rate (crying wolf)
- Very low recall (missing real problems)
- Alerts generated too late to be actionable
- Significant variation between vehicles

## Integration with Main System

The simulation module uses the same core analysis components as the production system:
- `TripExtractor`: Extracts trips from vehicle history
- `ShiftAnalyzer`: Analyzes shifts and makes predictions
- `AlertManager`: Determines when to generate alerts

This ensures that simulation results accurately reflect how the system would perform in production.