# Claude AI Assistant Context

## Project Overview
Qube Trip Prediction - Fleet management system with shift analysis and trip prediction capabilities.

## Current Task: Driver-Based Shift Detection Implementation

### Implementation Checklist

#### Phase 1: Configuration & Data Models
- [ ] Update `config.py` with new driver-based shift parameters
  - [ ] Add `shift_detection_mode` parameter
  - [ ] Add `max_shift_duration_hours` parameter
  - [ ] Add `handle_missing_driver` parameter
  - [ ] Add performance baseline parameters
  - [ ] Add analysis threshold parameters
- [ ] Update `models.py` data structures
  - [ ] Add `is_partial` and `handover_reason` to Trip class
  - [ ] Add driver fields to Shift class
  - [ ] Add `actual_end_time` for overtime calculation

#### Phase 2: Core Shift Detection Logic
- [ ] Modify `analyzers/shift_analyzer.py`
  - [ ] Replace `_group_trips_into_shifts()` with driver-based logic
  - [ ] Implement `_detect_driver_changes()` method
  - [ ] Update `_analyze_single_shift()` for overtime calculation
  - [ ] Modify risk calculations to prioritize overtime avoidance
- [ ] Update `analyzers/trip_extractor.py`
  - [ ] Add driver change detection during active trips
  - [ ] Implement `_split_trips_at_driver_changes()` method
  - [ ] Handle missing driver IDs (assign "UNKNOWN")

#### Phase 3: Fleet Analysis Updates
- [ ] Modify `analyzers/fleet_analyzer.py`
  - [ ] Pass raw history data to shift analyzer
  - [ ] Update method signatures with `history_data` parameter
- [ ] Update `predictors/trip_predictor.py`
  - [ ] Use `shift_duration_hours` for predictions
  - [ ] Add overtime cost calculations
  - [ ] Generate overtime prevention alerts

#### Phase 4: Monitoring & Alerts
- [ ] Enhance `monitoring/alert_manager.py`
  - [ ] Create overtime-specific alert types
  - [ ] Include projected overtime hours and costs
  - [ ] Add proactive overtime prevention alerts

#### Phase 5: Reporting & Visualization
- [ ] Update `analyzers/trip_report_generator.py`
  - [ ] Add driver information to existing reports
  - [ ] Create new `driver_shift_performance.csv` report
  - [ ] Group reports by driver shifts
- [ ] Modify `visualization/dashboard_generator.py`
  - [ ] Display driver-based shifts
  - [ ] Show overtime warnings prominently
  - [ ] Update tables with driver information

#### Phase 6: Simulation & Testing
- [ ] Update `simulation/simulation_engine.py`
  - [ ] Implement driver-based shift detection in simulations
  - [ ] Ensure historical simulations use new logic
- [ ] Modify `main.py`
  - [ ] Add `--shift-detection-mode` option
  - [ ] Add `--max-shift-hours` option
  - [ ] Add `--baseline-days` option

#### Phase 7: Testing & Validation
- [ ] Write unit tests for driver detection logic
- [ ] Write integration tests for trip splitting
- [ ] Test backward compatibility with fixed-time mode
- [ ] Validate overtime calculations
- [ ] Test with missing driver data scenarios

### Key Business Logic to Remember

1. **Overtime Definition**: Starts after `shift_duration_hours` (default 12h), hard stop at `max_shift_duration_hours` (15h)
2. **Driver Continuity**: Same driver returning within max duration continues existing shift
3. **Trip Splitting**: When driver changes mid-trip, split proportionally by distance
4. **Missing Data**: Assign "UNKNOWN" driver when driverId is missing
5. **Performance Baseline**: Compare against last 30 days of same route data

### Important Commands

```bash
# Run tests
python -m pytest tests/

# Run with driver-based detection
python main.py analyze --shift-detection-mode driver_based

# Generate reports with driver shifts
python main.py report --group-by-driver

# Run simulation with overtime analysis
python main.py simulate --include-overtime-costs
```

### Common Issues & Solutions

1. **Missing Driver Data**: System assigns "UNKNOWN" - check data quality reports
2. **Rapid Driver Changes**: Flag potential data issues when changes < 5 min apart
3. **Overtime Projections**: Based on current performance vs remaining shift time

### Performance Considerations

- Driver detection should handle 1000+ trips per vehicle efficiently
- Trip splitting should not significantly impact processing time
- Baseline calculations cached for performance

### Next Steps After Implementation

1. Monitor overtime reduction metrics
2. Gather feedback on alert effectiveness
3. Fine-tune thresholds based on real-world data
4. Consider ML-based overtime prediction in future