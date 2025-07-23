# Driver-Based Shift Detection with Overtime Awareness - Requirements

## Executive Summary

Modify the fleet shift analyzer system to detect shifts based on driver changes rather than fixed time periods. The system will track overtime to minimize costs and provide detailed performance analysis comparing drivers on similar routes.

## Context

The current system uses fixed 6am-6pm shifts which don't reflect actual driver working patterns. We need to:
- Detect when drivers change to accurately track individual shifts
- Monitor overtime (paid after `shift_duration_hours`) to minimize costs
- Provide actionable insights to prevent overtime before it occurs
- Compare driver performance on same routes with detailed explanations

## Key Requirements

### 1. Driver Change Detection
- **Primary Signal**: Use `driverId` field changes to detect new driver shifts
- **Shift Continuity**: If same driver returns within `max_shift_duration_hours`, continue their existing shift
- **Missing Data**: Assign "UNKNOWN" driver when `driverId` is missing

### 2. Shift Duration Management
- **Target Duration**: Each driver's shift targets completion within `shift_duration_hours` (default: 12 hours)
- **Maximum Duration**: Hard cutoff at `max_shift_duration_hours` (default: 15 hours)
- **Overtime Definition**: Time worked beyond `shift_duration_hours` incurs overtime costs

### 3. Trip Handling
- **Mid-Trip Driver Changes**: Split trips proportionally by distance when driver changes occur
- **Partial Trips**: Mark split trips with `is_partial=True` flag
- **Target Trips**: Fixed number of target trips regardless of shift start time

### 4. Performance Analysis
- **Route Comparison**: Compare drivers on same routes over baseline period
- **Detailed Explanations**: Provide specific reasons for missed targets:
  - Long dwell times (> `long_dwell_threshold_minutes`)
  - Slow speeds (< `slow_speed_threshold_percent` of baseline)
  - Gaps between trips
- **Data Quality**: Flag issues like rapid driver changes or simultaneous vehicle operation

### 5. Overtime Minimization
- **Proactive Alerts**: Warn when overtime is projected based on current performance
- **Recommendations**: Suggest specific actions to avoid overtime (e.g., "Reduce next trip by 2 stops")
- **Cost Impact**: Display overtime costs in all relevant views

## Technical Implementation

### Configuration Updates (`config.py`)

```python
# Driver-based shift detection
shift_detection_mode: str = 'driver_based'  # 'fixed_time' or 'driver_based'
max_shift_duration_hours: float = 15.0      # Maximum allowed shift duration
handle_missing_driver: str = 'assign_unknown'  # 'skip' or 'assign_unknown'

# Performance baseline
baseline_period_days: int = 30  # Compare against last 30 days

# Analysis thresholds
long_dwell_threshold_minutes: float = 30.0  # Flag dwell times longer than this
slow_speed_threshold_percent: float = 0.8   # Flag speeds < 80% of baseline
```

### Data Model Updates (`models.py`)

**Trip Class**:
- `is_partial: bool = False` - Indicates trip was split at driver change
- `handover_reason: Optional[str] = None` - Explains why trip was handed over

**Shift Class**:
- `driver_name: Optional[str] = None` - Human-readable driver name
- `driver_ref: Optional[str] = None` - Driver reference/ID
- `actual_end_time: Optional[datetime] = None` - For overtime calculation

### Module-Specific Changes

#### `analyzers/shift_analyzer.py`
- Replace `_group_trips_into_shifts()` with driver-based detection
- Add `_detect_driver_changes()` method
- Calculate overtime in `_analyze_single_shift()`
- Update risk calculations to prioritize overtime avoidance

#### `analyzers/trip_extractor.py`
- Detect driver changes during active trips
- Implement `_split_trips_at_driver_changes()`
- Handle missing driver IDs appropriately

#### `analyzers/fleet_analyzer.py`
- Pass raw history data to shift analyzer
- Update method signatures to include `history_data` parameter

#### `predictors/trip_predictor.py`
- Use `shift_duration_hours` (not max) for predictions
- Add overtime cost calculations
- Generate overtime alerts

#### `monitoring/alert_manager.py`
- Add overtime-specific alert types
- Include projected overtime hours and costs

#### `analyzers/trip_report_generator.py`
- Add driver information to all reports
- Create new `driver_shift_performance.csv` report
- Group reports by driver shifts

#### `visualization/dashboard_generator.py`
- Display driver-based shifts
- Show overtime warnings prominently
- Update tables with driver information

#### `simulation/simulation_engine.py`
- Update to use driver-based shift detection
- Ensure historical simulations reflect new logic

#### `main.py`
- Add command-line options:
  - `--shift-detection-mode`
  - `--max-shift-hours`
  - `--baseline-days`

## Business Rules

### Overtime Calculation
1. Regular time: 0 to `shift_duration_hours`
2. Overtime: `shift_duration_hours` to `max_shift_duration_hours`
3. Not allowed: Beyond `max_shift_duration_hours`

### Shift Boundaries
1. New shift starts when new driver detected
2. Shift continues if same driver returns within max duration
3. Shifts end for target calculations at configured duration, not when driver logs off

### Cost Implications
1. System must display overtime costs in all recommendations
2. Proactive alerts when overtime is projected
3. Clear ROI for following system recommendations

## Backward Compatibility

- Default to driver-based detection
- Maintain fixed-time mode as configuration option
- Existing reports continue to work with additional driver information

## Success Metrics

1. **Overtime Reduction**: Track % reduction in overtime hours
2. **Alert Effectiveness**: Measure how often alerts prevent overtime
3. **Driver Performance**: Compare drivers on same routes fairly
4. **Data Quality**: Monitor % of trips with unknown drivers

## Data Quality Requirements

### Input Validation
- Detect rapid driver changes (< 5 minutes apart)
- Flag simultaneous operation of same vehicle by multiple drivers
- Track percentage of trips with missing driver data

### Output Validation
- All shifts must have start and end times
- Overtime calculations must be consistent
- Performance comparisons must use same route baselines

## Testing Requirements

1. **Unit Tests**: Cover all new driver detection logic
2. **Integration Tests**: Verify trip splitting works correctly
3. **Performance Tests**: Ensure system handles large fleets efficiently
4. **Regression Tests**: Verify fixed-time mode still works

## Deployment Considerations

1. **Migration**: Handle existing data without driver information
2. **Training**: Update user documentation for new features
3. **Monitoring**: Add metrics for driver shift detection accuracy
4. **Rollback**: Ability to revert to fixed-time mode if needed