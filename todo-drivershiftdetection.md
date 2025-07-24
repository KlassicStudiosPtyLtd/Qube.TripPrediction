# Claude AI Assistant Context: Driver Shift Detection Todo List

## Project Overview
Implementation of a driver-change-based shift detection system that identifies shifts based on when the driverId changes in vehicle history data.

## Todo List

### High Priority Tasks
- [ ] **Task 1**: Set up project structure and create main script detect_driver_shifts.py with command line interface
- [ ] **Task 2**: Implement configuration handling for all parameters (timezone, min duration, gaps, etc)
- [ ] **Task 3**: Implement cache file detection and loading logic for existing simulation cache
- [ ] **Task 4**: Implement core shift detection algorithm based on driver ID changes
- [ ] **Task 5**: Add secondary trigger for large time gaps between events
- [ ] **Task 10**: Implement CSV report generation with all required columns

### Medium Priority Tasks
- [ ] **Task 6**: Implement shift validation (minimum duration and event count filters)
- [ ] **Task 7**: Calculate shift metrics (duration, distance, speed, fuel, engine hours)
- [ ] **Task 8**: Implement timezone conversion for all timestamps (UTC to local)
- [ ] **Task 9**: Create console summary report with statistics
- [ ] **Task 11**: Implement JSON summary file generation
- [ ] **Task 12**: Add error handling for data quality issues and missing driver IDs
- [ ] **Task 14**: Add vehicle filter functionality

### Low Priority Tasks
- [ ] **Task 13**: Implement API fallback when cache is unavailable
- [ ] **Task 15**: Test with edge cases (single events, rapid changes, missing data)
- [ ] **Task 16**: Add logging throughout the system
- [ ] **Task 17**: Create usage documentation and examples

## Implementation Notes

### Key Requirements to Remember
- Primary trigger: Driver ID changes
- Secondary trigger: Time gaps > 4 hours (configurable)
- Minimum shift duration: 30 minutes (configurable)
- Minimum events per shift: 5 (configurable)
- Default timezone: Australia/Perth
- Support existing cache file formats
- Generate CSV, JSON, and console reports

### Data Fields to Track
- driverId (primary field for detection)
- deviceTimeUtc (for event ordering)
- vehicleId, vehicleRef
- latitude/longitude, placeName
- odometer (for distance)
- speedKmh
- engineData.totalFuelUsed
- engineData.totalEngineHours

### Output CSV Columns Required
- shift_id, vehicle_id, vehicle_ref
- driver_id, driver_name, driver_ref
- shift_date, start_time, end_time
- duration_hours, total_events, distance_km
- start/end location coordinates and place names
- max_speed_kmh, total_fuel_used, engine_hours
- shift_type (always "driver_change")

### Integration Points
- Reuse existing MTDataApiClient
- Follow existing cache mechanisms
- Use existing timezone utilities
- Match existing file naming conventions