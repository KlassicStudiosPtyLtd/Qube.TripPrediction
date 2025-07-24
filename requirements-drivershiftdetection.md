# Driver Shift Detection Requirements

## Overview
Create a driver-change-based shift detection system that identifies shifts based on when the `driverId` changes in vehicle history data, rather than using fixed time boundaries.

## Core Requirements

### 1. Shift Detection Logic
- **Primary Trigger**: Detect shift changes when `driverId` field changes in vehicle history
- **Secondary Trigger**: Detect shift boundaries when there's a large time gap (configurable, default: 4+ hours) between consecutive events
- **Minimum Shift Duration**: Only consider shifts longer than a configurable minimum (default: 30 minutes)
- **Minimum Events**: Only consider shifts with a minimum number of events (default: 5 events)

### 2. Input Parameters
- **Fleet ID**: Required - specify which fleet to analyze
- **Date Range**: Required - start and end date/time for analysis period
- **Timezone**: Configurable timezone for date input and output display (default: Australia/Perth)
- **Vehicle Filter**: Optional - filter to specific vehicle by vehicle reference (e.g., "PRM00843")
- **Cache Directory**: Optional - use existing cache files if available

### 3. Data Sources
- **Primary**: Use existing cache files from simulation/analysis runs if available
- **Fallback**: Fetch fresh data from MTDATA API if cache not available
- **Cache Format**: Support existing cache file formats from the current system

### 4. Shift Detection Algorithm

#### 4.1 Core Logic
1. Sort vehicle history events by `deviceTimeUtc` timestamp
2. Track current driver ID (`driverId` field)
3. When driver ID changes OR time gap exceeds threshold:
   - Finalize current shift (if valid)
   - Start new shift with new driver
4. Handle null/empty driver IDs appropriately
5. Apply minimum duration and event count filters

#### 4.2 Shift Boundaries
- **Start**: First event with a driver ID
- **End**: Last event before driver change or large time gap
- **Validation**: Ensure shifts meet minimum requirements before including

### 5. Output Requirements

#### 5.1 Shift Summary Report (Console)
Display summary statistics:
- Total shifts detected
- Total vehicles analyzed  
- Date range analyzed
- Number of unique drivers
- Average shift duration
- Total shift hours across all vehicles

#### 5.2 Detailed Shifts CSV Report
Generate CSV file with columns:
- `shift_id` - Unique identifier for the shift
- `vehicle_id` - Vehicle ID from API
- `vehicle_ref` - Vehicle reference (e.g., "PRM00843")
- `driver_id` - Driver ID (may be null)
- `driver_name` - Driver name from history
- `driver_ref` - Driver reference number
- `shift_date` - Date of shift start (local timezone)
- `start_time` - Shift start time (local timezone with timezone indicator)
- `end_time` - Shift end time (local timezone with timezone indicator)
- `duration_hours` - Total shift duration in hours
- `total_events` - Number of history events in this shift
- `distance_km` - Distance traveled (calculated from odometer if available)
- `start_location_lat` - Starting latitude
- `start_location_lon` - Starting longitude  
- `end_location_lat` - Ending latitude
- `end_location_lon` - Ending longitude
- `start_place` - Starting place name
- `end_place` - Ending place name
- `max_speed_kmh` - Maximum speed during shift
- `total_fuel_used` - Fuel consumed during shift (if available)
- `engine_hours` - Engine hours during shift (if available)
- `shift_type` - Always "driver_change" to distinguish from fixed-time shifts

#### 5.3 JSON Summary File
Generate JSON file with:
- Overall statistics
- Per-vehicle summaries
- Configuration used
- Analysis metadata (timestamps, timezone, etc.)

### 6. Data Processing Requirements

#### 6.1 Vehicle History JSON Structure
Process the provided vehicle history format with fields:
- `driverId` - Primary field for shift detection
- `driver` - Driver name for display
- `driverRef` - Driver reference number
- `deviceTimeUtc` - Timestamp for event ordering
- `vehicleId` - Vehicle identifier
- `vehicleRef` - Vehicle reference (e.g., "PRM00843")
- `latitude`/`longitude` - Location data
- `placeName` - Location name
- `odometer` - For distance calculation
- `speedKmh` - For speed tracking
- `engineData.totalFuelUsed` - For fuel consumption
- `engineData.totalEngineHours` - For engine hour tracking

#### 6.2 Time Handling
- All input times in UTC (as stored in `deviceTimeUtc`)
- Convert to specified timezone for display and reporting
- Handle timezone-aware datetime objects throughout processing
- Default timezone: Australia/Perth

#### 6.3 Cache Integration
- Check for existing cache files first (from simulation cache directory)
- Use cache file naming pattern: `fleet_{fleet_id}_vehicle_{vehicle_id}_*.json`
- Support both direct history arrays and wrapped cache formats
- Fall back to API calls only if cache unavailable
- Optionally save new API data to cache for future use

### 7. Configuration Options

#### 7.1 Configurable Parameters
- `min_shift_duration_minutes` - Default: 30
- `max_event_gap_hours` - Default: 4  
- `min_events_per_shift` - Default: 5
- `timezone` - Default: "Australia/Perth"

#### 7.2 Command Line Interface
```bash
detect_driver_shifts.py \
    --fleet-id 104002 \
    --start-date "2025-07-09" \
    --end-date "2025-07-16" \
    --timezone "Australia/Perth" \
    --output-dir "./driver_shift_analysis" \
    --cache-dir "./simulation_cache" \
    --vehicle-ref "PRM00843"  # Optional filter
```

### 8. Error Handling

#### 8.1 Data Quality Issues
- Handle missing or null driver IDs gracefully
- Skip vehicles with insufficient data
- Log warnings for data quality issues
- Continue processing other vehicles if one fails

#### 8.2 Cache Issues
- Gracefully handle missing or corrupted cache files
- Fall back to API when cache unavailable
- Log cache hit/miss statistics

#### 8.3 API Issues
- Retry logic for API failures
- Clear error messages for authentication issues
- Graceful degradation when API unavailable

### 9. Integration with Existing System

#### 9.1 Reuse Existing Components
- Use existing API client (`MTDataApiClient`)
- Use existing cache mechanisms where possible
- Follow existing logging patterns
- Use existing timezone handling utilities

#### 9.2 Output Compatibility
- Generate outputs compatible with existing dashboard/visualization tools
- Use same timestamp formatting as existing reports
- Follow existing file naming conventions

### 10. Performance Requirements

#### 10.1 Processing Speed
- Process large datasets efficiently (1000+ vehicles, weeks of data)
- Use pandas for efficient data manipulation
- Minimize memory usage for large datasets

#### 10.2 Caching Strategy
- Prioritize cache usage to avoid redundant API calls
- Support partial cache scenarios (some vehicles cached, others not)
- Log cache usage statistics for debugging

### 11. Validation and Testing

#### 11.1 Data Validation
- Verify shift boundaries make logical sense
- Check for overlapping shifts (shouldn't happen with driver changes)
- Validate calculated metrics (duration, distance, etc.)

#### 11.2 Edge Cases
- Single-event shifts
- Rapid driver changes (< minimum duration)
- Missing driver information
- Large time gaps in data
- Vehicles with no driver changes (single long shift)

### 12. Documentation Requirements

#### 12.1 User Documentation
- Clear usage instructions
- Parameter explanations
- Example command lines
- Output format descriptions

#### 12.2 Technical Documentation
- Algorithm explanation
- Configuration options
- Integration points with existing system
- Troubleshooting guide