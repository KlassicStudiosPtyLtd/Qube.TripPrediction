# Driver Shift Detection Usage Guide

## Overview

The Driver Shift Detection system identifies driver shifts based on when the `driverId` changes in vehicle history data, rather than using fixed time boundaries. This provides more accurate shift tracking based on actual driver assignments.

## Prerequisites

- Python 3.7+
- Required packages: `pandas`, `pytz`
- Access to MTDATA API or cached vehicle history data
- Environment variables or config.yaml with API credentials

## Basic Usage

### Command Line Interface

```bash
python detect_driver_shifts.py \
    --fleet-id 104002 \
    --start-date "2025-07-09" \
    --end-date "2025-07-16" \
    --timezone "Australia/Perth" \
    --output-dir "./driver_shift_analysis" \
    --cache-dir "./simulation_cache"
```

### Required Parameters

- `--fleet-id`: The fleet ID to analyze (required)
- `--start-date`: Start date in YYYY-MM-DD format (required)
- `--end-date`: End date in YYYY-MM-DD format (required)

### Optional Parameters

- `--timezone`: Timezone for date input/output (default: "Australia/Perth")
- `--output-dir`: Directory for output files (default: "./driver_shift_analysis")
- `--cache-dir`: Directory with cached vehicle history files
- `--vehicle-ref`: Filter to specific vehicle (e.g., "PRM00843")
- `--min-shift-duration`: Minimum shift duration in minutes (default: 30)
- `--max-event-gap`: Maximum gap between events in hours (default: 4)
- `--min-events`: Minimum events per shift (default: 5)
- `--include-null-drivers`: Include periods with null driver IDs (idle/maintenance periods)
- `--log-level`: Logging level (default: "INFO")

## Examples

### Analyze All Vehicles in Fleet

```bash
python detect_driver_shifts.py \
    --fleet-id 104002 \
    --start-date "2025-07-09" \
    --end-date "2025-07-16" \
    --cache-dir "./simulation_cache"
```

### Analyze Single Vehicle

```bash
python detect_driver_shifts.py \
    --fleet-id 104002 \
    --start-date "2025-07-09" \
    --end-date "2025-07-16" \
    --vehicle-ref "PRM00843" \
    --cache-dir "./simulation_cache"
```

### Custom Parameters

```bash
python detect_driver_shifts.py \
    --fleet-id 104002 \
    --start-date "2025-07-09" \
    --end-date "2025-07-16" \
    --min-shift-duration 60 \
    --max-event-gap 6 \
    --min-events 10 \
    --timezone "Australia/Sydney" \
    --cache-dir "./simulation_cache"
```

### Using API (No Cache)

```bash
# Ensure MTDATA_API_USERNAME and MTDATA_API_PASSWORD are set
export MTDATA_API_USERNAME="your_username"
export MTDATA_API_PASSWORD="your_password"

python detect_driver_shifts.py \
    --fleet-id 104002 \
    --start-date "2025-07-09" \
    --end-date "2025-07-16"
```

### Include Idle/Maintenance Periods

```bash
# Include periods where no driver is logged in (useful for fleet utilization analysis)
python detect_driver_shifts.py \
    --fleet-id 104002 \
    --start-date "2025-07-09" \
    --end-date "2025-07-16" \
    --cache-dir "./simulation_cache" \
    --include-null-drivers
```

## Output Files

The system generates three types of output in the specified output directory:

### 1. Console Summary

Real-time summary displayed during execution showing:
- Total shifts detected
- Total vehicles analyzed
- Date range analyzed
- Number of unique drivers
- Average shift duration
- Total shift hours

### 2. CSV Report (`driver_shifts_detailed_[context]_[timestamp].csv`)

Detailed CSV file with timestamped filename and context. Example filename:
`driver_shifts_detailed_vehicle_PRM00843_fleet_104002_20250709_to_20250716_active_only_20250724_143022.csv`

Columns:
- `shift_id`: Unique identifier for the shift
- `vehicle_id`: Vehicle ID from API
- `vehicle_ref`: Vehicle reference (e.g., "PRM00843")
- `driver_id`: Driver ID (may be null)
- `driver_name`: Driver name from history
- `driver_ref`: Driver reference number
- `shift_date`: Date of shift start (local timezone)
- `start_time`: Shift start time (local timezone with timezone indicator)
- `end_time`: Shift end time (local timezone with timezone indicator)
- `duration_hours`: Total shift duration in hours
- `total_events`: Number of history events in this shift
- `distance_km`: Distance traveled (calculated from odometer)
- `start_location_lat/lon`: Starting coordinates
- `end_location_lat/lon`: Ending coordinates
- `start_place/end_place`: Place names
- `max_speed_kmh`: Maximum speed during shift
- `total_fuel_used`: Fuel consumed (if available)
- `engine_hours`: Engine hours (if available)
- `shift_type`: Always "driver_change"

### 3. JSON Summary (`driver_shifts_summary_[context]_[timestamp].json`)

JSON file with timestamped filename and context. Example filename:
`driver_shifts_summary_fleet_104002_20250709_to_20250716_active_only_20250724_143022.json`

Contents:
- Analysis metadata (timestamps, configuration)
- Summary statistics
- Per-vehicle shift details

### Output Filename Format

All output files use a standardized naming convention:
`[base_name]_[context]_[timestamp].[extension]`

**Context components (in order):**
- `vehicle_[ref]` - When analyzing single vehicle (e.g., `vehicle_PRM00843`)
- `fleet_[id]` - Fleet identifier (e.g., `fleet_104002`)
- `[start_date]_to_[end_date]` - Analysis date range (e.g., `20250709_to_20250716`)
- `active_only` or `with_idle` - Whether null driver periods are included

**Timestamp format:** `YYYYMMDD_HHMMSS` (UTC)

**Examples:**
- Single vehicle, active shifts only: `driver_shifts_detailed_vehicle_PRM00843_fleet_104002_20250709_to_20250716_active_only_20250724_143022.csv`
- Full fleet with idle periods: `driver_shifts_summary_fleet_104002_20250709_to_20250716_with_idle_20250724_143530.json`

## Algorithm Details

### Shift Detection Logic

1. **Primary Trigger**: Detects shift changes when `driverId` field changes
2. **Secondary Trigger**: Detects shift boundaries on large time gaps (default: 4+ hours)
3. **Validation**: Only considers shifts meeting minimum requirements:
   - Duration > 30 minutes (configurable)
   - Event count > 5 events (configurable)

### Data Processing

1. Vehicle history events sorted by `deviceTimeUtc`
2. Current driver ID tracked from `driverId` field
3. When driver changes OR time gap exceeds threshold:
   - Current shift finalized (if valid)
   - New shift started with new driver
4. Null/empty driver IDs handled gracefully
5. Minimum duration and event count filters applied

## Cache File Support

The system supports existing cache file formats:

### Supported Cache Patterns

- `fleet_{fleet_id}_vehicle_{vehicle_id}_*.json`
- Nested directory structures (e.g., `PLS/PRM00843/`)
- Both direct history arrays and wrapped cache formats

### Cache Priority

1. Check cache directory first
2. Fall back to API calls if cache unavailable
3. Support partial cache scenarios

## Error Handling

### Data Quality Issues

- Missing or null driver IDs handled gracefully
- Vehicles with insufficient data skipped
- Data quality issues logged as warnings
- Processing continues for other vehicles

### Cache Issues

- Missing/corrupted cache files handled
- Automatic fallback to API
- Cache hit/miss statistics logged

### API Issues

- Retry logic for API failures
- Clear error messages for auth issues
- Graceful degradation when API unavailable

## Troubleshooting

### Common Issues

1. **No shifts detected**
   - Check if vehicles have driver ID data
   - Verify date range contains vehicle activity
   - Check minimum shift duration settings

2. **Cache files not found**
   - Verify cache directory path
   - Check cache file naming pattern
   - Ensure API credentials are configured

3. **API authentication errors**
   - Verify MTDATA_API_USERNAME and MTDATA_API_PASSWORD
   - Check network connectivity
   - Verify API access permissions

### Debug Mode

Run with debug logging for detailed information:

```bash
python detect_driver_shifts.py \
    --fleet-id 104002 \
    --start-date "2025-07-09" \
    --end-date "2025-07-16" \
    --log-level DEBUG
```

### Configuration Validation

The system validates:
- Date ranges are logical
- Fleet ID exists and accessible
- Vehicle references exist (when specified)
- Output directories are writable
- Cache directories exist (when specified)

## Integration Notes

### Existing System Compatibility

- Uses existing `MTDataApiClient`
- Follows existing cache mechanisms
- Compatible with existing logging patterns
- Uses same timestamp formatting as other reports
- Follows existing file naming conventions

### Output Compatibility

- CSV format compatible with existing dashboard tools
- JSON structure matches existing report formats
- Timezone handling consistent with other analyzers

## Performance Considerations

- Processes large datasets efficiently (1000+ vehicles)
- Uses pandas for data manipulation
- Minimizes memory usage for large datasets
- Prioritizes cache usage to avoid redundant API calls
- Supports partial cache scenarios
- Logs cache usage statistics for debugging