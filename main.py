#!/usr/bin/env python3
"""
Fleet Shift Analyzer - Main entry point with three-point round trip support
"""
import click
import logging
from typing import Dict, Any
from datetime import datetime
import pytz

from core.mtdata_api_client import MTDataApiClient
from config import ShiftConfig, AnalysisConfig
from analyzers.fleet_analyzer import FleetAnalyzer
from core.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_datetime_string(datetime_str: str, timezone_str: str, is_end_date: bool = False) -> datetime:
    """
    Parse a datetime string that could be either a date or datetime.
    
    Args:
        datetime_str: Date or datetime string in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
        timezone_str: Timezone string (e.g., 'Australia/Perth', 'US/Eastern')
        is_end_date: If True and only date is provided, set to end of day
        
    Returns:
        UTC datetime object
    """
    # Get the timezone
    local_tz = pytz.timezone(timezone_str)
    
    # Try to parse as datetime first, then fall back to date
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d"
    ]
    
    parsed_datetime = None
    for fmt in formats:
        try:
            parsed_datetime = datetime.strptime(datetime_str, fmt)
            break
        except ValueError:
            continue
    
    if parsed_datetime is None:
        raise ValueError(f"Could not parse datetime string: {datetime_str}. "
                        f"Expected formats: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")
    
    # If only date was provided, set appropriate time
    if len(datetime_str) <= 10:  # Only date provided
        if is_end_date:
            # Set to end of day for end dates
            parsed_datetime = parsed_datetime.replace(hour=23, minute=59, second=59)
        # else: start of day (00:00:00) is already set by default
    
    # Localize to the specified timezone
    local_datetime = local_tz.localize(parsed_datetime)
    
    # Convert to UTC
    utc_datetime = local_datetime.astimezone(pytz.UTC)
    
    return utc_datetime


def convert_utc_to_local(utc_datetime: datetime, timezone_str: str) -> datetime:
    """
    Convert UTC datetime to local timezone.
    
    Args:
        utc_datetime: UTC datetime object
        timezone_str: Timezone string
        
    Returns:
        Local datetime object
    """
    if utc_datetime.tzinfo is None:
        # If naive datetime, assume it's UTC
        utc_datetime = pytz.UTC.localize(utc_datetime)
    
    local_tz = pytz.timezone(timezone_str)
    return utc_datetime.astimezone(local_tz)


@click.command()
@click.option('--fleet-id', type=int, required=True, help='Fleet ID to analyze')
@click.option('--start-date', required=True, 
              help='Start date/time in local timezone. Formats: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS')
@click.option('--end-date', required=True, 
              help='End date/time in local timezone. Formats: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS')
@click.option('--timezone', default='Australia/Perth', 
              help='Timezone for dates (e.g., Australia/Perth, US/Eastern)')
@click.option('--output-dir', default='./fleet_shift_analysis', help='Output directory')
@click.option('--shift-hours', default=12, type=float, help='Shift duration in hours')
@click.option('--target-trips', default=4, type=int, help='Target trips per shift')
@click.option('--buffer-minutes', default=30, type=int, help='Buffer time before shift end')
@click.option('--default-trip-duration', default=45, type=float, 
              help='Default trip duration in minutes when no historical data available')
@click.option('--start-waypoint', default=None, help='Start waypoint for trips')
@click.option('--target-waypoint', default=None, 
              help='Target/intermediate waypoint for three-point round trips')
@click.option('--end-waypoint', default=None, help='End waypoint for trips')
@click.option('--waypoint-matching', 
              type=click.Choice(['exact', 'contains', 'normalized']), 
              default='exact', 
              help='How to match waypoint names')
@click.option('--round-trip-mode', 
              type=click.Choice(['simple', 'three_point']), 
              default='simple',
              help='Round trip mode: simple (A->B->A) or three_point (A->B->C)')
@click.option('--require-waypoint-order/--no-require-waypoint-order', 
              default=True,
              help='Whether waypoints must be visited in strict order')
@click.option('--allow-partial-trips/--no-allow-partial-trips', 
              default=False,
              help='Whether to count incomplete round trips')
@click.option('--segment-duration-start-target', default=None, type=float,
              help='Default duration for Start->Target segment (three-point mode)')
@click.option('--segment-duration-target-end', default=None, type=float,
              help='Default duration for Target->End segment (three-point mode)')
@click.option('--vehicle-ref', default=None, help='Filter analysis to a specific vehicle by reference (e.g., T123)')
@click.option('--cache-dir', default=None, help='Directory for caching API responses (speeds up repeated analyses)')
@click.option('--parallel', is_flag=True, default=True, help='Enable parallel processing')
@click.option('--max-workers', default=4, type=int, help='Maximum parallel workers')
@click.option('--log-level', default='INFO', help='Logging level')
def main(fleet_id: int, start_date: str, end_date: str, timezone: str, output_dir: str,
         shift_hours: float, target_trips: int, buffer_minutes: int,
         default_trip_duration: float,
         start_waypoint: str, target_waypoint: str, end_waypoint: str, 
         waypoint_matching: str, round_trip_mode: str,
         require_waypoint_order: bool, allow_partial_trips: bool,
         segment_duration_start_target: float, segment_duration_target_end: float,
         vehicle_ref: str, cache_dir: str,
         parallel: bool, max_workers: int, log_level: str):
    """
    Analyze fleet-wide shift completion with three-point round trip support.
    
    The system supports two modes:
    
    1. Simple Mode: Traditional two-point trips (A → B or A → B → A)
    2. Three-Point Mode: Complex round trips (A → B → C)
    
    Three-Point Round Trip Example:
        Start: Depot
        Target: Customer Site (delivery)
        End: Depot (can be same as Start)
    
    Examples:
        # Simple round trip
        --start-waypoint "Port Berth" --end-waypoint "Port Berth"
        
        # Three-point round trip
        --round-trip-mode three_point \\
        --start-waypoint "Depot_A" \\
        --target-waypoint "Customer_Site" \\
        --end-waypoint "Depot_A"
        
        # Three-point with different start/end
        --round-trip-mode three_point \\
        --start-waypoint "Depot_West" \\
        --target-waypoint "Loading_Dock" \\
        --end-waypoint "Depot_East"
    """
    # Set up logging
    setup_logging(log_level)
    
    logger.info("Starting Fleet Shift Analysis")
    logger.info(f"Fleet ID: {fleet_id}")
    logger.info(f"Date Range: {start_date} to {end_date} ({timezone})")
    logger.info(f"Round Trip Mode: {round_trip_mode}")
    
    if vehicle_ref:
        logger.info(f"Filtering by Vehicle Ref: {vehicle_ref}")
    
    if cache_dir:
        logger.info(f"Using cache directory: {cache_dir}")
    
    # Validate timezone
    try:
        pytz.timezone(timezone)
    except pytz.exceptions.UnknownTimeZoneError:
        logger.error(f"Unknown timezone: {timezone}")
        logger.info("Common timezones: Australia/Perth, Australia/Sydney, US/Eastern, US/Pacific, Europe/London")
        return
    
    # Validate waypoint configuration
    if round_trip_mode == 'three_point':
        if not all([start_waypoint, target_waypoint, end_waypoint]):
            logger.error("Three-point mode requires all waypoints: --start-waypoint, --target-waypoint, --end-waypoint")
            return
        logger.info(f"Three-Point Route: {start_waypoint} → {target_waypoint} → {end_waypoint}")
        logger.info(f"Waypoint Matching: {waypoint_matching}")
        logger.info(f"Require Order: {require_waypoint_order}")
        logger.info(f"Allow Partial Trips: {allow_partial_trips}")
    elif start_waypoint and end_waypoint:
        logger.info(f"Waypoint Route: {start_waypoint} → {end_waypoint}")
        logger.info(f"Waypoint Matching: {waypoint_matching}")
    
    try:
        # Parse dates/times with enhanced parser
        start_datetime_utc = parse_datetime_string(start_date, timezone, is_end_date=False)
        end_datetime_utc = parse_datetime_string(end_date, timezone, is_end_date=True)
        
        # Validate date range
        if end_datetime_utc <= start_datetime_utc:
            logger.error("End date/time must be after start date/time")
            return
        
        # Log the parsed times
        tz = pytz.timezone(timezone)
        start_local = start_datetime_utc.astimezone(tz)
        end_local = end_datetime_utc.astimezone(tz)
        
        logger.info(f"Parsed time range ({timezone}):")
        logger.info(f"  Start: {start_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.info(f"  End: {end_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.info(f"UTC time range:")
        logger.info(f"  Start: {start_datetime_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info(f"  End: {end_datetime_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Prepare segment durations
        default_segment_durations = {
            'start_to_target': segment_duration_start_target or default_trip_duration,
            'target_to_end': segment_duration_target_end or default_trip_duration
        }
        
        # Initialize configurations with timezone and three-point support
        shift_config = ShiftConfig(
            shift_duration_hours=shift_hours,
            target_trips=target_trips,
            buffer_time_minutes=buffer_minutes,
            default_trip_duration_minutes=default_trip_duration,
            start_waypoint=start_waypoint,
            target_waypoint=target_waypoint,
            end_waypoint=end_waypoint,
            waypoint_matching=waypoint_matching,
            round_trip_mode=round_trip_mode,
            require_waypoint_order=require_waypoint_order,
            allow_partial_trips=allow_partial_trips,
            default_segment_durations=default_segment_durations,
            timezone=timezone
        )
        
        analysis_config = AnalysisConfig(
            parallel_processing=parallel,
            max_workers=max_workers
        )
        
        # Initialize API client
        api_logger = logging.getLogger('mtdata_api_client')
        api_client = MTDataApiClient(logger=api_logger)
        
        # Create analyzer with cache support
        analyzer = FleetAnalyzer(api_client, shift_config, analysis_config, cache_dir=cache_dir)
        
        # Run analysis with UTC times
        fleet_summary = analyzer.analyze_fleet(
            fleet_id=fleet_id,
            start_datetime=start_datetime_utc,
            end_datetime=end_datetime_utc,
            output_dir=output_dir,
            timezone=timezone,
            vehicle_ref_filter=vehicle_ref
        )
        
        # Print summary
        if fleet_summary:
            print_analysis_summary(fleet_summary, shift_config, timezone)
        
    except ValueError as e:
        logger.error(f"Date parsing error: {str(e)}")
        logger.info("Hint: Use format 'YYYY-MM-DD' for dates or 'YYYY-MM-DD HH:MM:SS' for date+time")
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)


def print_analysis_summary(fleet_summary: Dict[str, Any], 
                         shift_config: ShiftConfig,
                         timezone: str = 'UTC'):
    """Print analysis summary to console with three-point trip details."""
    summary = fleet_summary.get('fleet_summary', {})
    
    print(f"\n{'='*60}")
    print("FLEET SHIFT ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Timezone: {timezone}")
    print(f"Round Trip Mode: {shift_config.round_trip_mode}")
    
    if shift_config.round_trip_mode == 'three_point':
        print(f"Route: {shift_config.start_waypoint} → {shift_config.target_waypoint} → {shift_config.end_waypoint}")
    elif shift_config.start_waypoint and shift_config.end_waypoint:
        print(f"Route: {shift_config.start_waypoint} → {shift_config.end_waypoint}")
    
    print(f"{'='*60}")
    print(f"Vehicles Analyzed: {summary.get('total_vehicles_analyzed', 0)}")
    print(f"Total Shifts: {summary.get('total_shifts_analyzed', 0)}")
    print(f"Shifts Meeting Target: {summary.get('shifts_meeting_target', 0)}")
    print(f"Shifts at Risk: {summary.get('shifts_at_risk', 0)}")
    print(f"Target Completion Rate: {summary.get('target_completion_rate', 0):.1%}")
    print(f"Active Alerts: {summary.get('total_alerts', 0)}")
    
    # Additional three-point statistics
    if shift_config.round_trip_mode == 'three_point':
        print(f"\nTHREE-POINT ROUND TRIP STATISTICS:")
        total_complete_round_trips = 0
        total_incomplete_round_trips = 0
        
        for vehicle_id, analysis in fleet_summary.get('vehicle_analyses', {}).items():
            trips = analysis.get('trips', [])
            complete = sum(1 for t in trips if t.get('trip_type') == 'three_point_round' and t.get('is_complete_round_trip'))
            incomplete = sum(1 for t in trips if t.get('trip_type') == 'three_point_round' and not t.get('is_complete_round_trip'))
            total_complete_round_trips += complete
            total_incomplete_round_trips += incomplete
        
        print(f"  • Complete Round Trips: {total_complete_round_trips}")
        print(f"  • Incomplete Round Trips: {total_incomplete_round_trips}")
        if total_complete_round_trips + total_incomplete_round_trips > 0:
            completion_rate = total_complete_round_trips / (total_complete_round_trips + total_incomplete_round_trips)
            print(f"  • Round Trip Completion Rate: {completion_rate:.1%}")
    
    # Show detailed shifts and trips...
    # (Rest of the print_analysis_summary function remains the same as the original)
    
    # Show alerts
    alerts = fleet_summary.get('alerts', [])
    if alerts:
        print(f"\n\033[91mACTIVE ALERTS ({len(alerts)}):\033[0m")
        for i, alert in enumerate(alerts[:10]):  # Show first 10 alerts
            print(f"  {i+1}. [{alert['severity'].upper()}] Vehicle {alert['vehicle_name']} "
                  f"(Driver: {alert['driver_id']}, Date: {alert['shift_date']})")
            print(f"     → {alert['message']}")
        
        if len(alerts) > 10:
            print(f"  ... and {len(alerts) - 10} more alerts")
    else:
        print("\n\033[92mNo active alerts - all shifts within acceptable parameters.\033[0m")


if __name__ == "__main__":
    main()