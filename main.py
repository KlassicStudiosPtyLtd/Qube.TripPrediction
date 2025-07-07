#!/usr/bin/env python3
"""
Fleet Shift Analyzer - Main entry point with timezone support
Enhanced to support both date-only and datetime inputs
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
@click.option('--start-waypoint', default=None, help='Start waypoint for trips')
@click.option('--end-waypoint', default=None, help='End waypoint for trips')
@click.option('--waypoint-matching', 
              type=click.Choice(['exact', 'contains', 'normalized']), 
              default='exact', 
              help='How to match waypoint names')
@click.option('--parallel', is_flag=True, default=True, help='Enable parallel processing')
@click.option('--max-workers', default=4, type=int, help='Maximum parallel workers')
@click.option('--log-level', default='INFO', help='Logging level')
def main(fleet_id: int, start_date: str, end_date: str, timezone: str, output_dir: str,
         shift_hours: float, target_trips: int, buffer_minutes: int,
         start_waypoint: str, end_waypoint: str, waypoint_matching: str,
         parallel: bool, max_workers: int, log_level: str):
    """
    Analyze fleet-wide shift completion and generate trip predictions.
    
    Date/time inputs can be provided in the following formats:
    - Date only: YYYY-MM-DD (assumes 00:00:00 for start, 23:59:59 for end)
    - Date and time: YYYY-MM-DD HH:MM:SS or YYYY-MM-DD HH:MM
    
    Times are specified in the local timezone and converted to UTC for API calls.
    
    Examples:
        # Analyze full day
        --start-date 2025-06-17 --end-date 2025-06-17
        
        # Analyze specific time period
        --start-date "2025-06-17 06:00:00" --end-date "2025-06-17 18:00:00"
    """
    # Set up logging
    setup_logging(log_level)
    
    logger.info("Starting Fleet Shift Analysis")
    logger.info(f"Fleet ID: {fleet_id}")
    logger.info(f"Date Range: {start_date} to {end_date} ({timezone})")
    
    # Validate timezone
    try:
        pytz.timezone(timezone)
    except pytz.exceptions.UnknownTimeZoneError:
        logger.error(f"Unknown timezone: {timezone}")
        logger.info("Common timezones: Australia/Perth, Australia/Sydney, US/Eastern, US/Pacific, Europe/London")
        return
    
    if start_waypoint and end_waypoint:
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
        
        # Initialize configurations with timezone
        shift_config = ShiftConfig(
            shift_duration_hours=shift_hours,
            target_trips=target_trips,
            buffer_time_minutes=buffer_minutes,
            start_waypoint=start_waypoint,
            end_waypoint=end_waypoint,
            waypoint_matching=waypoint_matching,
            timezone=timezone
        )
        
        analysis_config = AnalysisConfig(
            parallel_processing=parallel,
            max_workers=max_workers
        )
        
        # Initialize API client
        api_logger = logging.getLogger('mtdata_api_client')
        api_client = MTDataApiClient(logger=api_logger)
        
        # Create analyzer
        analyzer = FleetAnalyzer(api_client, shift_config, analysis_config)
        
        # Run analysis with UTC times
        fleet_summary = analyzer.analyze_fleet(
            fleet_id=fleet_id,
            start_datetime=start_datetime_utc,
            end_datetime=end_datetime_utc,
            output_dir=output_dir,
            timezone=timezone
        )
        
        # Print summary
        if fleet_summary:
            print_analysis_summary(fleet_summary, start_waypoint, end_waypoint, timezone)
        
    except ValueError as e:
        logger.error(f"Date parsing error: {str(e)}")
        logger.info("Hint: Use format 'YYYY-MM-DD' for dates or 'YYYY-MM-DD HH:MM:SS' for date+time")
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)


def print_analysis_summary(fleet_summary: Dict[str, Any], 
                         start_waypoint: str = None, 
                         end_waypoint: str = None,
                         timezone: str = 'UTC'):
    """Print analysis summary to console with timezone-aware times."""
    summary = fleet_summary.get('fleet_summary', {})
    
    print(f"\n{'='*60}")
    print("FLEET SHIFT ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Timezone: {timezone}")
    
    if start_waypoint and end_waypoint:
        print(f"Route: {start_waypoint} → {end_waypoint}")
        print(f"{'='*60}")
    
    print(f"Vehicles Analyzed: {summary.get('total_vehicles_analyzed', 0)}")
    print(f"Total Shifts: {summary.get('total_shifts_analyzed', 0)}")
    print(f"Shifts Meeting Target: {summary.get('shifts_meeting_target', 0)}")
    print(f"Shifts at Risk: {summary.get('shifts_at_risk', 0)}")
    print(f"Target Completion Rate: {summary.get('target_completion_rate', 0):.1%}")
    print(f"Active Alerts: {summary.get('total_alerts', 0)}")
    
    # Print detailed shift analysis table
    print(f"\n{'='*150}")
    print("SHIFT ANALYSIS DETAILS")
    print(f"{'='*150}")
    
    # Define column widths
    col_widths = {
        'Shift Date': 12,
        'Vehicle Name': 15,
        'Driver ID': 12,
        'Shift Start': 20,
        'Shift End': 20,
        'Trips': 7,
        'Target': 7,
        'Met': 5,
        'Risk': 8,
        'Alert': 7,
        'Recommendation': 40
    }
    
    # Print header
    header = (
        f"{'Shift Date':<{col_widths['Shift Date']}} | "
        f"{'Vehicle Name':<{col_widths['Vehicle Name']}} | "
        f"{'Driver ID':<{col_widths['Driver ID']}} | "
        f"{'Shift Start':<{col_widths['Shift Start']}} | "
        f"{'Shift End':<{col_widths['Shift End']}} | "
        f"{'Trips':<{col_widths['Trips']}} | "
        f"{'Target':<{col_widths['Target']}} | "
        f"{'Met':<{col_widths['Met']}} | "
        f"{'Risk':<{col_widths['Risk']}} | "
        f"{'Alert':<{col_widths['Alert']}} | "
        f"{'Recommendation':<{col_widths['Recommendation']}}"
    )
    print(header)
    print("-" * 150)
    
    # Collect all shifts from all vehicles
    all_shifts = []
    vehicle_analyses = fleet_summary.get('vehicle_analyses', {})
    
    for vehicle_id, analysis in vehicle_analyses.items():
        vehicle_name = analysis.get('vehicle_name', f'Vehicle_{vehicle_id}')
        
        for shift in analysis.get('shift_analyses', []):
            shift_info = shift.get('shift', {})
            metrics = shift.get('metrics', {})
            
            # Convert UTC times to local timezone for display
            shift_start_utc = datetime.fromisoformat(shift_info.get('start_time', '').replace('Z', '+00:00'))
            shift_end_utc = datetime.fromisoformat(shift_info.get('end_time', '').replace('Z', '+00:00'))
            
            shift_start_local = convert_utc_to_local(shift_start_utc, timezone)
            shift_end_local = convert_utc_to_local(shift_end_utc, timezone)
            
            # Extract data
            shift_date = shift_start_local.strftime('%Y-%m-%d')
            driver_id = str(shift_info.get('driver_id', 'Unknown'))
            shift_start = shift_start_local.strftime('%Y-%m-%d %H:%M:%S')
            shift_end = shift_end_local.strftime('%Y-%m-%d %H:%M:%S')
            trips_completed = metrics.get('trips_completed', 0)
            target_trips = metrics.get('trips_target', 0)
            target_met = 'Yes' if shift.get('can_complete_target', False) else 'No'
            risk_level = shift.get('risk_level', 'Unknown')
            alert_required = 'Yes' if shift.get('alert_required', False) else 'No'
            recommendation = shift.get('recommendation', '')
            
            # Truncate long values to fit column widths
            vehicle_name_display = vehicle_name[:col_widths['Vehicle Name']-1] + '.' if len(vehicle_name) > col_widths['Vehicle Name'] else vehicle_name
            driver_id_display = driver_id[:col_widths['Driver ID']-1] + '.' if len(driver_id) > col_widths['Driver ID'] else driver_id
            recommendation_display = recommendation[:col_widths['Recommendation']-1] + '.' if len(recommendation) > col_widths['Recommendation'] else recommendation
            
            # Color coding for risk level
            if risk_level == 'high' or risk_level == 'critical':
                risk_display = f"\033[91m{risk_level:<{col_widths['Risk']}}\033[0m"  # Red
                alert_display = f"\033[91m{alert_required:<{col_widths['Alert']}}\033[0m"  # Red
            elif risk_level == 'medium':
                risk_display = f"\033[93m{risk_level:<{col_widths['Risk']}}\033[0m"  # Yellow
                alert_display = f"\033[93m{alert_required:<{col_widths['Alert']}}\033[0m"  # Yellow
            else:
                risk_display = f"\033[92m{risk_level:<{col_widths['Risk']}}\033[0m"  # Green
                alert_display = f"{alert_required:<{col_widths['Alert']}}"
            
            # Print row
            row = (
                f"{shift_date:<{col_widths['Shift Date']}} | "
                f"{vehicle_name_display:<{col_widths['Vehicle Name']}} | "
                f"{driver_id_display:<{col_widths['Driver ID']}} | "
                f"{shift_start:<{col_widths['Shift Start']}} | "
                f"{shift_end:<{col_widths['Shift End']}} | "
                f"{trips_completed:<{col_widths['Trips']}} | "
                f"{target_trips:<{col_widths['Target']}} | "
                f"{target_met:<{col_widths['Met']}} | "
                f"{risk_display} | "
                f"{alert_display} | "
                f"{recommendation_display:<{col_widths['Recommendation']}}"
            )
            
            # Store for sorting
            all_shifts.append({
                'date': shift_date,
                'row': row,
                'risk_level': risk_level
            })
    
    # Sort by date and risk level
    all_shifts.sort(key=lambda x: (x['date'], x['risk_level'] == 'high', x['risk_level'] == 'medium'))
    
    # Print sorted rows
    for shift in all_shifts:
        print(shift['row'])
    
    # Print summary statistics at the bottom
    print(f"\n{'='*150}")
    print("SUMMARY STATISTICS")
    print(f"{'='*150}")
    
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
    
    # Show performance breakdown
    print(f"\nPERFORMANCE BREAKDOWN:")
    if summary.get('total_shifts_analyzed', 0) > 0:
        completion_rate = summary.get('target_completion_rate', 0) * 100
        at_risk_rate = (summary.get('shifts_at_risk', 0) / summary.get('total_shifts_analyzed', 0)) * 100
        
        print(f"  • Target Achievement: {completion_rate:.1f}% of shifts meeting target")
        print(f"  • Risk Distribution: {at_risk_rate:.1f}% of shifts at high risk")
        print(f"  • Average Completion: {summary.get('shifts_meeting_target', 0)}/{summary.get('total_shifts_analyzed', 0)} shifts")


if __name__ == "__main__":
    main()