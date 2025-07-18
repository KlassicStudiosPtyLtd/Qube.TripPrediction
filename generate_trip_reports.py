#!/usr/bin/env python3
"""
Generate trip reports as a standalone process with caching support.

This script can be run independently to generate trip timeline reports,
using cached data when available to avoid repeated API calls.
"""
import click
import logging
from datetime import datetime
from pathlib import Path
import pytz

from core.mtdata_api_client import MTDataApiClient
from config import ShiftConfig
from analyzers.trip_report_generator import TripReportGenerator
from core.utils import setup_logging
from main import parse_datetime_string

logger = logging.getLogger(__name__)


@click.command()
@click.option('--fleet-id', type=int, required=True, help='Fleet ID to analyze')
@click.option('--start-date', required=True, 
              help='Start date/time in local timezone. Formats: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS')
@click.option('--end-date', required=True, 
              help='End date/time in local timezone. Formats: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS')
@click.option('--timezone', default='Australia/Perth', 
              help='Timezone for dates and report display')
@click.option('--output-dir', default='./trip_reports', 
              help='Output directory for reports')
@click.option('--cache-dir', default='./trip_report_cache',
              help='Directory for caching API data')
@click.option('--no-cache', is_flag=True, default=False,
              help='Disable cache usage (force API calls)')
@click.option('--clear-cache', is_flag=True, default=False,
              help='Clear cache before running')
@click.option('--vehicle-ref', default=None, 
              help='Filter to specific vehicle reference')
@click.option('--start-waypoint', help='Start waypoint for trip extraction')
@click.option('--target-waypoint', help='Target waypoint (for 3-point trips)')
@click.option('--end-waypoint', help='End waypoint for trip extraction')
@click.option('--round-trip-mode', 
              type=click.Choice(['simple', 'three_point']), 
              default='simple',
              help='Round trip mode for trip extraction')
@click.option('--log-level', default='INFO', help='Logging level')
def generate_trip_reports(fleet_id: int, start_date: str, end_date: str,
                         timezone: str, output_dir: str, cache_dir: str,
                         no_cache: bool, clear_cache: bool, vehicle_ref: str,
                         start_waypoint: str, target_waypoint: str,
                         end_waypoint: str, round_trip_mode: str,
                         log_level: str):
    """
    Generate trip timeline reports with caching support.
    
    This script will:
    1. Check cache for existing data
    2. Fetch from API only if cache miss
    3. Save fetched data to cache for future use
    4. Generate trip timeline reports
    
    Examples:
        # First run - fetches from API and caches
        python generate_trip_reports.py --fleet-id 104002 --start-date 2025-06-17 --end-date 2025-06-17
        
        # Second run - uses cached data (instant)
        python generate_trip_reports.py --fleet-id 104002 --start-date 2025-06-17 --end-date 2025-06-17
        
        # Force fresh API call
        python generate_trip_reports.py --fleet-id 104002 --start-date 2025-06-17 --end-date 2025-06-17 --no-cache
        
        # Clear cache and start fresh
        python generate_trip_reports.py --fleet-id 104002 --start-date 2025-06-17 --end-date 2025-06-17 --clear-cache
    """
    setup_logging(log_level)
    
    logger.info("="*60)
    logger.info("TRIP TIMELINE REPORT GENERATOR")
    logger.info("="*60)
    logger.info(f"Fleet ID: {fleet_id}")
    logger.info(f"Date Range: {start_date} to {end_date} ({timezone})")
    logger.info(f"Cache Directory: {cache_dir}")
    logger.info(f"Cache Enabled: {not no_cache}")
    
    try:
        # Parse dates
        start_datetime_utc = parse_datetime_string(start_date, timezone, is_end_date=False)
        end_datetime_utc = parse_datetime_string(end_date, timezone, is_end_date=True)
        
        # Validate date range
        if end_datetime_utc <= start_datetime_utc:
            logger.error("End date/time must be after start date/time")
            return
        
        # Initialize shift config for trip extraction
        shift_config = ShiftConfig(
            start_waypoint=start_waypoint,
            target_waypoint=target_waypoint,
            end_waypoint=end_waypoint,
            round_trip_mode=round_trip_mode,
            timezone=timezone
        )
        
        # Initialize report generator with cache
        report_generator = TripReportGenerator(timezone=timezone, cache_dir=cache_dir)
        
        # Clear cache if requested
        if clear_cache:
            logger.info("Clearing cache...")
            report_generator.clear_cache(fleet_id)
        
        # Disable cache if requested
        if no_cache:
            report_generator.set_cache_enabled(False)
        
        # Initialize API client
        api_logger = logging.getLogger('mtdata_api_client')
        api_client = MTDataApiClient(logger=api_logger)
        
        # Get vehicles
        logger.info("Fetching vehicle list...")
        vehicles = api_client.get_vehicles(fleet_id)
        
        if not vehicles:
            logger.error(f"No vehicles found for fleet {fleet_id}")
            return
        
        # Filter by vehicle ref if specified
        if vehicle_ref:
            filtered_vehicles = []
            for vehicle in vehicles:
                # Need to check vehicle ref - might need a small API call
                # For now, assume we'll filter after getting data
                filtered_vehicles.append(vehicle)
            if filtered_vehicles:
                vehicles = filtered_vehicles
        
        logger.info(f"Processing {len(vehicles)} vehicles...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate reports using the API client method
        report_path = report_generator.generate_trip_reports_from_api_client(
            api_client=api_client,
            fleet_id=fleet_id,
            vehicles=vehicles,
            start_datetime=start_datetime_utc,
            end_datetime=end_datetime_utc,
            output_path=output_path,
            shift_config=shift_config,
            use_cache=not no_cache
        )
        
        logger.info("="*60)
        logger.info("REPORT GENERATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Main report: {report_path}")
        logger.info(f"Summary report: {output_path / 'trip_timeline_summary.csv'}")
        
    except Exception as e:
        logger.error(f"Error generating reports: {str(e)}", exc_info=True)


if __name__ == "__main__":
    generate_trip_reports()