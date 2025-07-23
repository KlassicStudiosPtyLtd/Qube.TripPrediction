#!/usr/bin/env python3
"""
Debug script to analyze driver-based shifts without running full simulation
"""
import click
import logging
from datetime import datetime
import pytz

from core.mtdata_api_client import MTDataApiClient
from config import ShiftConfig
from analyzers.trip_extractor import TripExtractor
from analyzers.shift_analyzer import ShiftAnalyzer
from core.utils import setup_logging

# Set up logging
setup_logging('DEBUG')
logger = logging.getLogger(__name__)

def debug_driver_shifts(fleet_id: int, vehicle_ref: str, 
                       start_date: str, end_date: str, timezone: str = 'Australia/Perth',
                       shift_hours: float = 12, target_trips: int = 5, max_shift_hours: float = 15,
                       round_trip_mode: str = 'three_point', 
                       start_waypoint: str = 'QUBE Wedgefield Yard_PHEVG180415',
                       target_waypoint: str = 'PLS Mine site_PHEHR190325',
                       end_waypoint: str = 'QUBE Wedgefield Yard_PHEVG180415'):
    """Debug driver-based shift detection for a specific vehicle."""
    
    # Parse dates
    tz = pytz.timezone(timezone)
    start_dt = tz.localize(datetime.strptime(start_date, '%Y-%m-%d'))
    end_dt = tz.localize(datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59))
    
    start_utc = start_dt.astimezone(pytz.UTC)
    end_utc = end_dt.astimezone(pytz.UTC)
    
    print(f"Analyzing vehicle {vehicle_ref} from {start_date} to {end_date}")
    print(f"UTC range: {start_utc} to {end_utc}")
    
    # Initialize configuration
    shift_config = ShiftConfig(
        shift_detection_mode='driver_based',
        shift_duration_hours=shift_hours,
        max_shift_duration_hours=max_shift_hours,
        target_trips=target_trips,
        round_trip_mode=round_trip_mode,
        start_waypoint=start_waypoint,
        target_waypoint=target_waypoint,
        end_waypoint=end_waypoint,
        waypoint_matching='exact',
        timezone=timezone
    )
    
    # Initialize components
    api_client = MTDataApiClient()
    trip_extractor = TripExtractor(shift_config)
    shift_analyzer = ShiftAnalyzer(shift_config)
    
    # Get vehicles
    vehicles = api_client.get_vehicles(fleet_id)
    target_vehicle = None
    
    for vehicle in vehicles:
        # Get small sample to check vehicle ref
        try:
            sample_history = api_client.get_vehicle_history(
                fleet_id=fleet_id,
                vehicle_id=vehicle['id'],
                start_date=start_utc,
                end_date=start_utc.replace(hour=start_utc.hour + 1)
            )
            if sample_history and sample_history[0].get('vehicleRef') == vehicle_ref:
                target_vehicle = vehicle
                break
        except:
            continue
    
    if not target_vehicle:
        print(f"Vehicle with ref {vehicle_ref} not found")
        return
    
    vehicle_id = target_vehicle['id']
    vehicle_name = target_vehicle.get('displayName', f'Vehicle_{vehicle_id}')
    
    print(f"Found vehicle: {vehicle_name} (ID: {vehicle_id})")
    
    # Get vehicle history
    print("Fetching vehicle history...")
    history = api_client.get_vehicle_history(
        fleet_id=fleet_id,
        vehicle_id=vehicle_id,
        start_date=start_utc,
        end_date=end_utc
    )
    
    print(f"Retrieved {len(history)} history records")
    
    # Create vehicle data structure
    vehicle_data = {
        'vehicleId': vehicle_id,
        'fleetId': fleet_id,
        'history': history
    }
    
    # Extract trips
    print("\nExtracting trips...")
    trips = trip_extractor.extract_trips(vehicle_data)
    print(f"Extracted {len(trips)} trips")
    
    # Split trips at driver changes
    print("\nSplitting trips at driver changes...")
    split_trips = trip_extractor.split_trips_at_driver_changes(trips, history)
    print(f"After splitting: {len(split_trips)} trips")
    
    # Show trip details
    print("\nTrip Details:")
    for i, trip in enumerate(split_trips):
        driver_info = getattr(trip, 'driver_id', 'Unknown')
        print(f"  Trip {i+1}: {trip.start_time} to {trip.end_time}")
        print(f"    Driver: {driver_info}")
        print(f"    Waypoints: {len(trip.waypoints)} waypoints")
        print(f"    Type: {getattr(trip, 'trip_type', 'unknown')}")
        
    # Analyze shifts
    print("\nAnalyzing shifts...")
    shift_analyses = shift_analyzer.analyze_shifts(
        vehicle_id, vehicle_name, split_trips, timezone,
        history_data=history
    )
    
    print(f"Generated {len(shift_analyses)} shift analyses")
    
    # Show shift details
    print("\nShift Details:")
    for i, analysis in enumerate(shift_analyses):
        shift = analysis.shift
        print(f"  Shift {i+1}: {shift.start_time} to {shift.end_time}")
        print(f"    Driver: {shift.driver_id}")
        print(f"    Trips: {analysis.metrics.get('trips_completed', 0)}/{analysis.metrics.get('trips_target', 0)}")
        print(f"    Risk Level: {analysis.risk_level}")
        print(f"    Alert Required: {analysis.alert_required}")
        print(f"    Can Complete Target: {analysis.can_complete_target}")
        if shift.actual_end_time:
            print(f"    Actual End Time: {shift.actual_end_time}")

@click.command()
@click.option('--fleet-id', type=int, required=True, help='Fleet ID to analyze')
@click.option('--vehicle-ref', required=True, help='Vehicle reference (e.g., PRM00843)')
@click.option('--start-date', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', required=True, help='End date (YYYY-MM-DD)')
@click.option('--timezone', default='Australia/Perth', help='Timezone')
@click.option('--shift-hours', default=12, type=float, help='Shift duration in hours')
@click.option('--target-trips', default=5, type=int, help='Target trips per shift')
@click.option('--max-shift-hours', default=15.0, type=float, help='Maximum shift hours')
@click.option('--round-trip-mode', default='three_point', help='Round trip mode')
@click.option('--start-waypoint', default='QUBE Wedgefield Yard_PHEVG180415', help='Start waypoint')
@click.option('--target-waypoint', default='PLS Mine site_PHEHR190325', help='Target waypoint')
@click.option('--end-waypoint', default='QUBE Wedgefield Yard_PHEVG180415', help='End waypoint')
@click.option('--log-level', default='DEBUG', help='Logging level')
def main(fleet_id: int, vehicle_ref: str, start_date: str, end_date: str, timezone: str,
         shift_hours: float, target_trips: int, max_shift_hours: float, round_trip_mode: str,
         start_waypoint: str, target_waypoint: str, end_waypoint: str, log_level: str):
    """Debug driver-based shift detection for a specific vehicle."""
    
    setup_logging(log_level)
    
    debug_driver_shifts(
        fleet_id=fleet_id,
        vehicle_ref=vehicle_ref,
        start_date=start_date,
        end_date=end_date,
        timezone=timezone,
        shift_hours=shift_hours,
        target_trips=target_trips,
        max_shift_hours=max_shift_hours,
        round_trip_mode=round_trip_mode,
        start_waypoint=start_waypoint,
        target_waypoint=target_waypoint,
        end_waypoint=end_waypoint
    )

if __name__ == "__main__":
    main()