#!/usr/bin/env python3
"""
Run historical simulation for fleet shift analysis with three-point round trip support
"""
import click
import logging
from datetime import datetime
from pathlib import Path
import pytz

from core.mtdata_api_client import MTDataApiClient
from config import ShiftConfig, AnalysisConfig
from simulation_engine import SimulationEngine
from core.utils import setup_logging
from main import parse_datetime_string

logger = logging.getLogger(__name__)


def print_time_period_alerts(current_time: datetime, alerts: list, timezone: str):
    """Print alerts generated during the current time period with detailed algorithm explanations."""
    if not alerts:
        print(f"\n⏰ {current_time.astimezone(pytz.timezone(timezone)).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print("   ✅ No alerts generated this period")
        return
    
    print(f"\n⏰ {current_time.astimezone(pytz.timezone(timezone)).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"   🚨 {len(alerts)} alert(s) generated:")
    
    for i, alert in enumerate(alerts, 1):
        vehicle_name = alert.get('vehicle_name', f"Vehicle {alert['vehicle_id']}")
        vehicle_ref = alert.get('vehicle_ref', 'Unknown')
        risk_level = alert.get('alert_type', 'unknown').upper()
        
        # Color coding
        if risk_level in ['HIGH', 'CRITICAL']:
            color = '\033[91m'  # Red
        elif risk_level == 'MEDIUM':
            color = '\033[93m'  # Yellow
        else:
            color = '\033[94m'  # Blue
        reset = '\033[0m'
        
        print(f"\n   {i}. {color}[{risk_level}]{reset} {vehicle_name} ({vehicle_ref})")
        print(f"      → {alert.get('message', 'No message')}")
        
        # Show trip progress
        metrics = alert.get('metrics', {})
        trips_completed = metrics.get('trips_completed', 0)
        trips_target = metrics.get('trips_target', 0)
        print(f"      → Progress: {trips_completed}/{trips_target} trips completed")
        
        # Show three-point specific metrics if applicable
        if 'incomplete_round_trips' in metrics and metrics['incomplete_round_trips'] > 0:
            print(f"      → Incomplete round trips: {metrics['incomplete_round_trips']}")
        
        # Show detailed algorithm explanation if in verbose mode
        if alert.get('algorithm_details'):
            print(f"\n      📊 ALGORITHM DETAILS:")
            # Indent each line of the algorithm details
            for line in alert['algorithm_details'].split('\n'):
                if line.strip():
                    print(f"         {line}")
        
        # Show key metrics
        print(f"\n      ⏱️  KEY METRICS:")
        print(f"         • Time into shift: {metrics.get('time_into_shift_hours', 0):.1f} hours")
        print(f"         • Remaining time: {metrics.get('remaining_time', 0):.0f} minutes")
        print(f"         • Estimated time needed: {metrics.get('total_estimated_time', 0):.0f} minutes")
        print(f"         • Completion probability: {metrics.get('completion_probability', 0):.1%}")
        
        print(f"      {'─' * 60}")


def wait_for_user_input():
    """Wait for user to press Enter to continue or 'd' for detailed view."""
    try:
        response = input("\n   Press Enter to continue simulation (or 'd' for more details)...")
        if response.lower() == 'd':
            print("\n   Detailed mode activated for next period.")
            return 'detailed'
        return 'continue'
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        raise


@click.command()
@click.option('--fleet-id', type=int, required=True, help='Fleet ID to simulate')
@click.option('--start-date', required=True, 
              help='Start date/time for simulation. Formats: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS')
@click.option('--end-date', required=True, 
              help='End date/time for simulation. Formats: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS')
@click.option('--timezone', default='Australia/Perth', 
              help='Timezone for dates and shift boundaries')
@click.option('--interval-hours', default=1.0, type=float,
              help='How often to run analysis during simulation (default: every hour)')
@click.option('--output-dir', default='./simulation_results', 
              help='Output directory for results')
@click.option('--vehicles', multiple=True, type=int,
              help='Specific vehicle IDs to simulate (default: all vehicles)')
@click.option('--shift-hours', default=12, type=float, help='Shift duration in hours')
@click.option('--target-trips', default=4, type=int, help='Target trips per shift')
@click.option('--buffer-minutes', default=30, type=int, help='Buffer time before shift end')
@click.option('--default-trip-duration', default=45, type=float, 
              help='Default trip duration in minutes')
@click.option('--start-waypoint', default=None, help='Start waypoint for trips')
@click.option('--target-waypoint', default=None, 
              help='Target/intermediate waypoint for three-point round trips')
@click.option('--end-waypoint', default=None, help='End waypoint for trips')
@click.option('--waypoint-matching', 
              type=click.Choice(['exact', 'contains', 'normalized']), 
              default='exact')
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
@click.option('--cache-dir', default='./simulation_cache',
              help='Directory for caching API responses (set to empty string to disable)')
@click.option('--no-cache', is_flag=True, default=False,
              help='Disable caching even if cache-dir is set')
@click.option('--interactive', is_flag=True, default=False,
              help='Interactive mode: pause after each time period to show alerts and wait for user input')
@click.option('--log-level', default='INFO', help='Logging level')
def run_simulation(fleet_id: int, start_date: str, end_date: str, timezone: str,
                  interval_hours: float, output_dir: str, vehicles: tuple,
                  shift_hours: float, target_trips: int, buffer_minutes: int,
                  default_trip_duration: float, start_waypoint: str,
                  target_waypoint: str, end_waypoint: str, waypoint_matching: str,
                  round_trip_mode: str, require_waypoint_order: bool,
                  allow_partial_trips: bool, segment_duration_start_target: float,
                  segment_duration_target_end: float,
                  cache_dir: str, no_cache: bool, interactive: bool, log_level: str):
    """
    Run historical simulation to test shift analysis algorithms with three-point support.
    
    This will simulate running the analysis at regular intervals throughout
    the specified time period, showing what alerts would have been generated
    and how accurate the predictions were.
    
    Examples:
        # Simple mode simulation
        run_simulation.py --fleet-id 104002 --start-date 2025-06-10 --end-date 2025-06-17
        
        # Three-point round trip simulation
        run_simulation.py --fleet-id 104002 --start-date 2025-06-15 --end-date 2025-06-17 \\
            --round-trip-mode three_point \\
            --start-waypoint "Depot_A" \\
            --target-waypoint "Customer_Site" \\
            --end-waypoint "Depot_A" \\
            --segment-duration-start-target 60 \\
            --segment-duration-target-end 45
            
        # Interactive mode with specific vehicles
        run_simulation.py --fleet-id 104002 --start-date "2025-06-15 06:00:00" \\
            --end-date "2025-06-15 18:00:00" --interval-hours 0.5 \\
            --vehicles 1234 --vehicles 5678 --interactive
    """
    # Set up logging
    setup_logging(log_level)
    
    logger.info("="*60)
    logger.info("FLEET SHIFT ANALYSIS - HISTORICAL SIMULATION")
    logger.info("="*60)
    logger.info(f"Fleet ID: {fleet_id}")
    logger.info(f"Simulation Period: {start_date} to {end_date} ({timezone})")
    logger.info(f"Analysis Interval: {interval_hours} hours")
    logger.info(f"Round Trip Mode: {round_trip_mode}")
    logger.info(f"Interactive Mode: {'ENABLED' if interactive else 'DISABLED'}")
    
    # Validate waypoint configuration
    if round_trip_mode == 'three_point':
        if not all([start_waypoint, target_waypoint, end_waypoint]):
            logger.error("Three-point mode requires all waypoints: --start-waypoint, --target-waypoint, --end-waypoint")
            return
        logger.info(f"Three-Point Route: {start_waypoint} → {target_waypoint} → {end_waypoint}")
        logger.info(f"Waypoint Order Required: {require_waypoint_order}")
        logger.info(f"Allow Partial Trips: {allow_partial_trips}")
    elif start_waypoint and end_waypoint:
        logger.info(f"Simple Route: {start_waypoint} → {end_waypoint}")
    
    if vehicles:
        logger.info(f"Simulating specific vehicles: {list(vehicles)}")
    else:
        logger.info("Simulating all fleet vehicles")
    
    try:
        # Parse dates
        start_datetime_utc = parse_datetime_string(start_date, timezone, is_end_date=False)
        end_datetime_utc = parse_datetime_string(end_date, timezone, is_end_date=True)
        
        # Validate date range
        if end_datetime_utc <= start_datetime_utc:
            logger.error("End date/time must be after start date/time")
            return
        
        # Prepare segment durations
        default_segment_durations = {
            'start_to_target': segment_duration_start_target or default_trip_duration,
            'target_to_end': segment_duration_target_end or default_trip_duration
        }
        
        # Initialize configurations
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
            parallel_processing=False,  # Sequential for simulation
            save_individual_analyses=False,
            generate_visualizations=False
        )
        
        # Initialize API client
        api_logger = logging.getLogger('mtdata_api_client')
        api_client = MTDataApiClient(logger=api_logger)
        
        # Create simulation engine with cache configuration
        cache_directory = None if no_cache or not cache_dir else cache_dir
        engine = SimulationEngine(api_client, shift_config, analysis_config, 
                                 cache_dir=cache_directory)
        
        # Run simulation with interactive mode
        logger.info("\nStarting simulation...")
        if cache_directory:
            logger.info(f"Using cache directory: {cache_directory}")
        else:
            logger.info("Caching disabled")
            
        result = engine.run_simulation(
            fleet_id=fleet_id,
            start_date=start_datetime_utc,
            end_date=end_datetime_utc,
            simulation_interval_hours=interval_hours,
            timezone=timezone,
            vehicles_to_simulate=list(vehicles) if vehicles else None,
            use_cache=not no_cache,
            interactive_mode=interactive
        )
        
        # Save results
        output_path = Path(output_dir)
        engine.save_simulation_results(result, output_path)
        
        # Print summary
        print_simulation_summary(result, output_path, shift_config)
        
    except Exception as e:
        logger.error(f"Error during simulation: {str(e)}", exc_info=True)


def print_simulation_summary(result, output_path: Path, shift_config: ShiftConfig):
    """Print simulation results summary with three-point support."""
    print(f"\n{'='*60}")
    print("SIMULATION RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Simulation ID: {result.simulation_id}")
    print(f"Period: {result.start_time.strftime('%Y-%m-%d %H:%M')} to {result.end_time.strftime('%Y-%m-%d %H:%M')} ({result.timezone})")
    print(f"Round Trip Mode: {shift_config.round_trip_mode}")
    
    if shift_config.round_trip_mode == 'three_point':
        print(f"Route: {shift_config.start_waypoint} → {shift_config.target_waypoint} → {shift_config.end_waypoint}")
    
    print(f"Total Analysis Points: {result.total_points}")
    print(f"Total Alerts Generated: {result.alerts_generated}")
    
    print(f"\n{'PREDICTION ACCURACY':^60}")
    print(f"{'-'*60}")
    print(f"Overall Accuracy: {result.accuracy:.1%}")
    print(f"Precision: {result.precision:.1%} (of alerts generated, how many were correct)")
    print(f"Recall: {result.recall:.1%} (of actual problems, how many were caught)")
    print(f"F1 Score: {result.f1_score:.1%}")
    
    print(f"\n{'CONFUSION MATRIX':^60}")
    print(f"{'-'*60}")
    print(f"True Positives:  {result.true_positives:4d} (correctly predicted problems)")
    print(f"False Positives: {result.false_positives:4d} (false alarms)")
    print(f"True Negatives:  {result.true_negatives:4d} (correctly predicted success)")
    print(f"False Negatives: {result.false_negatives:4d} (missed problems)")
    
    # Three-point specific statistics
    if shift_config.round_trip_mode == 'three_point':
        print(f"\n{'THREE-POINT ROUND TRIP STATISTICS':^60}")
        print(f"{'-'*60}")
        
        # Calculate round trip completion statistics
        total_complete = 0
        total_incomplete = 0
        
        for point in result.simulation_points:
            if 'incomplete_round_trips' in point.prediction_details:
                incomplete = point.prediction_details['incomplete_round_trips']
                if incomplete > 0:
                    total_incomplete += 1
        
        if total_incomplete > 0:
            print(f"Analysis points with incomplete round trips: {total_incomplete}")
            print(f"Percentage of points with incomplete trips: {total_incomplete/result.total_points:.1%}")
    
    if result.summary_by_vehicle:
        print(f"\n{'VEHICLE SUMMARY':^80}")
        print(f"{'-'*80}")
        print(f"{'Vehicle Name (Ref)':<30} {'Shifts':<8} {'Alerts':<8} {'Alert Types':<30}")
        print(f"{'-'*80}")
        
        for vehicle_id, summary in sorted(result.summary_by_vehicle.items()):
            # Get vehicle name and ref from summary
            vehicle_name = summary.get('vehicle_name', f'Vehicle_{vehicle_id}')
            vehicle_ref = summary.get('vehicle_ref', 'Unknown')
            vehicle_display = f"{vehicle_name} ({vehicle_ref})"
            
            # Truncate if too long
            if len(vehicle_display) > 29:
                vehicle_display = vehicle_display[:28] + '.'
            
            alert_types = ', '.join([f"{k}:{v}" for k, v in summary['alert_types'].items()])
            print(f"{vehicle_display:<30} {summary['total_shifts']:<8} "
                  f"{summary['total_alerts']:<8} {alert_types:<30}")
    
    # Show sample alerts
    if result.alert_timeline:
        print(f"\n{'SAMPLE ALERTS (First 5)':^60}")
        print(f"{'-'*60}")
        
        for i, alert in enumerate(result.alert_timeline[:5]):
            timestamp = datetime.fromisoformat(alert['timestamp'].replace('Z', '+00:00'))
            local_time = timestamp.astimezone(pytz.timezone(result.timezone))
            
            # Get vehicle display info from alert
            vehicle_name = alert.get('vehicle_name', f"Vehicle {alert['vehicle_id']}")
            vehicle_ref = alert.get('vehicle_ref', 'Unknown')
            vehicle_display = f"{vehicle_name} ({vehicle_ref})"
            
            print(f"\n{i+1}. {local_time.strftime('%Y-%m-%d %H:%M:%S')} - {vehicle_display}")
            print(f"   Type: {alert['alert_type'].upper()}")
            print(f"   Message: {alert['message']}")
            
            metrics = alert.get('metrics', {})
            print(f"   Metrics: Trips {metrics.get('trips_completed')}/{metrics.get('trips_target')}")
            
            if shift_config.round_trip_mode == 'three_point' and 'incomplete_round_trips' in metrics:
                print(f"   Incomplete Round Trips: {metrics['incomplete_round_trips']}")

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"  - Summary: {result.simulation_id}_results.json")
    print(f"  - Timeline: {result.simulation_id}_timeline.csv")
    print(f"  - Alerts: {result.simulation_id}_alerts_*.csv")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_simulation()