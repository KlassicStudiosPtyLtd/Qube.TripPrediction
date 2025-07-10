"""
Historical Simulation Engine for Fleet Shift Analysis

This module provides a framework for backtesting shift analysis algorithms
on historical data to validate predictions and alert generation.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path
import pytz
import pandas as pd
import numpy as np
from tqdm import tqdm

from config import ShiftConfig, AnalysisConfig
from models import Alert, Trip
from analyzers.trip_extractor import TripExtractor
from analyzers.shift_analyzer import ShiftAnalyzer
from monitoring.alert_manager import AlertManager

logger = logging.getLogger(__name__)


@dataclass
class SimulationPoint:
    """Represents a single point in the simulation timeline."""
    timestamp: datetime
    vehicle_id: int
    vehicle_name: str
    vehicle_ref: str
    shift_id: str
    trips_completed: int
    trips_predicted: int
    prediction_accuracy: Optional[float]
    alert_generated: bool
    alert_type: Optional[str]
    actual_outcome: Optional[str]
    prediction_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Results from a complete simulation run."""
    simulation_id: str
    start_time: datetime
    end_time: datetime
    timezone: str
    total_points: int
    alerts_generated: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    simulation_points: List[SimulationPoint]
    summary_by_vehicle: Dict[int, Dict[str, Any]]
    alert_timeline: List[Dict[str, Any]]
    vehicle_info: Dict[int, Dict[str, str]] = field(default_factory=dict)  # Store vehicle names and refs
    

class SimulationEngine:
    """Engine for running historical simulations of shift analysis."""
    
    def __init__(self, api_client, shift_config: ShiftConfig, 
                 analysis_config: AnalysisConfig,
                 cache_dir: Optional[str] = None):
        """
        Initialize the simulation engine.
        
        Args:
            api_client: MTDATA API client
            shift_config: Shift configuration
            analysis_config: Analysis configuration
            cache_dir: Directory for caching API responses (None to disable)
        """
        self.api_client = api_client
        self.shift_config = shift_config
        self.analysis_config = analysis_config
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Initialize components
        self.trip_extractor = TripExtractor(shift_config)
        self.shift_analyzer = ShiftAnalyzer(shift_config)
        self.alert_manager = AlertManager()
        
        # Storage
        self.simulation_points = []
        self.alert_history = []
        self.vehicle_info = {}  # Store vehicle names and refs
        
        # Create cache directory if specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cache directory: {self.cache_dir}")
    
    def _get_cache_filename(self, fleet_id: int, vehicle_id: int, 
                           start_date: datetime, end_date: datetime) -> Path:
        """Generate cache filename for vehicle history."""
        start_str = start_date.strftime('%Y%m%d_%H%M%S')
        end_str = end_date.strftime('%Y%m%d_%H%M%S')
        return self.cache_dir / f"fleet_{fleet_id}_vehicle_{vehicle_id}_{start_str}_to_{end_str}.json"
    
    def _load_cached_history(self, cache_file: Path) -> Optional[List[Dict]]:
        """Load vehicle history from cache if available."""
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.debug(f"Loaded cached data from {cache_file.name}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
        return None
    
    def _save_to_cache(self, cache_file: Path, data: List[Dict]):
        """Save vehicle history to cache."""
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug(f"Saved data to cache: {cache_file.name}")
        except Exception as e:
            logger.warning(f"Failed to save cache file {cache_file}: {e}")
    
    def run_simulation(self, fleet_id: int, start_date: datetime, end_date: datetime,
                      simulation_interval_hours: float = 1.0,
                      timezone: str = 'UTC',
                      vehicles_to_simulate: Optional[List[int]] = None,
                      use_cache: bool = True,
                      interactive_mode: bool = False) -> SimulationResult:
        """
        Run a historical simulation over the specified time period.
        
        Args:
            fleet_id: Fleet ID to simulate
            start_date: Start of simulation period (UTC)
            end_date: End of simulation period (UTC)
            simulation_interval_hours: How often to run analysis (default: every hour)
            timezone: Timezone for display and shift boundaries
            vehicles_to_simulate: Specific vehicles to simulate (None = all vehicles)
            use_cache: Whether to use cached API responses if available
            interactive_mode: If True, pause after each time period to show alerts
            
        Returns:
            SimulationResult with complete analysis
        """
        # Ensure dates are timezone-aware
        if start_date.tzinfo is None:
            start_date = pytz.UTC.localize(start_date)
        if end_date.tzinfo is None:
            end_date = pytz.UTC.localize(end_date)
            
        logger.info(f"Starting historical simulation for fleet {fleet_id}")
        logger.info(f"Simulation period: {start_date} to {end_date} (UTC)")
        logger.info(f"Simulation interval: {simulation_interval_hours} hours")
        logger.info(f"Interactive mode: {'ENABLED' if interactive_mode else 'DISABLED'}")
        
        # Get vehicles
        vehicles = self._get_fleet_vehicles(fleet_id)
        if vehicles_to_simulate:
            vehicles = [v for v in vehicles if v['id'] in vehicles_to_simulate]
        
        if not vehicles:
            logger.error("No vehicles found for simulation")
            return self._create_empty_result(start_date, end_date, timezone)
        
        # Store vehicle info
        for vehicle in vehicles:
            self.vehicle_info[vehicle['id']] = {
                'name': vehicle.get('displayName', f'Vehicle_{vehicle["id"]}'),
                'ref': 'Unknown'  # Will be updated when we get history
            }
        
        logger.info(f"Simulating {len(vehicles)} vehicles")
        
        # Get all historical data for the period (with buffer for context)
        buffer_days = 1  # Extra day before start for context
        data_start = start_date - timedelta(days=buffer_days)
        all_vehicle_data = self._load_all_vehicle_data(
            vehicles, fleet_id, data_start, end_date, use_cache
        )
        
        # Run simulation at each time point
        current_time = start_date
        simulation_points = []
        alert_timeline = []
        
        # Ensure current_time is timezone-aware
        if current_time.tzinfo is None:
            current_time = pytz.UTC.localize(current_time)
        
        # Import helper functions if interactive mode is enabled
        if interactive_mode:
            try:
                from run_simulation import print_time_period_alerts, wait_for_user_input
            except ImportError:
                logger.warning("Could not import interactive functions from run_simulation module")
                interactive_mode = False
        
        with tqdm(total=int((end_date - start_date).total_seconds() / 3600 / simulation_interval_hours),
                  desc="Running simulation", 
                  disable=interactive_mode) as pbar:  # Disable progress bar in interactive mode
            
            while current_time <= end_date:
                # Run analysis at this time point
                point_results = self._simulate_time_point(
                    current_time, all_vehicle_data, fleet_id, timezone
                )
                
                simulation_points.extend(point_results['points'])
                current_period_alerts = point_results['alerts']
                alert_timeline.extend(current_period_alerts)
                
                # Interactive mode: show alerts and wait for user input
                if interactive_mode:
                    try:
                        print_time_period_alerts(current_time, current_period_alerts, timezone)
                        if current_time < end_date:  # Don't wait after the last period
                            wait_for_user_input()
                    except KeyboardInterrupt:
                        logger.info("Simulation interrupted by user")
                        break
                    except Exception as e:
                        logger.warning(f"Error in interactive mode: {e}")
                        # Continue with simulation even if interactive functions fail
                else:
                    pbar.update(1)
                
                # Move to next time point
                current_time += timedelta(hours=simulation_interval_hours)
        
        # Analyze results
        result = self._analyze_simulation_results(
            simulation_points, alert_timeline, start_date, end_date, timezone
        )
        
        return result
    
    def _simulate_time_point(self, simulation_time: datetime, 
                            all_vehicle_data: Dict[int, List[Dict]],
                            fleet_id: int, timezone: str) -> Dict[str, Any]:
        """Simulate analysis at a specific point in time."""
        points = []
        alerts = []
        
        for vehicle_id, full_history in all_vehicle_data.items():
            # Get vehicle info
            vehicle_name = self.vehicle_info.get(vehicle_id, {}).get('name', f'Vehicle_{vehicle_id}')
            vehicle_ref = self.vehicle_info.get(vehicle_id, {}).get('ref', 'Unknown')
            
            # Filter data to only what would be available at simulation_time
            available_data = []
            for event in full_history:
                try:
                    # Parse the event timestamp
                    event_time_str = event.get('deviceTimeUtc', '')
                    if not event_time_str:
                        continue
                    
                    # Handle 'Z' suffix for UTC
                    if event_time_str.endswith('Z'):
                        event_time_str = event_time_str[:-1] + '+00:00'
                    
                    # Parse the timestamp
                    event_time = datetime.fromisoformat(event_time_str)
                    
                    # Ensure it's timezone-aware
                    if event_time.tzinfo is None:
                        event_time = pytz.UTC.localize(event_time)
                    
                    # Only include events before or at simulation time
                    if event_time <= simulation_time:
                        available_data.append(event)
                        
                except (ValueError, AttributeError) as e:
                    logger.debug(f"Skipping event with invalid timestamp: {e}")
                    continue
            
            if not available_data:
                continue
            
            # Create vehicle data structure
            vehicle_data = {
                'vehicleId': vehicle_id,
                'fleetId': fleet_id,
                'history': available_data
            }
            
            # Extract trips from available data
            trips = self.trip_extractor.extract_trips(vehicle_data)
            
            if not trips:
                continue
            
            # Analyze shifts with data up to simulation_time
            shift_analyses = self.shift_analyzer.analyze_shifts(
                vehicle_id, vehicle_name, trips, timezone,
                analysis_end_time=simulation_time
            )
            
            # Process each shift
            for shift_analysis in shift_analyses:
                shift = shift_analysis.shift
                
                # Only process active shifts (those that include the simulation time)
                if shift.start_time <= simulation_time <= shift.end_time:
                    # Create simulation point
                    point = self._create_simulation_point(
                        simulation_time, vehicle_id, vehicle_name, vehicle_ref, 
                        shift, shift_analysis
                    )
                    points.append(point)
                    
                    # Check for alerts
                    if shift_analysis.alert_required:
                        alert = self._create_simulation_alert(
                            simulation_time, vehicle_id, vehicle_name, vehicle_ref,
                            shift_analysis
                        )
                        alerts.append(alert)
        
        return {
            'points': points,
            'alerts': alerts
        }
    
    def _create_simulation_point(self, timestamp: datetime, vehicle_id: int,
                                vehicle_name: str, vehicle_ref: str,
                                shift: Any, shift_analysis: Any) -> SimulationPoint:
        """Create a simulation point from shift analysis."""
        metrics = shift_analysis.metrics
        
        return SimulationPoint(
            timestamp=timestamp,
            vehicle_id=vehicle_id,
            vehicle_name=vehicle_name,
            vehicle_ref=vehicle_ref,
            shift_id=shift.shift_id,
            trips_completed=metrics.get('trips_completed', 0),
            trips_predicted=metrics.get('trips_target', 0),
            prediction_accuracy=None,  # Will be calculated later
            alert_generated=shift_analysis.alert_required,
            alert_type=shift_analysis.risk_level if shift_analysis.alert_required else None,
            actual_outcome=None,  # Will be filled in post-processing
            prediction_details={
                'can_complete_target': shift_analysis.can_complete_target,
                'risk_level': shift_analysis.risk_level,
                'recommendation': shift_analysis.recommendation,
                'remaining_time': metrics.get('remaining_time', 0),
                'estimated_time_needed': metrics.get('total_estimated_time', 0)
            }
        )
    
    def _create_simulation_alert(self, timestamp: datetime, vehicle_id: int,
                                vehicle_name: str, vehicle_ref: str,
                                shift_analysis: Any) -> Dict[str, Any]:
        """Create alert record for simulation."""
        return {
            'timestamp': timestamp.isoformat(),
            'vehicle_id': vehicle_id,
            'vehicle_name': vehicle_name,
            'vehicle_ref': vehicle_ref,
            'shift_id': shift_analysis.shift.shift_id,
            'alert_type': shift_analysis.risk_level,
            'message': shift_analysis.recommendation,
            'predicted_outcome': not shift_analysis.can_complete_target,
            'metrics': shift_analysis.metrics
        }
    
    def _analyze_simulation_results(self, simulation_points: List[SimulationPoint],
                                   alert_timeline: List[Dict[str, Any]],
                                   start_time: datetime, end_time: datetime,
                                   timezone: str) -> SimulationResult:
        """Analyze simulation results and calculate metrics."""
        # Group points by shift to determine actual outcomes
        shifts_data = {}
        for point in simulation_points:
            key = (point.vehicle_id, point.shift_id)
            if key not in shifts_data:
                shifts_data[key] = []
            shifts_data[key].append(point)
        
        # Determine actual outcomes for each shift
        actual_outcomes = {}
        for (vehicle_id, shift_id), points in shifts_data.items():
            # Get the final state of the shift
            final_point = max(points, key=lambda p: p.timestamp)
            trips_completed = final_point.trips_completed
            trips_target = final_point.trips_predicted
            
            # Did the shift meet its target?
            actual_outcome = trips_completed >= trips_target
            actual_outcomes[(vehicle_id, shift_id)] = actual_outcome
        
        # Calculate prediction accuracy
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        # Check each alert
        for alert in alert_timeline:
            vehicle_id = alert['vehicle_id']
            shift_id = alert['shift_id']
            predicted_failure = alert['predicted_outcome']
            
            actual_outcome = actual_outcomes.get((vehicle_id, shift_id))
            if actual_outcome is not None:
                actual_failure = not actual_outcome
                
                if predicted_failure and actual_failure:
                    true_positives += 1
                elif predicted_failure and not actual_failure:
                    false_positives += 1
                elif not predicted_failure and actual_failure:
                    false_negatives += 1
                else:
                    true_negatives += 1
        
        # Calculate metrics
        total_predictions = true_positives + false_positives + true_negatives + false_negatives
        accuracy = (true_positives + true_negatives) / total_predictions if total_predictions > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Create vehicle summaries
        vehicle_summaries = self._create_vehicle_summaries(simulation_points, alert_timeline)
        
        return SimulationResult(
            simulation_id=f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=start_time,
            end_time=end_time,
            timezone=timezone,
            total_points=len(simulation_points),
            alerts_generated=len(alert_timeline),
            true_positives=true_positives,
            false_positives=false_positives,
            true_negatives=true_negatives,
            false_negatives=false_negatives,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            simulation_points=simulation_points,
            summary_by_vehicle=vehicle_summaries,
            alert_timeline=alert_timeline,
            vehicle_info=self.vehicle_info
        )
    
    def _create_vehicle_summaries(self, simulation_points: List[SimulationPoint],
                                 alert_timeline: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """Create summary statistics for each vehicle."""
        summaries = {}
        
        # Group by vehicle
        from collections import defaultdict
        vehicle_points = defaultdict(list)
        vehicle_alerts = defaultdict(list)
        
        for point in simulation_points:
            vehicle_points[point.vehicle_id].append(point)
        
        for alert in alert_timeline:
            vehicle_alerts[alert['vehicle_id']].append(alert)
        
        # Create summaries
        for vehicle_id in vehicle_points:
            points = vehicle_points[vehicle_id]
            alerts = vehicle_alerts[vehicle_id]
            
            summaries[vehicle_id] = {
                'vehicle_name': self.vehicle_info.get(vehicle_id, {}).get('name', f'Vehicle_{vehicle_id}'),
                'vehicle_ref': self.vehicle_info.get(vehicle_id, {}).get('ref', 'Unknown'),
                'total_shifts': len(set(p.shift_id for p in points)),
                'total_alerts': len(alerts),
                'alert_types': dict(pd.Series([a['alert_type'] for a in alerts]).value_counts()),
                'average_trips_completed': np.mean([p.trips_completed for p in points]),
                'simulation_points': len(points)
            }
        
        return summaries
    
    def _get_fleet_vehicles(self, fleet_id: int) -> List[Dict[str, Any]]:
        """Get all vehicles for a fleet."""
        try:
            vehicles = self.api_client.get_vehicles(fleet_id)
            return vehicles if isinstance(vehicles, list) else []
        except Exception as e:
            logger.error(f"Error getting vehicles: {str(e)}")
            return []
    
    def _load_all_vehicle_data(self, vehicles: List[Dict], fleet_id: int,
                              start_date: datetime, end_date: datetime,
                              use_cache: bool = True) -> Dict[int, List[Dict]]:
        """Load all vehicle data for the simulation period, with optional caching."""
        all_data = {}
        
        # Determine if we should use cache
        cache_enabled = use_cache and self.cache_dir is not None
        if cache_enabled:
            logger.info("Cache enabled for vehicle history data")
        else:
            logger.info("Cache disabled or not configured")
        
        for vehicle in tqdm(vehicles, desc="Loading vehicle data"):
            vehicle_id = vehicle['id']
            
            try:
                history = None
                
                # Try to load from cache first
                if cache_enabled:
                    cache_file = self._get_cache_filename(fleet_id, vehicle_id, start_date, end_date)
                    history = self._load_cached_history(cache_file)
                
                # If not in cache or cache disabled, fetch from API
                if history is None:
                    logger.debug(f"Fetching data from API for vehicle {vehicle_id}")
                    history = self.api_client.get_vehicle_history(
                        fleet_id=fleet_id,
                        vehicle_id=vehicle_id,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    # Save to cache if enabled
                    if cache_enabled and history:
                        cache_file = self._get_cache_filename(fleet_id, vehicle_id, start_date, end_date)
                        self._save_to_cache(cache_file, history)
                
                # Extract vehicleRef from first history record if available
                if history and len(history) > 0:
                    vehicle_ref = history[0].get('vehicleRef', 'Unknown')
                    if vehicle_id in self.vehicle_info:
                        self.vehicle_info[vehicle_id]['ref'] = vehicle_ref
                
                all_data[vehicle_id] = history if history else []
                
            except Exception as e:
                logger.error(f"Error loading data for vehicle {vehicle_id}: {str(e)}")
                all_data[vehicle_id] = []
        
        return all_data
    
    def _create_empty_result(self, start_time: datetime, end_time: datetime,
                           timezone: str) -> SimulationResult:
        """Create empty result when no data is available."""
        return SimulationResult(
            simulation_id=f"sim_empty_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=start_time,
            end_time=end_time,
            timezone=timezone,
            total_points=0,
            alerts_generated=0,
            true_positives=0,
            false_positives=0,
            true_negatives=0,
            false_negatives=0,
            accuracy=0,
            precision=0,
            recall=0,
            f1_score=0,
            simulation_points=[],
            summary_by_vehicle={},
            alert_timeline=[],
            vehicle_info={}
        )
    
    def save_simulation_results(self, result: SimulationResult, output_dir: Path):
        """Save simulation results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        result_dict = {
            'simulation_id': result.simulation_id,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat(),
            'timezone': result.timezone,
            'metrics': {
                'total_points': result.total_points,
                'alerts_generated': result.alerts_generated,
                'accuracy': result.accuracy,
                'precision': result.precision,
                'recall': result.recall,
                'f1_score': result.f1_score,
                'confusion_matrix': {
                    'true_positives': result.true_positives,
                    'false_positives': result.false_positives,
                    'true_negatives': result.true_negatives,
                    'false_negatives': result.false_negatives
                }
            },
            'summary_by_vehicle': result.summary_by_vehicle,
            'alert_timeline': result.alert_timeline,
            'vehicle_info': result.vehicle_info
        }
        
        with open(output_dir / f'{result.simulation_id}_results.json', 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        # Save detailed simulation points
        points_data = []
        for point in result.simulation_points:
            points_data.append({
                'timestamp': point.timestamp.isoformat(),
                'vehicle_id': point.vehicle_id,
                'vehicle_name': point.vehicle_name,
                'vehicle_ref': point.vehicle_ref,
                'shift_id': point.shift_id,
                'trips_completed': point.trips_completed,
                'trips_predicted': point.trips_predicted,
                'alert_generated': point.alert_generated,
                'alert_type': point.alert_type,
                'prediction_details': point.prediction_details
            })
        
        df = pd.DataFrame(points_data)
        df.to_csv(output_dir / f'{result.simulation_id}_timeline.csv', index=False)
        
        logger.info(f"Simulation results saved to {output_dir}")