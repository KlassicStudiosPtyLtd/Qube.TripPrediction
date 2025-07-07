"""
Main fleet analyzer that orchestrates the analysis with timezone support.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pytz

from tqdm import tqdm

from config import ShiftConfig, AnalysisConfig
from models import Alert
from analyzers.trip_extractor import TripExtractor
from analyzers.shift_analyzer import ShiftAnalyzer
from monitoring.alert_manager import AlertManager
from visualization.dashboard_generator import DashboardGenerator
from visualization.map_generator import MapGenerator

from core.mtdata_api_client import MTDataApiClient
from core.utils import setup_logging

logger = logging.getLogger(__name__)


class FleetAnalyzer:
    """Main analyzer for fleet-wide shift analysis with timezone support."""
    
    def __init__(self, api_client, shift_config: ShiftConfig, 
                 analysis_config: AnalysisConfig):
        self.api_client = api_client
        self.shift_config = shift_config
        self.analysis_config = analysis_config
        
        # Initialize components
        self.trip_extractor = TripExtractor(shift_config)
        self.shift_analyzer = ShiftAnalyzer(shift_config)
        self.alert_manager = AlertManager()
        self.dashboard_generator = DashboardGenerator()
        self.map_generator = MapGenerator()
        
        # Storage
        self.fleet_data = {}
        self.vehicle_analyses = {}
        self.alerts = []
        
    def analyze_fleet(self, fleet_id: int, start_datetime: datetime, end_datetime: datetime,
                     output_dir: str, timezone: str = 'UTC') -> Dict[str, Any]:
        """
        Perform complete fleet analysis.
        
        Args:
            fleet_id: Fleet ID
            start_datetime: Start datetime (UTC)
            end_datetime: End datetime (UTC)
            output_dir: Output directory path
            timezone: Timezone for display (e.g., 'Australia/Perth')
            
        Returns:
            Fleet analysis summary
        """
        logger.info(f"Starting fleet analysis for fleet {fleet_id}")
        logger.info(f"Analysis period (UTC): {start_datetime} to {end_datetime}")
        logger.info(f"Display timezone: {timezone}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get vehicles
        vehicles = self._get_fleet_vehicles(fleet_id)
        if not vehicles:
            logger.error(f"No vehicles found for fleet {fleet_id}")
            return {}
        
        logger.info(f"Found {len(vehicles)} vehicles")
        
        # Analyze vehicles
        if self.analysis_config.parallel_processing:
            self._analyze_vehicles_parallel(
                vehicles, fleet_id, start_datetime, end_datetime, output_path, timezone
            )
        else:
            self._analyze_vehicles_sequential(
                vehicles, fleet_id, start_datetime, end_datetime, output_path, timezone
            )
        
        # Generate fleet summary
        fleet_summary = self._generate_fleet_summary()
        
        # Add timezone info to summary
        fleet_summary['timezone'] = timezone
        fleet_summary['analysis_period'] = {
            'start_utc': start_datetime.isoformat(),
            'end_utc': end_datetime.isoformat(),
            'timezone': timezone
        }
        
        # Create visualizations
        if self.analysis_config.generate_visualizations:
            self._create_visualizations(fleet_summary, output_path)
        
        # Save results
        self._save_results(fleet_summary, output_path)
        
        logger.info(f"Fleet analysis complete. Results saved to {output_path}")
        
        return fleet_summary
    
    def _get_fleet_vehicles(self, fleet_id: int) -> List[Dict[str, Any]]:
        """Get all vehicles for a fleet."""
        try:
            vehicles = self.api_client.get_vehicles(fleet_id)
            return vehicles if isinstance(vehicles, list) else []
        except Exception as e:
            logger.error(f"Error getting vehicles: {str(e)}")
            return []
    
    def _analyze_vehicles_parallel(self, vehicles: List[Dict], fleet_id: int,
                                 start_datetime: datetime, end_datetime: datetime,
                                 output_path: Path, timezone: str):
        """Analyze vehicles in parallel."""
        with ThreadPoolExecutor(max_workers=self.analysis_config.max_workers) as executor:
            # Submit analysis tasks
            future_to_vehicle = {
                executor.submit(
                    self._analyze_single_vehicle,
                    vehicle, fleet_id, start_datetime, end_datetime, output_path, timezone
                ): vehicle
                for vehicle in vehicles
            }
            
            # Process results with progress bar
            with tqdm(total=len(vehicles), desc="Analyzing vehicles") as pbar:
                for future in as_completed(future_to_vehicle):
                    vehicle = future_to_vehicle[future]
                    vehicle_name = vehicle.get('displayName', 'Unknown')
                    
                    try:
                        result = future.result()
                        if result:
                            vehicle_id, analysis = result
                            self.vehicle_analyses[vehicle_id] = analysis
                            
                            # Check for alerts
                            alerts = self.alert_manager.check_for_alerts(analysis)
                            self.alerts.extend(alerts)
                            
                    except Exception as e:
                        logger.error(f"Error analyzing vehicle {vehicle_name}: {str(e)}")
                    
                    pbar.update(1)
    
    def _analyze_vehicles_sequential(self, vehicles: List[Dict], fleet_id: int,
                                   start_datetime: datetime, end_datetime: datetime,
                                   output_path: Path, timezone: str):
        """Analyze vehicles sequentially."""
        for vehicle in tqdm(vehicles, desc="Analyzing vehicles"):
            vehicle_name = vehicle.get('displayName', 'Unknown')
            
            try:
                result = self._analyze_single_vehicle(
                    vehicle, fleet_id, start_datetime, end_datetime, output_path, timezone
                )
                if result:
                    vehicle_id, analysis = result
                    self.vehicle_analyses[vehicle_id] = analysis
                    
                    # Check for alerts
                    alerts = self.alert_manager.check_for_alerts(analysis)
                    self.alerts.extend(alerts)
                    
            except Exception as e:
                logger.error(f"Error analyzing vehicle {vehicle_name}: {str(e)}")
    
    def _analyze_single_vehicle(self, vehicle: Dict[str, Any], fleet_id: int,
                          start_datetime: datetime, end_datetime: datetime,
                          output_path: Path, timezone: str) -> Optional[tuple]:
        """Analyze a single vehicle."""
        vehicle_id = vehicle['id']
        vehicle_name = vehicle.get('displayName', f'Vehicle_{vehicle_id}')
        
        try:
            # Get vehicle history (API expects UTC times)
            history_data = self._get_vehicle_history(
                fleet_id, vehicle_id, start_datetime, end_datetime
            )
            
            if not history_data:
                return None
            
            # Extract trips (they'll be in UTC)
            trips = self.trip_extractor.extract_trips(history_data)
            
            # Analyze shifts with timezone awareness AND analysis end time
            shift_analyses = self.shift_analyzer.analyze_shifts(
                vehicle_id, vehicle_name, trips, timezone, 
                analysis_end_time=end_datetime  # Pass the analysis end time
            )
            
            # Create vehicle analysis
            analysis = {
                'vehicle_id': vehicle_id,
                'vehicle_name': vehicle_name,
                'total_trips': len(trips),
                'shift_analyses': [sa.to_dict() for sa in shift_analyses],
                'trips': [trip.to_dict() for trip in trips],
                'timezone': timezone
            }
            
            # Save individual analysis if configured
            if self.analysis_config.save_individual_analyses:
                self._save_vehicle_analysis(
                    vehicle_id, vehicle_name, analysis, output_path
                )
            
            return vehicle_id, analysis
            
        except Exception as e:
            logger.error(f"Error analyzing vehicle {vehicle_id}: {str(e)}")
            return None
    
    def _get_vehicle_history(self, fleet_id: int, vehicle_id: int,
                       start_datetime: datetime, end_datetime: datetime) -> Optional[Dict[str, Any]]:
        """Get vehicle history from API using UTC times."""
        try:
            # Call API with datetime objects (already in UTC)
            history = self.api_client.get_vehicle_history(
                fleet_id=fleet_id,
                vehicle_id=vehicle_id,
                start_date=start_datetime,
                end_date=end_datetime
            )
            
            return {
                'vehicleId': vehicle_id,
                'fleetId': fleet_id,
                'history': history,
                'startServerTime': start_datetime.isoformat(),
                'endServerTime': end_datetime.isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting history for vehicle {vehicle_id}: {str(e)}")
            return None
    
    def _generate_fleet_summary(self) -> Dict[str, Any]:
        """Generate fleet-wide summary."""
        total_vehicles = len(self.vehicle_analyses)
        total_shifts = sum(
            len(v.get('shift_analyses', [])) 
            for v in self.vehicle_analyses.values()
        )
        
        # Calculate metrics
        shifts_meeting_target = 0
        shifts_at_risk = 0
        
        for vehicle_analysis in self.vehicle_analyses.values():
            for shift in vehicle_analysis.get('shift_analyses', []):
                if shift.get('can_complete_target'):
                    shifts_meeting_target += 1
                if shift.get('risk_level') in ['high', 'critical']:
                    shifts_at_risk += 1
        
        return {
            'fleet_summary': {
                'total_vehicles_analyzed': total_vehicles,
                'total_shifts_analyzed': total_shifts,
                'shifts_meeting_target': shifts_meeting_target,
                'shifts_at_risk': shifts_at_risk,
                'total_alerts': len(self.alerts),
                'target_completion_rate': shifts_meeting_target / total_shifts if total_shifts > 0 else 0
            },
            'vehicle_analyses': self.vehicle_analyses,
            'alerts': [alert.to_dict() for alert in self.alerts]
        }
    
    def _create_visualizations(self, fleet_summary: Dict[str, Any], output_path: Path):
        """Create all visualizations."""
        # Create alert map
        if self.alerts:
            self.map_generator.create_alert_map(
                self.alerts, 
                self.vehicle_analyses,
                output_path / 'alert_map.html'
            )
        
        # Create dashboard with timezone info
        self.dashboard_generator.create_fleet_dashboard(
            fleet_summary,
            output_path / 'dashboard.html',
            timezone=fleet_summary.get('timezone', 'UTC')
        )
    
    def _save_results(self, fleet_summary: Dict[str, Any], output_path: Path):
        """Save analysis results."""
        # Save complete analysis
        with open(output_path / 'fleet_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(fleet_summary, f, indent=2, default=str, ensure_ascii=False)  # default=str handles datetime
        
        # Save alerts separately
        if self.alerts:
            with open(output_path / 'alerts.json', 'w', encoding='utf-8') as f:
                json.dump([alert.to_dict() for alert in self.alerts], f, indent=2, default=str, ensure_ascii=False)
    
    def _save_vehicle_analysis(self, vehicle_id: int, vehicle_name: str,
                             analysis: Dict[str, Any], output_path: Path):
        """Save individual vehicle analysis."""
        vehicle_file = output_path / f'vehicle_{vehicle_id}_analysis.json'
        with open(vehicle_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, default=str, ensure_ascii=False)