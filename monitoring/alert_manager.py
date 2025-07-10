"""
Alert management for shift analysis with detailed algorithmic messages.
"""
import logging
from datetime import datetime
from typing import List, Dict, Any
import pytz

from models import Alert

logger = logging.getLogger(__name__)


class AlertManager:
    """Manages alert generation and tracking with detailed algorithmic explanations."""
    
    def __init__(self):
        self.alert_history = []
        
    def check_for_alerts(self, vehicle_analysis: Dict[str, Any]) -> List[Alert]:
        """Check vehicle analysis for alert conditions."""
        alerts = []
        
        vehicle_id = vehicle_analysis.get('vehicle_id')
        vehicle_name = vehicle_analysis.get('vehicle_name', f'Vehicle_{vehicle_id}')
        
        for shift_analysis in vehicle_analysis.get('shift_analyses', []):
            if shift_analysis.get('alert_required'):
                alert = self._create_alert(
                    vehicle_id=vehicle_id,
                    vehicle_name=vehicle_name,
                    shift_analysis=shift_analysis
                )
                alerts.append(alert)
                self.alert_history.append(alert)
        
        return alerts
    
    def _create_alert(self, vehicle_id: int, vehicle_name: str, 
                     shift_analysis: Dict[str, Any]) -> Alert:
        """Create an alert from shift analysis with detailed message."""
        shift = shift_analysis.get('shift', {})
        
        # Create timezone-aware timestamp
        now = datetime.now(pytz.UTC)
        
        # Extract the detailed recommendation which now includes algorithm details
        detailed_message = shift_analysis.get('recommendation', '')
        
        # Create a summary message for quick viewing
        risk_level = shift_analysis.get('risk_level', 'medium')
        trips_completed = shift.get('trips_completed', 0)
        trips_remaining = shift_analysis.get('metrics', {}).get('trips_remaining', 0)
        
        if risk_level in ['high', 'critical']:
            if trips_remaining == 1:
                summary = "DO NOT START final trip. High risk of exceeding shift duration."
            else:
                safe_trips = max(0, trips_remaining - 1)
                summary = f"HIGH RISK: Complete only {safe_trips} more trips."
        else:
            summary = f"Monitor closely. {trips_remaining} trips remaining."
        
        return Alert(
            alert_id=f"ALERT_{vehicle_id}_{now.strftime('%Y%m%d_%H%M%S')}",
            timestamp=now,
            vehicle_id=vehicle_id,
            vehicle_name=vehicle_name,
            driver_id=shift.get('driver_id'),
            shift_date=shift.get('start_time', '')[:10],  # Extract date
            severity=risk_level,
            message=summary,  # Short summary for display
            details={
                'trips_completed': trips_completed,
                'trips_target': shift_analysis.get('metrics', {}).get('trips_target', 0),
                'trips_remaining': trips_remaining,
                'can_complete_target': shift_analysis.get('can_complete_target', False),
                'remaining_time_minutes': shift_analysis.get('metrics', {}).get('remaining_time', 0),
                'estimated_time_needed_minutes': shift_analysis.get('metrics', {}).get('total_estimated_time', 0),
                'time_into_shift_hours': shift_analysis.get('metrics', {}).get('time_into_shift_hours', 0),
                'algorithm_details': detailed_message,  # Full algorithm explanation
                'shift_start': shift.get('start_time', ''),
                'shift_end': shift.get('end_time', ''),
                'average_trip_duration': shift_analysis.get('metrics', {}).get('avg_trip_duration', 0),
                'completion_probability': shift_analysis.get('metrics', {}).get('completion_probability', 0)
            },
            actions_required=['NOTIFY_DISPATCHER', 'MONITOR_DRIVER', 'REVIEW_ALGORITHM_DETAILS']
        )