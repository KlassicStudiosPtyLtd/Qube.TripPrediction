"""
Alert management for shift analysis.
"""
import logging
from datetime import datetime
from typing import List, Dict, Any

from models import Alert

logger = logging.getLogger(__name__)


class AlertManager:
    """Manages alert generation and tracking."""
    
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
        """Create an alert from shift analysis."""
        shift = shift_analysis.get('shift', {})
        
        return Alert(
            alert_id=f"ALERT_{vehicle_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            vehicle_id=vehicle_id,
            vehicle_name=vehicle_name,
            driver_id=shift.get('driver_id'),
            shift_date=shift.get('start_time', '')[:10],  # Extract date
            severity=shift_analysis.get('risk_level', 'medium'),
            message=shift_analysis.get('recommendation', ''),
            details={
                'trips_completed': shift.get('trips_completed', 0),
                'trips_target': shift_analysis.get('metrics', {}).get('trips_target', 0),
                'can_complete_target': shift_analysis.get('can_complete_target', False)
            },
            actions_required=['NOTIFY_DISPATCHER', 'MONITOR_DRIVER']
        )