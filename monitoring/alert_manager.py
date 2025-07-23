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
        metrics = shift_analysis.get('metrics', {})
        
        # Check for overtime
        overtime_hours = metrics.get('overtime_hours', 0)
        projected_overtime = metrics.get('projected_overtime_hours', 0)
        
        if overtime_hours > 0:
            summary = f"OVERTIME ALERT: Driver has worked {overtime_hours:.1f}h overtime (${overtime_hours * 1.5 * 50:.2f})"
        elif projected_overtime > 0:
            summary = f"OVERTIME WARNING: Completing all trips will result in {projected_overtime:.1f}h overtime"
        elif risk_level in ['high', 'critical']:
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
                'completion_probability': shift_analysis.get('metrics', {}).get('completion_probability', 0),
                'driver_id': metrics.get('driver_id'),
                'driver_name': metrics.get('driver_name'),
                'driver_ref': metrics.get('driver_ref'),
                'overtime_hours': overtime_hours,
                'overtime_cost': metrics.get('overtime_cost', 0),
                'projected_overtime_hours': projected_overtime,
                'projected_overtime_cost': projected_overtime * 1.5 * 50 if projected_overtime > 0 else 0
            },
            actions_required=self._determine_actions(risk_level, overtime_hours, projected_overtime)
        )
    
    def _determine_actions(self, risk_level: str, overtime_hours: float, 
                          projected_overtime: float) -> List[str]:
        """Determine required actions based on alert severity and overtime."""
        actions = ['NOTIFY_DISPATCHER', 'MONITOR_DRIVER']
        
        if overtime_hours > 0:
            actions.extend(['OVERTIME_LOGGED', 'REVIEW_COST_IMPACT'])
        
        if projected_overtime > 0:
            actions.extend(['PREVENT_OVERTIME', 'REDUCE_TRIPS'])
        
        if risk_level in ['high', 'critical']:
            actions.append('IMMEDIATE_ACTION_REQUIRED')
        
        actions.append('REVIEW_ALGORITHM_DETAILS')
        
        return actions
    
    def get_overtime_alerts(self, start_date: datetime = None, 
                           end_date: datetime = None) -> List[Alert]:
        """Get all overtime-related alerts within a date range."""
        overtime_alerts = []
        
        for alert in self.alert_history:
            # Check if alert is within date range
            if start_date and alert.timestamp < start_date:
                continue
            if end_date and alert.timestamp > end_date:
                continue
            
            # Check if alert is overtime-related
            if (alert.details.get('overtime_hours', 0) > 0 or 
                alert.details.get('projected_overtime_hours', 0) > 0):
                overtime_alerts.append(alert)
        
        return overtime_alerts
    
    def generate_overtime_summary(self, fleet_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of overtime across the fleet."""
        summary = {
            'total_overtime_hours': 0.0,
            'total_overtime_cost': 0.0,
            'drivers_in_overtime': [],
            'drivers_at_risk': [],
            'recommendations': []
        }
        
        for vehicle_analysis in fleet_analyses:
            for shift_analysis in vehicle_analysis.get('shift_analyses', []):
                metrics = shift_analysis.get('metrics', {})
                
                # Current overtime
                overtime_hours = metrics.get('overtime_hours', 0)
                if overtime_hours > 0:
                    summary['total_overtime_hours'] += overtime_hours
                    summary['total_overtime_cost'] += metrics.get('overtime_cost', 0)
                    summary['drivers_in_overtime'].append({
                        'driver_id': metrics.get('driver_id'),
                        'driver_name': metrics.get('driver_name'),
                        'vehicle_id': vehicle_analysis.get('vehicle_id'),
                        'overtime_hours': overtime_hours,
                        'cost': metrics.get('overtime_cost', 0)
                    })
                
                # Projected overtime
                projected_overtime = metrics.get('projected_overtime_hours', 0)
                if projected_overtime > 0:
                    summary['drivers_at_risk'].append({
                        'driver_id': metrics.get('driver_id'),
                        'driver_name': metrics.get('driver_name'),
                        'vehicle_id': vehicle_analysis.get('vehicle_id'),
                        'projected_overtime': projected_overtime,
                        'trips_remaining': metrics.get('trips_remaining', 0)
                    })
        
        # Generate recommendations
        if summary['drivers_in_overtime']:
            summary['recommendations'].append(
                f"Immediate action: {len(summary['drivers_in_overtime'])} drivers already in overtime"
            )
        
        if summary['drivers_at_risk']:
            summary['recommendations'].append(
                f"Prevention needed: {len(summary['drivers_at_risk'])} drivers at risk of overtime"
            )
        
        if summary['total_overtime_cost'] > 0:
            summary['recommendations'].append(
                f"Current overtime cost: ${summary['total_overtime_cost']:.2f}"
            )
        
        return summary