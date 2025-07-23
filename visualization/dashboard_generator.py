"""
Dashboard generation for fleet shift analysis with timezone support.
"""
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import pytz

logger = logging.getLogger(__name__)


class DashboardGenerator:
    """Generates HTML dashboards with timezone awareness."""
    
    def create_fleet_dashboard(self, fleet_summary: Dict[str, Any], output_file: Path, 
                             timezone: str = 'UTC'):
        """Create fleet performance dashboard with timezone information."""
        summary = fleet_summary.get('fleet_summary', {})
        alerts = fleet_summary.get('alerts', [])
        analysis_period = fleet_summary.get('analysis_period', {})
        
        # Convert current time to specified timezone
        now_utc = datetime.now(pytz.UTC)
        tz = pytz.timezone(timezone)
        now_local = now_utc.astimezone(tz)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fleet Shift Performance Dashboard</title>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .header-info {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    border-left: 4px solid #007bff;
                }}
                .summary-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .summary-card {{ 
                    background: #f8f9fa; 
                    padding: 20px; 
                    border-radius: 8px;
                    text-align: center;
                    border: 1px solid #e9ecef;
                }}
                .summary-card h3 {{
                    margin-top: 0;
                    color: #495057;
                    font-size: 14px;
                    font-weight: normal;
                }}
                .summary-card .value {{
                    font-size: 32px;
                    font-weight: bold;
                    color: #212529;
                    margin: 10px 0;
                }}
                .alert-high {{ background-color: #f8d7da; }}
                .alert-medium {{ background-color: #fff3cd; }}
                .alert-low {{ background-color: #d1ecf1; }}
                table {{ 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin-top: 20px; 
                }}
                th, td {{ 
                    border: 1px solid #dee2e6; 
                    padding: 12px; 
                    text-align: left; 
                }}
                th {{ 
                    background-color: #e9ecef; 
                    font-weight: bold;
                    color: #495057;
                }}
                tr:nth-child(even) {{
                    background-color: #f8f9fa;
                }}
                .completion-rate {{
                    color: {'#28a745' if summary.get('target_completion_rate', 0) >= 0.8 else '#dc3545'};
                }}
                .timezone-info {{
                    font-size: 14px;
                    color: #6c757d;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Fleet Shift Performance Analysis</h1>
                
                <div class="header-info">
                    <p><strong>Report Generated:</strong> {now_local.strftime('%Y-%m-%d %H:%M:%S')} {timezone}</p>
                    <p><strong>Analysis Period:</strong> {self._format_analysis_period(analysis_period, timezone)}</p>
                    <p class="timezone-info"><em>All times displayed in {timezone} timezone</em></p>
                </div>
                
                <div class="summary-grid">
                    <div class="summary-card">
                        <h3>Vehicles Analyzed</h3>
                        <div class="value">{summary.get('total_vehicles_analyzed', 0)}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Total Shifts</h3>
                        <div class="value">{summary.get('total_shifts_analyzed', 0)}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Target Completion Rate</h3>
                        <div class="value completion-rate">
                            {summary.get('target_completion_rate', 0):.1%}
                        </div>
                    </div>
                    <div class="summary-card">
                        <h3>Active Alerts</h3>
                        <div class="value" style="color: #dc3545;">
                            {summary.get('total_alerts', 0)}
                        </div>
                    </div>
                    <div class="summary-card">
                        <h3>Overtime Hours</h3>
                        <div class="value" style="color: #f57c00;">
                            {summary.get('total_overtime_hours', 0):.1f}h
                        </div>
                    </div>
                    <div class="summary-card">
                        <h3>Overtime Cost</h3>
                        <div class="value" style="color: #d32f2f;">
                            ${summary.get('total_overtime_cost', 0):.0f}
                        </div>
                    </div>
                </div>
                
                {self._generate_alerts_table(alerts, timezone)}
                
                {self._generate_shift_details_table(fleet_summary, timezone)}
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard saved to {output_file}")
    
    def _format_analysis_period(self, period: Dict[str, Any], timezone: str) -> str:
        """Format the analysis period for display."""
        if not period:
            return "Unknown"
        
        try:
            # Parse UTC times
            start_utc = datetime.fromisoformat(period.get('start_utc', '').replace('Z', '+00:00'))
            end_utc = datetime.fromisoformat(period.get('end_utc', '').replace('Z', '+00:00'))
            
            # Convert to local timezone
            tz = pytz.timezone(timezone)
            start_local = start_utc.astimezone(tz)
            end_local = end_utc.astimezone(tz)
            
            # Format based on whether it's same day or multiple days
            if start_local.date() == end_local.date():
                return f"{start_local.strftime('%Y-%m-%d')} ({start_local.strftime('%H:%M')} - {end_local.strftime('%H:%M')})"
            else:
                return f"{start_local.strftime('%Y-%m-%d %H:%M')} to {end_local.strftime('%Y-%m-%d %H:%M')}"
        except:
            return "Invalid period"
    
    def _generate_alerts_table(self, alerts: List[Dict[str, Any]], timezone: str) -> str:
        """Generate alerts table HTML with timezone-aware times."""
        if not alerts:
            return "<p style='color: #28a745; font-weight: bold;'>✓ No active alerts - all shifts operating normally.</p>"
        
        rows = ""
        for alert in alerts[:20]:  # Show top 20 alerts
            severity_class = f"alert-{alert.get('severity', 'low')}"
            
            # Convert timestamp to local timezone if available
            timestamp_str = alert.get('timestamp', '')
            if timestamp_str:
                try:
                    timestamp_utc = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    tz = pytz.timezone(timezone)
                    timestamp_local = timestamp_utc.astimezone(tz)
                    time_display = timestamp_local.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    time_display = timestamp_str
            else:
                time_display = 'Unknown'
            
            rows += f"""
            <tr class="{severity_class}">
                <td>{time_display}</td>
                <td>{alert.get('vehicle_name', 'Unknown')}</td>
                <td>{alert.get('driver_id', 'Unknown')}</td>
                <td>{alert.get('shift_date', 'Unknown')}</td>
                <td>{alert.get('severity', 'Unknown').upper()}</td>
                <td>{alert.get('message', '')}</td>
            </tr>
            """
        
        return f"""
        <h2>Active Alerts</h2>
        <table>
            <tr>
                <th>Alert Time</th>
                <th>Vehicle</th>
                <th>Driver</th>
                <th>Shift Date</th>
                <th>Severity</th>
                <th>Message</th>
            </tr>
            {rows}
        </table>
        """
    
    def _generate_shift_details_table(self, fleet_summary: Dict[str, Any], timezone: str) -> str:
        """Generate shift details table with timezone-aware times."""
        vehicle_analyses = fleet_summary.get('vehicle_analyses', {})
        
        if not vehicle_analyses:
            return ""
        
        rows = ""
        tz = pytz.timezone(timezone)
        
        # Collect all shifts
        all_shifts = []
        
        for vehicle_id, analysis in vehicle_analyses.items():
            vehicle_name = analysis.get('vehicle_name', f'Vehicle_{vehicle_id}')
            vehicle_ref = analysis.get('vehicle_ref', 'Unknown')
            
            for shift_analysis in analysis.get('shift_analyses', []):
                shift = shift_analysis.get('shift', {})
                metrics = shift_analysis.get('metrics', {})
                
                # Convert times to local timezone
                try:
                    start_utc = datetime.fromisoformat(shift.get('start_time', '').replace('Z', '+00:00'))
                    end_utc = datetime.fromisoformat(shift.get('end_time', '').replace('Z', '+00:00'))
                    
                    start_local = start_utc.astimezone(tz)
                    end_local = end_utc.astimezone(tz)
                    
                    shift_date = start_local.strftime('%Y-%m-%d')
                    shift_times = f"{start_local.strftime('%H:%M')} - {end_local.strftime('%H:%M')}"
                except:
                    shift_date = "Invalid"
                    shift_times = "Invalid"
                
                all_shifts.append({
                    'date': shift_date,
                    'vehicle_name': vehicle_name,
                    'vehicle_ref': vehicle_ref,
                    'driver_id': shift.get('driver_id', 'Unknown'),
                    'driver_name': metrics.get('driver_name', 'Unknown'),
                    'shift_times': shift_times,
                    'trips_completed': metrics.get('trips_completed', 0),
                    'trips_target': metrics.get('trips_target', 0),
                    'can_complete': shift_analysis.get('can_complete_target', False),
                    'risk_level': shift_analysis.get('risk_level', 'unknown'),
                    'recommendation': shift_analysis.get('recommendation', ''),
                    'overtime_hours': metrics.get('overtime_hours', 0),
                    'overtime_cost': metrics.get('overtime_cost', 0),
                    'projected_overtime_hours': metrics.get('projected_overtime_hours', 0)
                })
        
        # Sort by date
        all_shifts.sort(key=lambda x: x['date'])
        
        # Generate rows
        for shift in all_shifts[:50]:  # Show top 50 shifts
            risk_class = ''
            if shift['risk_level'] in ['high', 'critical']:
                risk_class = 'alert-high'
            elif shift['risk_level'] == 'medium':
                risk_class = 'alert-medium'
            
            completion_symbol = '✓' if shift['can_complete'] else '✗'
            completion_color = '#28a745' if shift['can_complete'] else '#dc3545'
            
            # Format overtime info
            overtime_display = ""
            if shift['overtime_hours'] > 0:
                overtime_display = f"{shift['overtime_hours']:.1f}h (${shift['overtime_cost']:.0f})"
            elif shift['projected_overtime_hours'] > 0:
                overtime_display = f"Proj: {shift['projected_overtime_hours']:.1f}h"
            else:
                overtime_display = "None"
            
            # Driver display with name if available
            driver_display = shift['driver_name'] if shift['driver_name'] != 'Unknown' else shift['driver_id']
            
            rows += f"""
            <tr class="{risk_class}">
                <td>{shift['date']}</td>
                <td>{shift['vehicle_name']} ({shift['vehicle_ref']})</td>
                <td>{driver_display}</td>
                <td>{shift['shift_times']}</td>
                <td>{shift['trips_completed']}/{shift['trips_target']}</td>
                <td style="color: {completion_color}; font-weight: bold;">{completion_symbol}</td>
                <td>{shift['risk_level'].upper()}</td>
                <td>{overtime_display}</td>
                <td style="font-size: 12px; max-width: 200px;">{shift['recommendation'][:100]}...</td>
            </tr>
            """
        
        if not rows:
            return ""
        
        return f"""
        <h2>Shift Details</h2>
        <table>
            <tr>
                <th>Date</th>
                <th>Vehicle (Ref)</th>
                <th>Driver</th>
                <th>Shift Times</th>
                <th>Trips</th>
                <th>Target Met</th>
                <th>Risk Level</th>
                <th>Overtime</th>
                <th>Recommendation</th>
            </tr>
            {rows}
        </table>
        """