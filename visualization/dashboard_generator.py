"""
Dashboard generation for fleet shift analysis.
"""
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class DashboardGenerator:
    """Generates HTML dashboards."""
    
    def create_fleet_dashboard(self, fleet_summary: Dict[str, Any], output_file: Path):
        """Create fleet performance dashboard."""
        summary = fleet_summary.get('fleet_summary', {})
        alerts = fleet_summary.get('alerts', [])
        
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
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Fleet Shift Performance Analysis</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
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
                </div>
                
                {self._generate_alerts_table(alerts)}
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard saved to {output_file}")
    
    def _generate_alerts_table(self, alerts: List[Dict[str, Any]]) -> str:
        """Generate alerts table HTML."""
        if not alerts:
            return "<p>No active alerts.</p>"
        
        rows = ""
        for alert in alerts[:20]:  # Show top 20 alerts
            severity_class = f"alert-{alert.get('severity', 'low')}"
            rows += f"""
            <tr class="{severity_class}">
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
                <th>Vehicle</th>
                <th>Driver</th>
                <th>Shift Date</th>
                <th>Severity</th>
                <th>Message</th>
            </tr>
            {rows}
        </table>
        """