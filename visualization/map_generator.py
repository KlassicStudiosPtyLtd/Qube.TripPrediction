"""
Map generation for fleet shift analysis with Unicode support.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any

import folium
import numpy as np

logger = logging.getLogger(__name__)


class MapGenerator:
    """Generates map visualizations with proper Unicode handling."""
    
    def create_alert_map(self, alerts: List[Any], vehicle_analyses: Dict[str, Any], 
                        output_file: Path):
        """Create a map showing vehicles with alerts."""
        if not alerts:
            logger.info("No alerts to map")
            return
        
        # Get last known positions
        alert_locations = []
        for alert in alerts:
            vehicle_id = alert.vehicle_id
            if vehicle_id in vehicle_analyses:
                analysis = vehicle_analyses[vehicle_id]
                trips = analysis.get('trips', [])
                
                if trips and trips[-1]['route']:
                    last_point = trips[-1]['route'][-1]
                    alert_locations.append({
                        'location': (last_point['lat'], last_point['lon']),
                        'alert': alert
                    })
        
        if not alert_locations:
            logger.warning("No location data for alerts")
            return
        
        # Create map
        center_lat = np.mean([loc['location'][0] for loc in alert_locations])
        center_lon = np.mean([loc['location'][1] for loc in alert_locations])
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Add markers
        for loc_data in alert_locations:
            location = loc_data['location']
            alert = loc_data['alert']
            
            color = 'red' if alert.severity == 'high' else 'orange'
            
            folium.Marker(
                location=location,
                popup=f"Vehicle: {alert.vehicle_name}<br>"
                     f"Alert: {alert.message}",
                icon=folium.Icon(color=color, icon='warning')
            ).add_to(m)
        
        # Save with UTF-8 encoding
        m.save(str(output_file))
        logger.info(f"Alert map saved to {output_file}")