"""
Trip extraction from vehicle history data.
"""
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from geopy.distance import geodesic

from config import ShiftConfig
from models import Trip
from utils.data_processor import DataProcessor

logger = logging.getLogger(__name__)


class TripExtractor:
    """Extracts trips from vehicle history data."""
    
    def __init__(self, config: ShiftConfig):
        self.config = config
        self.data_processor = DataProcessor()
        
    def extract_trips(self, vehicle_data: Dict[str, Any]) -> List[Trip]:
        """
        Extract trips from vehicle history based on waypoint events.
        
        Args:
            vehicle_data: Vehicle history data
            
        Returns:
            List of Trip objects
        """
        history = vehicle_data.get('history', [])
        if not history:
            return []
        
        # Convert to DataFrame
        df = pd.DataFrame(history)
        if 'deviceTimeUtc' not in df.columns:
            return []
        
        df['timestamp'] = pd.to_datetime(df['deviceTimeUtc'])
        df = df.sort_values('timestamp')
        
        # Extract trips based on configuration
        if self.config.start_waypoint and self.config.end_waypoint:
            # Use specific waypoint-based extraction
            trips = self._extract_waypoint_specific_trips(
                df, 
                vehicle_data.get('vehicleId'),
                self.config.start_waypoint,
                self.config.end_waypoint
            )
        else:
            # Use general waypoint-based extraction
            trips = self._extract_general_waypoint_trips(
                df, 
                vehicle_data.get('vehicleId')
            )
        
        logger.info(f"Extracted {len(trips)} trips for vehicle {vehicle_data.get('vehicleId')}")
        
        return trips
    
    def _extract_waypoint_specific_trips(self, df: pd.DataFrame, vehicle_id: int,
                                       start_waypoint: str, end_waypoint: str) -> List[Trip]:
        """
        Extract trips between specific start and end waypoints.
        
        Args:
            df: Vehicle history DataFrame
            vehicle_id: Vehicle ID
            start_waypoint: Name of start waypoint
            end_waypoint: Name of end waypoint
            
        Returns:
            List of Trip objects
        """
        trips = []
        
        # Find all departures from start waypoint
        start_departures = self._find_waypoint_events(
            df, start_waypoint, 'DepWayPoint'
        )
        
        # Find all arrivals at end waypoint
        end_arrivals = self._find_waypoint_events(
            df, end_waypoint, 'ArrWayPoint'
        )
        
        logger.debug(f"Found {len(start_departures)} departures from {start_waypoint}")
        logger.debug(f"Found {len(end_arrivals)} arrivals at {end_waypoint}")
        
        # Match departures with arrivals
        for dep_idx, dep_row in start_departures.iterrows():
            dep_time = dep_row['timestamp']
            
            # Find the next arrival after this departure
            subsequent_arrivals = end_arrivals[end_arrivals['timestamp'] > dep_time]
            
            if not subsequent_arrivals.empty:
                # Take the first arrival
                arr_row = subsequent_arrivals.iloc[0]
                arr_idx = arr_row.name
                
                # Create trip from these waypoints
                trip_data = {
                    'start_idx': dep_idx,
                    'end_idx': arr_idx,
                    'start_time': dep_time,
                    'end_time': arr_row['timestamp'],
                    'start_place': start_waypoint,
                    'end_place': end_waypoint,
                    'start_location': (dep_row['latitude'], dep_row['longitude']),
                    'end_location': (arr_row['latitude'], arr_row['longitude']),
                    'driver_id': dep_row.get('driverId')
                }
                
                # Extract the complete route
                trip = self._create_trip_from_waypoints(
                    df, trip_data, vehicle_id
                )
                
                if trip:
                    trips.append(trip)
        
        return trips
    
    def _extract_general_waypoint_trips(self, df: pd.DataFrame, 
                                      vehicle_id: int) -> List[Trip]:
        """Extract trips based on all DepWayPoint and ArrWayPoint events."""
        trips = []
        current_trip_data = None
        
        for idx, row in df.iterrows():
            reason_code = row.get('reasonCode')
            
            if reason_code == 'DepWayPoint':
                current_trip_data = self._handle_departure(
                    current_trip_data, row, idx, df.loc[idx, 'timestamp']
                )
                
            elif reason_code == 'ArrWayPoint' and current_trip_data:
                trip = self._handle_arrival(
                    current_trip_data, row, idx, df, vehicle_id
                )
                if trip:
                    trips.append(trip)
                    current_trip_data = None
        
        # Handle last trip if not completed
        if current_trip_data and current_trip_data.get('status') == 'arriving':
            trip = self._finalize_trip(current_trip_data, df, vehicle_id)
            if trip:
                trips.append(trip)
        
        return trips
    
    def _find_waypoint_events(self, df: pd.DataFrame, waypoint_name: str,
                            event_type: str) -> pd.DataFrame:
        """
        Find waypoint events matching the given criteria.
        
        Args:
            df: Vehicle history DataFrame
            waypoint_name: Waypoint name to match
            event_type: 'DepWayPoint' or 'ArrWayPoint'
            
        Returns:
            DataFrame of matching events
        """
        # Filter by event type
        events = df[df['reasonCode'] == event_type].copy()
        
        if events.empty:
            return events
        
        # Apply waypoint matching based on configuration
        if self.config.waypoint_matching == 'exact':
            # Exact match
            matching_events = events[events['placeName'] == waypoint_name]
        
        elif self.config.waypoint_matching == 'contains':
            # Contains match (case-insensitive)
            matching_events = events[
                events['placeName'].str.contains(
                    waypoint_name, case=False, na=False
                )
            ]
        
        elif self.config.waypoint_matching == 'normalized':
            # Normalized match
            normalized_waypoint = self.data_processor.normalize_place_names(waypoint_name)
            events['normalized_place'] = events['placeName'].apply(
                self.data_processor.normalize_place_names
            )
            matching_events = events[events['normalized_place'] == normalized_waypoint]
        
        else:
            matching_events = events[events['placeName'] == waypoint_name]
        
        return matching_events
    
    def _create_trip_from_waypoints(self, df: pd.DataFrame, trip_data: Dict[str, Any],
                                  vehicle_id: int) -> Optional[Trip]:
        """Create a Trip object from waypoint data."""
        start_idx = trip_data['start_idx']
        end_idx = trip_data['end_idx']
        
        # Extract route between waypoints
        route_df = df.loc[start_idx:end_idx]
        
        # Calculate metrics
        duration = (trip_data['end_time'] - trip_data['start_time']).total_seconds() / 60
        
        # Calculate distance
        total_distance = self._calculate_route_distance(route_df)
        
        # Check minimum requirements
        if total_distance < self.config.min_trip_distance_m:
            logger.debug(f"Trip distance {total_distance}m below minimum {self.config.min_trip_distance_m}m")
            return None
        
        if duration < self.config.min_trip_duration_minutes:
            logger.debug(f"Trip duration {duration}min below minimum {self.config.min_trip_duration_minutes}min")
            return None
        
        # Extract route points
        route_points = self._extract_route_points(route_df)
        
        # Determine if round trip
        is_round_trip = trip_data['start_place'] == trip_data['end_place']
        
        # Create Trip object
        return Trip(
            trip_id=f"{vehicle_id}_{trip_data['start_time'].strftime('%Y%m%d_%H%M%S')}",
            vehicle_id=vehicle_id,
            driver_id=trip_data.get('driver_id'),
            start_place=trip_data['start_place'],
            end_place=trip_data['end_place'],
            start_time=trip_data['start_time'],
            end_time=trip_data['end_time'],
            duration_minutes=round(duration, 1),
            distance_m=round(total_distance, 1),
            route=route_points,
            is_round_trip=is_round_trip
        )
    
    def _handle_departure(self, current_trip_data: Optional[Dict], 
                         row: pd.Series, idx: int, timestamp: pd.Timestamp) -> Dict:
        """Handle departure waypoint event."""
        if current_trip_data and current_trip_data['status'] == 'departing':
            # Check if this is a continuation
            time_diff = (timestamp - current_trip_data['last_dep_time']).total_seconds() / 60
            
            if time_diff < self.config.min_trip_duration_minutes:
                # Update departure info
                current_trip_data['last_dep_time'] = timestamp
                current_trip_data['dep_indices'].append(idx)
                return current_trip_data
        
        # Start new trip
        return {
            'status': 'departing',
            'start_time': timestamp,
            'last_dep_time': timestamp,
            'start_place': row.get('placeName', 'Unknown'),
            'start_location': (row['latitude'], row['longitude']),
            'dep_indices': [idx],
            'arr_indices': [],
            'start_idx': idx,
            'driver_id': row.get('driverId')
        }
    
    def _handle_arrival(self, trip_data: Dict, row: pd.Series, idx: int,
                       df: pd.DataFrame, vehicle_id: int) -> Optional[Trip]:
        """Handle arrival waypoint event."""
        timestamp = df.loc[idx, 'timestamp']
        
        if trip_data['status'] == 'departing':
            # First arrival
            trip_data['status'] = 'arriving'
            trip_data['first_arr_time'] = timestamp
            trip_data['last_arr_time'] = timestamp
            trip_data['end_place'] = row.get('placeName', 'Unknown')
            trip_data['end_location'] = (row['latitude'], row['longitude'])
            trip_data['arr_indices'].append(idx)
            trip_data['end_idx'] = idx
            
        elif trip_data['status'] == 'arriving':
            # Check if extending arrival
            time_diff = (timestamp - trip_data['last_arr_time']).total_seconds() / 60
            
            if time_diff < self.config.min_trip_duration_minutes:
                # Extend arrival
                trip_data['last_arr_time'] = timestamp
                trip_data['arr_indices'].append(idx)
                trip_data['end_idx'] = idx
            else:
                # Finalize current trip
                return self._finalize_trip(trip_data, df, vehicle_id)
        
        return None
    
    def _finalize_trip(self, trip_data: Dict, df: pd.DataFrame, 
                      vehicle_id: int) -> Optional[Trip]:
        """Finalize a trip and create Trip object."""
        # Calculate metrics
        duration = (trip_data['last_arr_time'] - trip_data['start_time']).total_seconds() / 60
        
        # Extract route
        route_df = df.iloc[trip_data['start_idx']:trip_data['end_idx']+1]
        
        # Calculate distance
        total_distance = self._calculate_route_distance(route_df)
        
        # Check minimum requirements
        if total_distance < self.config.min_trip_distance_m:
            return None
        
        # Extract route points
        route_points = self._extract_route_points(route_df)
        
        # Create Trip object
        return Trip(
            trip_id=f"{vehicle_id}_{trip_data['start_time'].strftime('%Y%m%d_%H%M%S')}",
            vehicle_id=vehicle_id,
            driver_id=trip_data.get('driver_id'),
            start_place=trip_data['start_place'],
            end_place=trip_data['end_place'],
            start_time=trip_data['start_time'],
            end_time=trip_data['last_arr_time'],
            duration_minutes=round(duration, 1),
            distance_m=round(total_distance, 1),
            route=route_points,
            is_round_trip=trip_data['start_place'] == trip_data['end_place']
        )
    
    def _calculate_route_distance(self, route_df: pd.DataFrame) -> float:
        """Calculate total distance of a route."""
        coords = list(zip(route_df['latitude'], route_df['longitude']))
        return self.data_processor.calculate_route_distance(coords)
    
    def _extract_route_points(self, route_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract route points from DataFrame."""
        route_points = []
        
        for _, point in route_df.iterrows():
            route_points.append({
                'lat': float(point['latitude']),
                'lon': float(point['longitude']),
                'timestamp': point['timestamp'].isoformat(),
                'speed_kmh': float(point.get('speedKmh', 0)),
                'place_name': point.get('placeName'),
                'reason_code': point.get('reasonCode')
            })
        
        return route_points