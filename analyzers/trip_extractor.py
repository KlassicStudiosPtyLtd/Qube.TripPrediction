"""
Trip extraction from vehicle history data with three-point round trip support.
"""
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pytz

import pandas as pd
from geopy.distance import geodesic

from config import ShiftConfig
from models import Trip, TripSegment
from utils.data_processor import DataProcessor

logger = logging.getLogger(__name__)


class TripExtractor:
    """Extracts trips from vehicle history data with three-point round trip support."""
    
    def __init__(self, config: ShiftConfig):
        self.config = config
        self.data_processor = DataProcessor()
        self._reason_code_stats = {}  # Track reason codes we encounter
        
    def _is_departure_event(self, reason_code: str) -> bool:
        """Check if a reason code indicates a departure event."""
        if not reason_code:
            return False
        code_upper = str(reason_code).strip().upper()
        return code_upper[:3] == 'DEP'
    
    def _is_arrival_event(self, reason_code: str) -> bool:
        """Check if a reason code indicates an arrival event."""
        if not reason_code:
            return False
        code_upper = str(reason_code).strip().upper()
        return code_upper[:3] == 'ARR'
    
    def _track_reason_code(self, reason_code: str):
        """Track reason codes for debugging purposes."""
        if reason_code not in self._reason_code_stats:
            self._reason_code_stats[reason_code] = 0
        self._reason_code_stats[reason_code] += 1
        
    def extract_trips(self, vehicle_data: Dict[str, Any]) -> List[Trip]:
        """
        Extract trips from vehicle history based on configuration.
        
        Args:
            vehicle_data: Vehicle history data
            
        Returns:
            List of Trip objects with timezone-aware timestamps
        """
        history = vehicle_data.get('history', [])
        if not history:
            return []
        
        # Convert to DataFrame
        df = pd.DataFrame(history)
        if 'deviceTimeUtc' not in df.columns:
            return []
        
        # Convert timestamps to timezone-aware UTC
        df['timestamp'] = pd.to_datetime(df['deviceTimeUtc'])
        # Ensure timestamps are timezone-aware (UTC)
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
        
        df = df.sort_values('timestamp')
        
        # Extract trips based on configuration
        if self.config.round_trip_mode == 'three_point' and self.config.target_waypoint:
            # Use three-point round trip extraction
            trips = self._extract_three_point_round_trips(
                df, 
                vehicle_data.get('vehicleId'),
                self.config.start_waypoint,
                self.config.target_waypoint,
                self.config.end_waypoint
            )
        elif self.config.start_waypoint and self.config.end_waypoint:
            # Use specific waypoint-based extraction (simple mode)
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
        
        logger.info(f"Extracted {len(trips)} trips for vehicle {vehicle_data.get('vehicleId')} "
                   f"(mode: {self.config.round_trip_mode})")
        
        # Log reason code statistics if in debug mode
        if logger.isEnabledFor(logging.DEBUG) and self._reason_code_stats:
            logger.debug("Reason code statistics:")
            for code, count in sorted(self._reason_code_stats.items()):
                event_type = "departure" if self._is_departure_event(code) else \
                            "arrival" if self._is_arrival_event(code) else "unknown"
                logger.debug(f"  {code}: {count} occurrences ({event_type})")
        
        return trips
    
    def _extract_three_point_round_trips(self, df: pd.DataFrame, vehicle_id: int,
                                       start_waypoint: str, target_waypoint: str,
                                       end_waypoint: str) -> List[Trip]:
        """
        Extract three-point round trips: Start -> Target -> End.
        
        Args:
            df: Vehicle history DataFrame with timezone-aware timestamps
            vehicle_id: Vehicle ID
            start_waypoint: Starting waypoint name
            target_waypoint: Target/intermediate waypoint name
            end_waypoint: Ending waypoint name
            
        Returns:
            List of Trip objects representing complete or partial round trips
        """
        trips = []
        
        # State machine for tracking trip progress
        trip_state = {
            'status': 'waiting_for_start',  # States: waiting_for_start, departing_start, going_to_target, at_target, going_to_end
            'current_trip_data': None,
            'segments': [],
            'waypoints_visited': []
        }
        
        logger.debug(f"Looking for three-point trips: {start_waypoint} -> {target_waypoint} -> {end_waypoint}")
        
        for idx, row in df.iterrows():
            reason_code = str(row.get('reasonCode', '')).strip()
            place_name = row.get('placeName', '')
            
            # Track reason codes for debugging
            if reason_code:
                self._track_reason_code(reason_code)
            
            # Determine if this is an arrival or departure
            is_departure = self._is_departure_event(reason_code)
            is_arrival = self._is_arrival_event(reason_code)
            
            # Log unexpected reason codes
            if reason_code and not is_departure and not is_arrival:
                a = 1 # Placeholder for unexpected reason codes
                #logger.debug(f"Unrecognized reason code: {reason_code} at {place_name}")
            else:
                logger.debug(f"Processing {('departure' if is_departure else 'arrival')}: {place_name} at {row['timestamp']}")
            
            # Check for start waypoint
            if self._matches_waypoint(place_name, start_waypoint):
                if is_departure and trip_state['status'] == 'waiting_for_start':
                    # Starting new round trip
                    logger.debug(f"Starting new three-point trip from {start_waypoint} at {row['timestamp']}")
                    trip_state['status'] = 'going_to_target'
                    trip_state['current_trip_data'] = {
                        'start_idx': idx,
                        'start_time': row['timestamp'],
                        'start_location': (row['latitude'], row['longitude']),
                        'driver_id': row.get('driverId') or ('UNKNOWN' if self.config.handle_missing_driver == 'assign_unknown' else None),
                        'segment_start_idx': idx,
                        'segment_start_time': row['timestamp'],
                        'segment_start_location': (row['latitude'], row['longitude'])
                    }
                    trip_state['segments'] = []
                    trip_state['waypoints_visited'] = [start_waypoint]
                    
                # Check if this is actually the end waypoint (when start == end)
                elif is_arrival and trip_state['status'] == 'going_to_end' and start_waypoint == end_waypoint:
                    # This is arrival at end waypoint (which happens to be same as start)
                    logger.debug(f"Completed three-point trip at {end_waypoint} (same as start) at {row['timestamp']}")
                    trip_state['waypoints_visited'].append(end_waypoint)
                    
                    # Save second segment (Target -> End)
                    segment = self._create_trip_segment(
                        df,
                        trip_state['current_trip_data']['segment_start_idx'],
                        idx,
                        target_waypoint,
                        end_waypoint,
                        f"{vehicle_id}_seg2_{trip_state['current_trip_data']['start_time'].strftime('%Y%m%d_%H%M%S')}"
                    )
                    if segment:
                        trip_state['segments'].append(segment)
                    
                    # Create complete trip
                    trip = self._create_three_point_trip(
                        df, vehicle_id, trip_state, idx, row
                    )
                    
                    if trip:
                        trips.append(trip)
                    
                    # Reset state
                    trip_state['status'] = 'waiting_for_start'
                    trip_state['current_trip_data'] = None
                    trip_state['segments'] = []
                    trip_state['waypoints_visited'] = []
                    
            # Check for target waypoint
            elif self._matches_waypoint(place_name, target_waypoint):
                if is_arrival and trip_state['status'] == 'going_to_target':
                    # Arrived at target
                    logger.debug(f"Arrived at target {target_waypoint} at {row['timestamp']}")
                    trip_state['status'] = 'at_target'
                    trip_state['waypoints_visited'].append(target_waypoint)
                    
                    # Save first segment (Start -> Target)
                    segment = self._create_trip_segment(
                        df,
                        trip_state['current_trip_data']['segment_start_idx'],
                        idx,
                        start_waypoint,
                        target_waypoint,
                        f"{vehicle_id}_seg1_{trip_state['current_trip_data']['start_time'].strftime('%Y%m%d_%H%M%S')}"
                    )
                    if segment:
                        trip_state['segments'].append(segment)
                    
                elif is_departure and trip_state['status'] == 'at_target':
                    # Departing from target
                    logger.debug(f"Departing from target {target_waypoint} at {row['timestamp']}")
                    trip_state['status'] = 'going_to_end'
                    trip_state['current_trip_data']['segment_start_idx'] = idx
                    trip_state['current_trip_data']['segment_start_time'] = row['timestamp']
                    trip_state['current_trip_data']['segment_start_location'] = (row['latitude'], row['longitude'])
                    
            # Check for end waypoint (only if different from start)
            elif self._matches_waypoint(place_name, end_waypoint) and end_waypoint != start_waypoint:
                if is_arrival and trip_state['status'] == 'going_to_end':
                    # Completed round trip!
                    logger.debug(f"Completed three-point trip at {end_waypoint} at {row['timestamp']}")
                    trip_state['waypoints_visited'].append(end_waypoint)
                    
                    # Save second segment (Target -> End)
                    segment = self._create_trip_segment(
                        df,
                        trip_state['current_trip_data']['segment_start_idx'],
                        idx,
                        target_waypoint,
                        end_waypoint,
                        f"{vehicle_id}_seg2_{trip_state['current_trip_data']['start_time'].strftime('%Y%m%d_%H%M%S')}"
                    )
                    if segment:
                        trip_state['segments'].append(segment)
                    
                    # Create complete trip
                    trip = self._create_three_point_trip(
                        df, vehicle_id, trip_state, idx, row
                    )
                    
                    if trip:
                        trips.append(trip)
                    
                    # Reset state
                    trip_state['status'] = 'waiting_for_start'
                    trip_state['current_trip_data'] = None
                    trip_state['segments'] = []
                    trip_state['waypoints_visited'] = []
        
        # Handle incomplete trips if configured
        if self.config.allow_partial_trips and trip_state['current_trip_data'] is not None:
            logger.debug(f"Creating partial trip with {len(trip_state['segments'])} segments")
            partial_trip = self._create_partial_three_point_trip(
                df, vehicle_id, trip_state
            )
            if partial_trip:
                trips.append(partial_trip)
        
        return trips
    
    def _create_trip_segment(self, df: pd.DataFrame, start_idx: int, end_idx: int,
                        from_waypoint: str, to_waypoint: str, segment_id: str) -> Optional[TripSegment]:
        """Create a trip segment from waypoint indices."""
        # Extract route between waypoints
        route_df = df.loc[start_idx:end_idx]
        
        # Get times
        start_time = df.loc[start_idx, 'timestamp']
        end_time = df.loc[end_idx, 'timestamp']
        duration = (end_time - start_time).total_seconds() / 60
        
        # ADD LOGGING HERE
        logger.debug(f"Creating segment {segment_id}:")
        logger.debug(f"  From: {from_waypoint} at index {start_idx}, time {start_time}")
        logger.debug(f"  To: {to_waypoint} at index {end_idx}, time {end_time}")
        logger.debug(f"  Duration: {duration:.1f} minutes")
        logger.debug(f"  Start event: {df.loc[start_idx, 'reasonCode']} at {df.loc[start_idx, 'placeName']}")
        logger.debug(f"  End event: {df.loc[end_idx, 'reasonCode']} at {df.loc[end_idx, 'placeName']}")
        
        # Calculate distance
        coords = list(zip(route_df['latitude'], route_df['longitude']))
        total_distance = self.data_processor.calculate_route_distance(coords)
        
        # Extract route points
        route_points = self._extract_route_points(route_df)
        
        return TripSegment(
            segment_id=segment_id,
            from_waypoint=from_waypoint,
            to_waypoint=to_waypoint,
            start_time=start_time,
            end_time=end_time,
            duration_minutes=duration,
            distance_m=total_distance,
            route_points=route_points
        )
    
    def _create_three_point_trip(self, df: pd.DataFrame, vehicle_id: int,
                               trip_state: Dict[str, Any], end_idx: int, 
                               end_row: pd.Series) -> Optional[Trip]:
        """Create a complete three-point trip from collected segments."""
        trip_data = trip_state['current_trip_data']
        segments = trip_state['segments']

        if not segments:
            return None

        logger.info(f"Creating three-point trip for vehicle {vehicle_id}")
        logger.info(f"Number of segments: {len(segments)}")
        for i, seg in enumerate(segments):
            logger.debug(f"  Segment {i}: {seg.from_waypoint} -> {seg.to_waypoint}")
            logger.debug(f"    Start: {seg.start_time}, End: {seg.end_time}")
            logger.debug(f"    Duration: {seg.duration_minutes:.1f} min")
        
       
        # Calculate total metrics
        total_distance = sum(seg.distance_m for seg in segments)
        start_time = trip_data['start_time']
        end_time = end_row['timestamp']
        total_duration = (end_time - start_time).total_seconds() / 60
        
        # Check minimum requirements
        if total_distance < self.config.min_trip_distance_m:
            logger.debug(f"Three-point trip distance {total_distance}m below minimum")
            return None
        
        if total_duration < self.config.min_trip_duration_minutes:
            logger.debug(f"Three-point trip duration {total_duration}min below minimum")
            return None
        
        # Extract complete route
        route_df = df.loc[trip_data['start_idx']:end_idx]
        route_points = self._extract_route_points(route_df)
        
        # Ensure timestamps are timezone-aware
        if start_time.tzinfo is None:
            start_time = pytz.UTC.localize(start_time)
        if end_time.tzinfo is None:
            end_time = pytz.UTC.localize(end_time)


        
        # Create Trip object
        return Trip(
            trip_id=f"{vehicle_id}_{start_time.strftime('%Y%m%d_%H%M%S')}_3pt",
            vehicle_id=vehicle_id,
            driver_id=trip_data.get('driver_id'),
            start_place=self.config.start_waypoint,
            end_place=self.config.end_waypoint,
            start_time=start_time,
            end_time=end_time,
            duration_minutes=round(total_duration, 1),
            distance_m=round(total_distance, 1),
            route=route_points,
            is_round_trip=self.config.start_waypoint == self.config.end_waypoint,
            trip_type='three_point_round',
            trip_segments=segments,
            waypoints_visited=trip_state['waypoints_visited'],
            is_complete_round_trip=True,
            target_waypoint=self.config.target_waypoint
        )
    
    def _create_partial_three_point_trip(self, df: pd.DataFrame, vehicle_id: int,
                                       trip_state: Dict[str, Any]) -> Optional[Trip]:
        """Create a partial three-point trip from incomplete segments."""
        trip_data = trip_state['current_trip_data']
        segments = trip_state['segments']
        
        if not trip_data:
            return None
        
        # Determine the end point based on current state
        if trip_state['status'] in ['going_to_target', 'at_target', 'going_to_end']:
            # Find the last relevant event
            last_idx = len(df) - 1
            last_row = df.iloc[-1]
            end_time = last_row['timestamp']
            end_place = trip_state['waypoints_visited'][-1] if trip_state['waypoints_visited'] else 'Unknown'
        else:
            return None
        
        # Calculate metrics
        start_time = trip_data['start_time']
        total_duration = (end_time - start_time).total_seconds() / 60
        
        # Calculate distance from segments + current progress
        total_distance = sum(seg.distance_m for seg in segments) if segments else 0
        
        # Add distance for current incomplete segment if applicable
        if trip_state['status'] in ['going_to_target', 'going_to_end']:
            current_start_idx = trip_data.get('segment_start_idx', trip_data['start_idx'])
            current_route_df = df.loc[current_start_idx:last_idx]
            coords = list(zip(current_route_df['latitude'], current_route_df['longitude']))
            total_distance += self.data_processor.calculate_route_distance(coords)
        
        # Extract route points
        route_df = df.loc[trip_data['start_idx']:last_idx]
        route_points = self._extract_route_points(route_df)
        
        # Ensure timestamps are timezone-aware
        if start_time.tzinfo is None:
            start_time = pytz.UTC.localize(start_time)
        if end_time.tzinfo is None:
            end_time = pytz.UTC.localize(end_time)
        
        return Trip(
            trip_id=f"{vehicle_id}_{start_time.strftime('%Y%m%d_%H%M%S')}_3pt_partial",
            vehicle_id=vehicle_id,
            driver_id=trip_data.get('driver_id'),
            start_place=self.config.start_waypoint,
            end_place=end_place,
            start_time=start_time,
            end_time=end_time,
            duration_minutes=round(total_duration, 1),
            distance_m=round(total_distance, 1),
            route=route_points,
            is_round_trip=False,
            trip_type='three_point_round',
            trip_segments=segments,
            waypoints_visited=trip_state['waypoints_visited'],
            is_complete_round_trip=False,
            target_waypoint=self.config.target_waypoint
        )
    
    def _matches_waypoint(self, place_name: str, waypoint_name: str) -> bool:
        """Check if a place name matches a waypoint based on matching configuration."""
        if not place_name or not waypoint_name:
            return False
        
        if self.config.waypoint_matching == 'exact':
            return place_name == waypoint_name
        elif self.config.waypoint_matching == 'contains':
            return waypoint_name.lower() in place_name.lower()
        elif self.config.waypoint_matching == 'normalized':
            return (self.data_processor.normalize_place_names(place_name) == 
                   self.data_processor.normalize_place_names(waypoint_name))
        else:
            return place_name == waypoint_name
    
    # Keep existing methods for backward compatibility
    def _extract_waypoint_specific_trips(self, df: pd.DataFrame, vehicle_id: int,
                                       start_waypoint: str, end_waypoint: str) -> List[Trip]:
        """Extract trips between specific start and end waypoints (simple mode)."""
        trips = []
        
        # Find all departures from start waypoint
        start_departures = self._find_waypoint_events(
            df, start_waypoint, 'departure'
        )
        
        # Find all arrivals at end waypoint
        end_arrivals = self._find_waypoint_events(
            df, end_waypoint, 'arrival'
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
        """Extract trips based on all departure and arrival events."""
        trips = []
        current_trip_data = None
        
        for idx, row in df.iterrows():
            reason_code = str(row.get('reasonCode', '')).strip()
            
            # Track reason codes for debugging
            if reason_code:
                self._track_reason_code(reason_code)
            
            # Determine if this is an arrival or departure
            is_departure = self._is_departure_event(reason_code)
            is_arrival = self._is_arrival_event(reason_code)
            
            if is_departure:
                current_trip_data = self._handle_departure(
                    current_trip_data, row, idx, df.loc[idx, 'timestamp']
                )
                
            elif is_arrival and current_trip_data:
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
            event_type: 'departure' or 'arrival'
            
        Returns:
            DataFrame of matching events
        """
        # Filter by event type based on first 3 letters of reason code
        if event_type == 'departure':
            events = df[df['reasonCode'].str[:3].str.upper() == 'DEP'].copy()
        elif event_type == 'arrival':
            events = df[df['reasonCode'].str[:3].str.upper() == 'ARR'].copy()
        else:
            logger.warning(f"Unknown event type: {event_type}")
            return pd.DataFrame()
        
        if events.empty:
            return events
        
        # Apply waypoint matching
        matching_indices = []
        for idx, row in events.iterrows():
            if self._matches_waypoint(row.get('placeName', ''), waypoint_name):
                matching_indices.append(idx)
        
        return events.loc[matching_indices]
    
    def _create_trip_from_waypoints(self, df: pd.DataFrame, trip_data: Dict[str, Any],
                                  vehicle_id: int) -> Optional[Trip]:
        """Create a Trip object from waypoint data with timezone-aware timestamps."""
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
        
        # Ensure timestamps are timezone-aware
        start_time = trip_data['start_time']
        end_time = trip_data['end_time']
        
        if start_time.tzinfo is None:
            start_time = pytz.UTC.localize(start_time)
        if end_time.tzinfo is None:
            end_time = pytz.UTC.localize(end_time)
        
        # Create Trip object
        return Trip(
            trip_id=f"{vehicle_id}_{start_time.strftime('%Y%m%d_%H%M%S')}",
            vehicle_id=vehicle_id,
            driver_id=trip_data.get('driver_id'),
            start_place=trip_data['start_place'],
            end_place=trip_data['end_place'],
            start_time=start_time,
            end_time=end_time,
            duration_minutes=round(duration, 1),
            distance_m=round(total_distance, 1),
            route=route_points,
            is_round_trip=is_round_trip,
            trip_type='two_point_round' if is_round_trip else 'simple'
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
        """Finalize a trip and create Trip object with timezone-aware timestamps."""
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
        
        # Ensure timestamps are timezone-aware
        start_time = trip_data['start_time']
        end_time = trip_data['last_arr_time']
        
        if hasattr(start_time, 'tzinfo') and start_time.tzinfo is None:
            start_time = pytz.UTC.localize(start_time)
        if hasattr(end_time, 'tzinfo') and end_time.tzinfo is None:
            end_time = pytz.UTC.localize(end_time)
        
        # Create Trip object
        return Trip(
            trip_id=f"{vehicle_id}_{start_time.strftime('%Y%m%d_%H%M%S')}",
            vehicle_id=vehicle_id,
            driver_id=trip_data.get('driver_id'),
            start_place=trip_data['start_place'],
            end_place=trip_data['end_place'],
            start_time=start_time,
            end_time=end_time,
            duration_minutes=round(duration, 1),
            distance_m=round(total_distance, 1),
            route=route_points,
            is_round_trip=trip_data['start_place'] == trip_data['end_place'],
            trip_type='two_point_round' if trip_data['start_place'] == trip_data['end_place'] else 'simple'
        )
    
    def _calculate_route_distance(self, route_df: pd.DataFrame) -> float:
        """Calculate total distance of a route."""
        coords = list(zip(route_df['latitude'], route_df['longitude']))
        return self.data_processor.calculate_route_distance(coords)
    
    def _extract_route_points(self, route_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract route points from DataFrame."""
        route_points = []
        
        for _, point in route_df.iterrows():
            timestamp = point['timestamp']
            # Convert to ISO format string with timezone
            if hasattr(timestamp, 'isoformat'):
                timestamp_str = timestamp.isoformat()
            else:
                timestamp_str = str(timestamp)
            
            route_points.append({
                'lat': float(point['latitude']),
                'lon': float(point['longitude']),
                'timestamp': timestamp_str,
                'speed_kmh': float(point.get('speedKmh', 0)),
                'place_name': point.get('placeName'),
                'reason_code': point.get('reasonCode')
            })
        
        return route_points
    
    def split_trips_at_driver_changes(self, trips: List[Trip], history_data: List[Dict[str, Any]]) -> List[Trip]:
        """
        Split trips when driver changes occur mid-trip.
        
        Args:
            trips: List of trips to check for driver changes
            history_data: Raw history data with driver information
            
        Returns:
            List of trips with splits at driver changes
        """
        if self.config.shift_detection_mode != 'driver_based':
            return trips
        
        split_trips = []
        
        for trip in trips:
            # Check if driver changed during this trip
            driver_changes = self._detect_driver_changes_during_trip(
                trip, history_data
            )
            
            if not driver_changes:
                # No driver changes, keep trip as is
                split_trips.append(trip)
            else:
                # Split the trip at each driver change
                split_parts = self._split_trip_at_changes(trip, driver_changes)
                split_trips.extend(split_parts)
        
        return split_trips
    
    def _detect_driver_changes_during_trip(self, trip: Trip, 
                                          history_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect if driver changed during a trip.
        
        Args:
            trip: Trip to check
            history_data: Raw history data
            
        Returns:
            List of driver changes with timestamps and locations
        """
        changes = []
        
        # Filter history data to trip time period
        trip_history = []
        for record in history_data:
            timestamp = record.get('deviceTimeUtc') or record.get('timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                elif isinstance(timestamp, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp, pytz.UTC)
                
                if timestamp.tzinfo is None:
                    timestamp = pytz.UTC.localize(timestamp)
                
                if trip.start_time <= timestamp <= trip.end_time:
                    trip_history.append({
                        'timestamp': timestamp,
                        'driver_id': record.get('driverId') or 'UNKNOWN',
                        'driver_name': record.get('driverName'),
                        'latitude': record.get('latitude'),
                        'longitude': record.get('longitude'),
                        'place_name': record.get('placeName')
                    })
        
        # Sort by timestamp
        trip_history.sort(key=lambda x: x['timestamp'])
        
        # Detect changes
        prev_driver = trip.driver_id or 'UNKNOWN'
        for record in trip_history:
            current_driver = record['driver_id']
            if current_driver != prev_driver:
                changes.append({
                    'timestamp': record['timestamp'],
                    'from_driver': prev_driver,
                    'to_driver': current_driver,
                    'driver_name': record.get('driver_name'),
                    'location': (record['latitude'], record['longitude']),
                    'place_name': record.get('place_name')
                })
                prev_driver = current_driver
        
        return changes
    
    def _split_trip_at_changes(self, trip: Trip, 
                              driver_changes: List[Dict[str, Any]]) -> List[Trip]:
        """
        Split a trip at driver change points.
        
        Args:
            trip: Trip to split
            driver_changes: List of driver changes with timestamps
            
        Returns:
            List of split trips
        """
        split_trips = []
        
        # Sort changes by timestamp
        driver_changes.sort(key=lambda x: x['timestamp'])
        
        # Create trips for each segment
        start_time = trip.start_time
        start_driver = trip.driver_id or 'UNKNOWN'
        
        for i, change in enumerate(driver_changes):
            change_time = change['timestamp']
            
            # Calculate split ratio based on time
            total_duration = (trip.end_time - trip.start_time).total_seconds()
            segment_duration = (change_time - start_time).total_seconds()
            split_ratio = segment_duration / total_duration if total_duration > 0 else 0.5
            
            # Create partial trip for current driver
            partial_trip = Trip(
                trip_id=f"{trip.trip_id}_part{i+1}",
                vehicle_id=trip.vehicle_id,
                driver_id=start_driver,
                start_place=trip.start_place if i == 0 else f"Handover at {change.get('place_name', 'Unknown')}",
                end_place=f"Handover at {change.get('place_name', 'Unknown')}",
                start_time=start_time,
                end_time=change_time,
                duration_minutes=(segment_duration / 60),
                distance_m=trip.distance_m * split_ratio,  # Proportional split
                route=self._split_route(trip.route, start_time, change_time),
                is_round_trip=False,  # Partial trips are not complete round trips
                trip_type=trip.trip_type,
                trip_segments=[],  # Will be recalculated if needed
                waypoints_visited=trip.waypoints_visited if i == 0 else [],
                is_complete_round_trip=False,
                target_waypoint=trip.target_waypoint,
                is_partial=True,
                handover_reason=f"Driver change: {change['from_driver']} to {change['to_driver']}"
            )
            split_trips.append(partial_trip)
            
            # Update for next segment
            start_time = change_time
            start_driver = change['to_driver']
        
        # Create final segment
        final_duration = (trip.end_time - start_time).total_seconds()
        final_ratio = final_duration / total_duration if total_duration > 0 else 0.5
        
        final_trip = Trip(
            trip_id=f"{trip.trip_id}_part{len(driver_changes)+1}",
            vehicle_id=trip.vehicle_id,
            driver_id=start_driver,
            start_place=f"Handover at {driver_changes[-1].get('place_name', 'Unknown')}",
            end_place=trip.end_place,
            start_time=start_time,
            end_time=trip.end_time,
            duration_minutes=(final_duration / 60),
            distance_m=trip.distance_m * final_ratio,
            route=self._split_route(trip.route, start_time, trip.end_time),
            is_round_trip=False,
            trip_type=trip.trip_type,
            trip_segments=[],
            waypoints_visited=[trip.end_place] if trip.end_place in trip.waypoints_visited else [],
            is_complete_round_trip=False,
            target_waypoint=trip.target_waypoint,
            is_partial=True,
            handover_reason=f"Continuation after driver change"
        )
        split_trips.append(final_trip)
        
        return split_trips
    
    def _split_route(self, route: List[Dict[str, Any]], 
                    start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        Split route points based on time range.
        
        Args:
            route: Original route points
            start_time: Start time for segment
            end_time: End time for segment
            
        Returns:
            Route points within the time range
        """
        segment_route = []
        
        for point in route:
            point_time = datetime.fromisoformat(point['timestamp'])
            if point_time.tzinfo is None:
                point_time = pytz.UTC.localize(point_time)
            
            if start_time <= point_time <= end_time:
                segment_route.append(point)
        
        return segment_route