"""
Trip Report Generator for Fleet Shift Analysis

Generates detailed reports showing arrival times, dwell times, and departure times
at each waypoint for all vehicle trips. Includes caching support for API data.
"""
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import pytz

from models import Trip, TripSegment

logger = logging.getLogger(__name__)


@dataclass
class WaypointVisit:
    """Represents a visit to a waypoint with timing details."""
    waypoint_name: str
    arrival_time: Optional[datetime]
    departure_time: Optional[datetime]
    dwell_time_minutes: float
    arrival_reason_code: Optional[str] = None
    departure_reason_code: Optional[str] = None
    is_complete: bool = True  # False if missing arrival or departure
    notes: str = ""


@dataclass
class TripTimeline:
    """Complete timeline for a single trip."""
    trip_id: str
    vehicle_id: int
    vehicle_ref: str
    vehicle_name: str
    driver_id: Optional[str]
    trip_type: str
    trip_date: str  # Local date
    
    # Waypoint visits
    start_visit: Optional[WaypointVisit]
    target_visit: Optional[WaypointVisit] = None  # For 3-point trips
    end_visit: Optional[WaypointVisit] = None
    
    # Overall metrics
    total_trip_time_minutes: float = 0
    total_travel_time_minutes: float = 0
    total_dwell_time_minutes: float = 0
    distance_km: float = 0
    
    # Status
    is_complete: bool = True
    missing_segments: List[str] = field(default_factory=list)


class TripReportGenerator:
    """Generates detailed trip reports with waypoint timings and caching support."""
    
    def __init__(self, timezone: str = 'Australia/Perth', cache_dir: str = './trip_report_cache'):
        """
        Initialize the report generator.
        
        Args:
            timezone: Timezone for displaying times in reports
            cache_dir: Directory for caching API data
        """
        self.timezone = timezone
        self.tz = pytz.timezone(timezone)
        self.grouping_window_minutes = 10  # Group arrivals/departures within 10 minutes
        
        # Cache configuration
        self.cache_dir = Path(cache_dir)
        self.cache_enabled = True
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Cache directory: {self.cache_dir}")
    
    def _get_cache_key(self, fleet_id: int, vehicle_id: int, start_date: str, end_date: str) -> str:
        """Generate cache key for vehicle data."""
        return f"fleet_{fleet_id}_vehicle_{vehicle_id}_{start_date}_{end_date}"
    
    def _get_cache_file(self, cache_key: str) -> Path:
        """Get cache file path for a cache key."""
        return self.cache_dir / f"{cache_key}.json"
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load vehicle data from cache if available."""
        if not self.cache_enabled:
            return None
            
        cache_file = self._get_cache_file(cache_key)
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.debug(f"Loaded cached data: {cache_key}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]):
        """Save vehicle data to cache."""
        if not self.cache_enabled:
            return
            
        cache_file = self._get_cache_file(cache_key)
        try:
            # Add metadata
            cached_data = {
                'cached_at': datetime.now().isoformat(),
                'cache_key': cache_key,
                'data': data
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cached_data, f, indent=2, default=str)
            logger.debug(f"Saved to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")
    
    def set_cache_enabled(self, enabled: bool):
        """Enable or disable caching."""
        self.cache_enabled = enabled
        logger.info(f"Cache {'enabled' if enabled else 'disabled'}")
        
    def clear_cache(self, fleet_id: Optional[int] = None):
        """Clear cache files."""
        if fleet_id:
            # Clear only specific fleet
            pattern = f"fleet_{fleet_id}_*.json"
            cache_files = list(self.cache_dir.glob(pattern))
            for file in cache_files:
                file.unlink()
            logger.info(f"Cleared {len(cache_files)} cache files for fleet {fleet_id}")
        else:
            # Clear all cache
            cache_files = list(self.cache_dir.glob("*.json"))
            for file in cache_files:
                file.unlink()
            logger.info(f"Cleared {len(cache_files)} cache files")
    
    def generate_trip_reports(self, fleet_data: Dict[int, Dict[str, Any]], 
                            output_path: Path,
                            fleet_id: Optional[int] = None,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            use_cache: bool = True) -> Path:
        """
        Generate consolidated trip report for all vehicles.
        
        Args:
            fleet_data: Dictionary of vehicle_id -> vehicle analysis data
            output_path: Directory to save reports
            fleet_id: Fleet ID for caching
            start_date: Start date for caching (YYYYMMDD format)
            end_date: End date for caching (YYYYMMDD format)
            use_cache: Whether to use cached data if available
            
        Returns:
            Path to generated report file
        """
        logger.info("Generating trip timeline reports")
        
        # Store original cache setting
        original_cache_enabled = self.cache_enabled
        self.cache_enabled = self.cache_enabled and use_cache
        
        all_timelines = []
        
        # Process each vehicle
        for vehicle_id, vehicle_data in fleet_data.items():
            vehicle_name = vehicle_data.get('vehicle_name', f'Vehicle_{vehicle_id}')
            vehicle_ref = vehicle_data.get('vehicle_ref', 'Unknown')
            
            # Try to load from cache if we have the necessary info
            cached_data = None
            cache_key = None
            
            if self.cache_enabled and fleet_id and start_date and end_date:
                cache_key = self._get_cache_key(fleet_id, vehicle_id, start_date, end_date)
                cached_data = self._load_from_cache(cache_key)
            
            if cached_data:
                # Use cached data
                vehicle_data = cached_data['data']
                logger.info(f"Using cached data for {vehicle_name} ({vehicle_ref})")
            else:
                # Save to cache if we have the necessary info
                if self.cache_enabled and fleet_id and start_date and end_date and cache_key:
                    self._save_to_cache(cache_key, vehicle_data)
            
            trips = vehicle_data.get('trips', [])
            
            # Get raw history if available (for detailed timing)
            history = vehicle_data.get('raw_history', [])
            
            logger.debug(f"Processing {len(trips)} trips for {vehicle_name} ({vehicle_ref})")
            
            # Process each trip
            for trip_data in trips:
                # Convert trip dict back to Trip object if needed
                if isinstance(trip_data, dict):
                    trip = self._dict_to_trip(trip_data)
                else:
                    trip = trip_data
                
                # Generate timeline for this trip
                timeline = self._generate_trip_timeline(
                    trip, vehicle_id, vehicle_name, vehicle_ref, history
                )
                
                if timeline:
                    all_timelines.append(timeline)
        
        # Restore original cache setting
        self.cache_enabled = original_cache_enabled
        
        # Generate reports
        report_path = self._save_consolidated_report(all_timelines, output_path)
        
        # Also generate summary statistics
        self._save_summary_report(all_timelines, output_path)
        
        logger.info(f"Trip reports saved to {report_path}")
        
        # Log cache statistics
        if self.cache_enabled:
            cache_files = list(self.cache_dir.glob("*.json"))
            logger.info(f"Cache contains {len(cache_files)} files")
        
        return report_path
    
    def generate_trip_reports_from_api_client(self, api_client, fleet_id: int, 
                                            vehicles: List[Dict[str, Any]],
                                            start_datetime: datetime, 
                                            end_datetime: datetime,
                                            output_path: Path,
                                            shift_config = None,
                                            use_cache: bool = True) -> Path:
        """
        Generate trip reports by fetching data from API client with caching.
        
        Args:
            api_client: MTDATA API client
            fleet_id: Fleet ID
            vehicles: List of vehicle dictionaries
            start_datetime: Start datetime (UTC)
            end_datetime: End datetime (UTC)
            output_path: Output directory
            shift_config: Shift configuration for trip extraction
            use_cache: Whether to use cached data
            
        Returns:
            Path to generated report file
        """
        from analyzers.trip_extractor import TripExtractor
        
        # Initialize trip extractor if shift_config provided
        trip_extractor = TripExtractor(shift_config) if shift_config else None
        
        # Format dates for cache keys
        start_date_str = start_datetime.strftime('%Y%m%d')
        end_date_str = end_datetime.strftime('%Y%m%d')
        
        # Build fleet data
        fleet_data = {}
        
        for vehicle in vehicles:
            vehicle_id = vehicle['id']
            vehicle_name = vehicle.get('displayName', f'Vehicle_{vehicle_id}')
            
            # Check cache first
            cache_key = self._get_cache_key(fleet_id, vehicle_id, start_date_str, end_date_str)
            cached_data = self._load_from_cache(cache_key) if use_cache else None
            
            if cached_data:
                logger.info(f"Using cached data for {vehicle_name}")
                fleet_data[vehicle_id] = cached_data['data']
            else:
                logger.info(f"Fetching data for {vehicle_name} from API")
                
                try:
                    # Get vehicle history from API
                    history = api_client.get_vehicle_history(
                        fleet_id=fleet_id,
                        vehicle_id=vehicle_id,
                        start_date=start_datetime,
                        end_date=end_datetime
                    )
                    
                    if history and len(history) > 0:
                        vehicle_ref = history[0].get('vehicleRef', 'Unknown')
                    else:
                        vehicle_ref = 'Unknown'
                    
                    # Extract trips if we have trip extractor
                    trips = []
                    if trip_extractor and history:
                        history_data = {
                            'vehicleId': vehicle_id,
                            'fleetId': fleet_id,
                            'history': history
                        }
                        trips = trip_extractor.extract_trips(history_data)
                        trips = [trip.to_dict() for trip in trips]
                    
                    # Build vehicle data
                    vehicle_data = {
                        'vehicle_id': vehicle_id,
                        'vehicle_name': vehicle_name,
                        'vehicle_ref': vehicle_ref,
                        'trips': trips,
                        'raw_history': history
                    }
                    
                    fleet_data[vehicle_id] = vehicle_data
                    
                    # Save to cache
                    if self.cache_enabled:
                        self._save_to_cache(cache_key, vehicle_data)
                    
                except Exception as e:
                    logger.error(f"Error processing vehicle {vehicle_id}: {e}")
                    continue
        
        # Generate reports
        return self.generate_trip_reports(
            fleet_data, output_path, fleet_id, start_date_str, end_date_str, use_cache=False
        )
    
    
    def _generate_trip_timeline(self, trip: Trip, vehicle_id: int, 
                              vehicle_name: str, vehicle_ref: str,
                              history: List[Dict]) -> Optional[TripTimeline]:
        """Generate timeline for a single trip."""
        # Convert trip start/end times to local timezone for display
        local_start = trip.start_time.astimezone(self.tz)
        trip_date = local_start.strftime('%Y-%m-%d')
        
        # Initialize timeline
        timeline = TripTimeline(
            trip_id=trip.trip_id,
            vehicle_id=vehicle_id,
            vehicle_ref=vehicle_ref,
            vehicle_name=vehicle_name,
            driver_id=trip.driver_id,
            trip_type=trip.trip_type,
            trip_date=trip_date,
            start_visit=None,
            total_trip_time_minutes=trip.duration_minutes,
            distance_km=trip.distance_km
        )
        
        # Extract waypoint visits based on trip type
        if trip.trip_type == 'three_point_round':
            timeline = self._extract_three_point_timeline(trip, timeline, history)
        else:
            timeline = self._extract_simple_timeline(trip, timeline, history)
        
        # Calculate travel vs dwell time
        timeline.total_dwell_time_minutes = self._calculate_total_dwell_time(timeline)
        timeline.total_travel_time_minutes = timeline.total_trip_time_minutes - timeline.total_dwell_time_minutes
        
        return timeline
    
    def _extract_three_point_timeline(self, trip: Trip, timeline: TripTimeline,
                                    history: List[Dict]) -> TripTimeline:
        """Extract timeline for three-point round trip."""
        # For three-point trips, we have segments
        if trip.trip_segments:
            # Start waypoint (from first segment)
            if len(trip.trip_segments) > 0:
                seg1 = trip.trip_segments[0]
                timeline.start_visit = WaypointVisit(
                    waypoint_name=seg1.from_waypoint,
                    arrival_time=None,  # Trip starts with departure
                    departure_time=seg1.start_time,
                    dwell_time_minutes=0,
                    departure_reason_code='DEP',
                    notes="Trip origin"
                )
                
                # Target waypoint
                timeline.target_visit = WaypointVisit(
                    waypoint_name=seg1.to_waypoint,
                    arrival_time=seg1.end_time,
                    departure_time=None,
                    dwell_time_minutes=0,
                    arrival_reason_code='ARR',
                    is_complete=False,
                    notes="Waiting for departure data"
                )
            
            # If we have second segment, update target departure and add end
            if len(trip.trip_segments) > 1:
                seg2 = trip.trip_segments[1]
                
                # Update target departure
                if timeline.target_visit:
                    timeline.target_visit.departure_time = seg2.start_time
                    timeline.target_visit.departure_reason_code = 'DEP'
                    timeline.target_visit.is_complete = True
                    
                    # Calculate dwell time at target
                    if timeline.target_visit.arrival_time and timeline.target_visit.departure_time:
                        dwell = (timeline.target_visit.departure_time - timeline.target_visit.arrival_time).total_seconds() / 60
                        timeline.target_visit.dwell_time_minutes = max(0, dwell)
                        timeline.target_visit.notes = f"Dwell time: {dwell:.1f} min"
                
                # End waypoint
                timeline.end_visit = WaypointVisit(
                    waypoint_name=seg2.to_waypoint,
                    arrival_time=seg2.end_time,
                    departure_time=None,
                    dwell_time_minutes=0,
                    arrival_reason_code='ARR',
                    notes="Trip end"
                )
            else:
                timeline.is_complete = False
                timeline.missing_segments.append("Target to End segment")
        else:
            # No segments - try to extract from route points
            timeline = self._extract_timeline_from_route(trip, timeline, history)
        
        # Check completeness
        if not trip.is_complete_round_trip:
            timeline.is_complete = False
            if len(trip.waypoints_visited) < 3:
                missing = []
                if trip.target_waypoint not in trip.waypoints_visited:
                    missing.append("Target waypoint")
                if trip.end_place not in trip.waypoints_visited:
                    missing.append("End waypoint")
                timeline.missing_segments.extend(missing)
        
        return timeline
    
    def _extract_simple_timeline(self, trip: Trip, timeline: TripTimeline,
                               history: List[Dict]) -> TripTimeline:
        """Extract timeline for simple trip."""
        # For simple trips, we have start and end
        timeline.start_visit = WaypointVisit(
            waypoint_name=trip.start_place,
            arrival_time=None,  # Trip starts with departure
            departure_time=trip.start_time,
            dwell_time_minutes=0,
            departure_reason_code='DEP',
            notes="Trip origin"
        )
        
        timeline.end_visit = WaypointVisit(
            waypoint_name=trip.end_place,
            arrival_time=trip.end_time,
            departure_time=None,
            dwell_time_minutes=0,
            arrival_reason_code='ARR',
            notes="Trip end"
        )
        
        # Try to find dwell times from history if available
        if history:
            timeline = self._enhance_timeline_from_history(trip, timeline, history)
        
        return timeline
    
    def _extract_timeline_from_route(self, trip: Trip, timeline: TripTimeline,
                                   history: List[Dict]) -> TripTimeline:
        """Extract timeline from route points when segments aren't available."""
        # This is a fallback method using the route points
        # Would need the raw history data to get accurate arrival/departure times
        logger.debug(f"Extracting timeline from route for trip {trip.trip_id}")
        
        # For now, estimate based on trip data
        if trip.waypoints_visited:
            # We know which waypoints were visited
            total_duration = trip.duration_minutes
            num_waypoints = len(trip.waypoints_visited)
            
            # Simple estimation: divide time equally
            estimated_segment_time = total_duration / max(num_waypoints - 1, 1)
            
            current_time = trip.start_time
            
            for i, waypoint in enumerate(trip.waypoints_visited):
                if i == 0:  # Start
                    timeline.start_visit = WaypointVisit(
                        waypoint_name=waypoint,
                        arrival_time=None,
                        departure_time=current_time,
                        dwell_time_minutes=0,
                        notes="Estimated from route"
                    )
                elif i == 1 and trip.trip_type == 'three_point_round':  # Target
                    arrival = current_time
                    departure = arrival + timedelta(minutes=10)  # Assume 10 min dwell
                    timeline.target_visit = WaypointVisit(
                        waypoint_name=waypoint,
                        arrival_time=arrival,
                        departure_time=departure,
                        dwell_time_minutes=10,
                        notes="Estimated times"
                    )
                    current_time = departure + timedelta(minutes=estimated_segment_time)
                elif i == len(trip.waypoints_visited) - 1:  # End
                    timeline.end_visit = WaypointVisit(
                        waypoint_name=waypoint,
                        arrival_time=current_time,
                        departure_time=None,
                        dwell_time_minutes=0,
                        notes="Estimated from route"
                    )
        
        return timeline
    
    def _enhance_timeline_from_history(self, trip: Trip, timeline: TripTimeline,
                                     history: List[Dict]) -> TripTimeline:
        """Enhance timeline with actual arrival/departure times from history."""
        # Convert history to DataFrame for easier processing
        if not history:
            return timeline
        
        df = pd.DataFrame(history)
        if 'deviceTimeUtc' not in df.columns:
            return timeline
        
        # Convert timestamps
        df['timestamp'] = pd.to_datetime(df['deviceTimeUtc'])
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        
        # Filter to trip time range (with small buffer)
        trip_start = trip.start_time - timedelta(minutes=30)
        trip_end = trip.end_time + timedelta(minutes=30)
        
        mask = (df['timestamp'] >= trip_start) & (df['timestamp'] <= trip_end)
        trip_df = df[mask].copy()
        
        if trip_df.empty:
            return timeline
        
        # Look for arrival/departure events at waypoints
        # This would require matching place names and reason codes
        # For now, return the timeline as-is
        
        return timeline
    
    def _calculate_total_dwell_time(self, timeline: TripTimeline) -> float:
        """Calculate total dwell time across all waypoints."""
        total = 0
        
        if timeline.start_visit:
            total += timeline.start_visit.dwell_time_minutes
        
        if timeline.target_visit:
            total += timeline.target_visit.dwell_time_minutes
            
        if timeline.end_visit:
            total += timeline.end_visit.dwell_time_minutes
        
        return total
    
    def _save_consolidated_report(self, timelines: List[TripTimeline], 
                                output_path: Path) -> Path:
        """Save consolidated report as CSV."""
        # Convert timelines to DataFrame-friendly format
        report_data = []
        
        for timeline in timelines:
            # Base row data
            base_data = {
                'trip_date': timeline.trip_date,
                'vehicle_ref': timeline.vehicle_ref,
                'vehicle_name': timeline.vehicle_name,
                'driver_id': timeline.driver_id or 'Unknown',
                'trip_type': timeline.trip_type,
                'trip_id': timeline.trip_id,
                'total_duration_min': f"{timeline.total_trip_time_minutes:.1f}",
                'travel_time_min': f"{timeline.total_travel_time_minutes:.1f}",
                'total_dwell_min': f"{timeline.total_dwell_time_minutes:.1f}",
                'distance_km': f"{timeline.distance_km:.1f}",
                'is_complete': 'Yes' if timeline.is_complete else 'No',
                'missing': ', '.join(timeline.missing_segments) if timeline.missing_segments else ''
            }
            
            # Add waypoint details
            # Start waypoint
            if timeline.start_visit:
                base_data.update({
                    'start_waypoint': timeline.start_visit.waypoint_name,
                    'start_departure': self._format_time(timeline.start_visit.departure_time),
                    'start_dwell_min': f"{timeline.start_visit.dwell_time_minutes:.1f}"
                })
            
            # Target waypoint (for 3-point trips)
            if timeline.target_visit:
                base_data.update({
                    'target_waypoint': timeline.target_visit.waypoint_name,
                    'target_arrival': self._format_time(timeline.target_visit.arrival_time),
                    'target_departure': self._format_time(timeline.target_visit.departure_time),
                    'target_dwell_min': f"{timeline.target_visit.dwell_time_minutes:.1f}"
                })
            else:
                base_data.update({
                    'target_waypoint': '',
                    'target_arrival': '',
                    'target_departure': '',
                    'target_dwell_min': ''
                })
            
            # End waypoint
            if timeline.end_visit:
                base_data.update({
                    'end_waypoint': timeline.end_visit.waypoint_name,
                    'end_arrival': self._format_time(timeline.end_visit.arrival_time),
                    'end_dwell_min': f"{timeline.end_visit.dwell_time_minutes:.1f}"
                })
            else:
                base_data.update({
                    'end_waypoint': '',
                    'end_arrival': '',
                    'end_dwell_min': ''
                })
            
            report_data.append(base_data)
        
        # Create DataFrame and save
        df = pd.DataFrame(report_data)
        
        # Sort by date and vehicle
        if not df.empty:
            df = df.sort_values(['trip_date', 'vehicle_ref', 'start_departure'])
        
        # Save to CSV
        report_file = output_path / f'trip_timeline_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(report_file, index=False, encoding='utf-8')
        
        logger.info(f"Saved trip timeline report with {len(timelines)} trips")
        
        return report_file
    
    def generate_trip_reports_from_cache_dir(self, cache_dir: Path, output_path: Path, 
                                            fleet_id: int, shift_config = None,
                                            vehicle_ref_filter: Optional[str] = None) -> Path:
        """
        Generate trip reports from a directory of simulation cache files.
        
        This method is designed to work with simulation cache files that follow
        the naming pattern: fleet_{fleet_id}_vehicle_{vehicle_id}_*.json
        
        Args:
            cache_dir: Directory containing simulation cache files
            output_path: Directory to save reports
            fleet_id: Fleet ID to filter cache files
            shift_config: Shift configuration for trip extraction
            vehicle_ref_filter: Optional vehicle ref to filter to specific vehicle
            
        Returns:
            Path to generated report file
        """
        from analyzers.trip_extractor import TripExtractor
        
        logger.info(f"Generating trip reports from cache directory: {cache_dir}")

        if vehicle_ref_filter:
            logger.info(f"Filtering to vehicle ref: {vehicle_ref_filter}")
        
        # Initialize trip extractor if shift_config provided
        trip_extractor = TripExtractor(shift_config) if shift_config else None
        
        # Find all cache files for this fleet
        cache_pattern = f"fleet_{fleet_id}_vehicle_*.json"
        cache_files = list(cache_dir.glob(cache_pattern))
        
        if not cache_files:
            logger.warning(f"No cache files found matching pattern: {cache_pattern}")
            return None
        
        logger.info(f"Found {len(cache_files)} cached vehicle files")
        
        # Build fleet data from cache
        fleet_data = {}
        
        for cache_file in cache_files:
            try:
                # Extract vehicle ID from filename
                # Format: fleet_104002_vehicle_12345_20250616_000000_to_20250618_000000.json
                parts = cache_file.stem.split('_')
                if len(parts) >= 4 and parts[2] == 'vehicle':
                    vehicle_id = int(parts[3])
                else:
                    logger.warning(f"Unexpected filename format: {cache_file.name}")
                    continue
                
                # Load cached data
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # The simulation cache stores data as a direct array of history events
                if isinstance(cached_data, list):
                    history = cached_data
                else:
                    logger.warning(f"Unexpected data format in {cache_file.name} - expected list")
                    continue
                
                # Get vehicle info from first history record
                if history and len(history) > 0:
                    vehicle_ref = history[0].get('vehicleRef', 'Unknown')
                    if vehicle_ref_filter and vehicle_ref != vehicle_ref_filter:
                        logger.debug(f"Skipping vehicle {vehicle_ref} (not matching filter)")
                        continue
                else:
                    vehicle_ref = 'Unknown'
                    logger.warning(f"No history data in {cache_file.name}")
                    continue
                
                # Extract trips if we have trip extractor
                trips = []
                if trip_extractor and history:
                    # Wrap history in expected format for trip extractor
                    history_data = {
                        'vehicleId': vehicle_id,
                        'fleetId': fleet_id,
                        'history': history
                    }
                    trips = trip_extractor.extract_trips(history_data)
                    trips = [trip.to_dict() for trip in trips]
                    logger.debug(f"Extracted {len(trips)} trips for vehicle {vehicle_id} ({vehicle_ref})")
                
                # Build vehicle data
                fleet_data[vehicle_id] = {
                    'vehicle_id': vehicle_id,
                    'vehicle_name': f'Vehicle_{vehicle_id}',
                    'vehicle_ref': vehicle_ref,
                    'trips': trips,
                    'raw_history': history
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {cache_file.name}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing cache file {cache_file.name}: {e}")
                continue
        
        if not fleet_data:
            logger.error("No valid vehicle data extracted from cache files")
            return None
        
        logger.info(f"Successfully loaded data for {len(fleet_data)} vehicles")
        
        # Generate reports (disable cache since we're already using cached data)
        return self.generate_trip_reports(
            fleet_data, 
            output_path,
            use_cache=False
        )
    
    def _save_summary_report(self, timelines: List[TripTimeline], output_path: Path):
        """Save summary statistics report."""
        if not timelines:
            return
        
        # Calculate summary statistics
        summary_data = []
        
        # Group by vehicle
        from collections import defaultdict
        vehicle_stats = defaultdict(lambda: {
            'trips': 0,
            'complete_trips': 0,
            'total_duration': 0,
            'total_travel': 0,
            'total_dwell': 0,
            'total_distance': 0,
            'dwell_times': []
        })
        
        for timeline in timelines:
            stats = vehicle_stats[timeline.vehicle_ref]
            stats['trips'] += 1
            if timeline.is_complete:
                stats['complete_trips'] += 1
            stats['total_duration'] += timeline.total_trip_time_minutes
            stats['total_travel'] += timeline.total_travel_time_minutes
            stats['total_dwell'] += timeline.total_dwell_time_minutes
            stats['total_distance'] += timeline.distance_km
            
            # Collect dwell times
            if timeline.start_visit and timeline.start_visit.dwell_time_minutes > 0:
                stats['dwell_times'].append(timeline.start_visit.dwell_time_minutes)
            if timeline.target_visit and timeline.target_visit.dwell_time_minutes > 0:
                stats['dwell_times'].append(timeline.target_visit.dwell_time_minutes)
            if timeline.end_visit and timeline.end_visit.dwell_time_minutes > 0:
                stats['dwell_times'].append(timeline.end_visit.dwell_time_minutes)
        
        # Create summary rows
        for vehicle_ref, stats in vehicle_stats.items():
            avg_dwell = sum(stats['dwell_times']) / len(stats['dwell_times']) if stats['dwell_times'] else 0
            
            summary_data.append({
                'vehicle_ref': vehicle_ref,
                'total_trips': stats['trips'],
                'complete_trips': stats['complete_trips'],
                'completion_rate': f"{(stats['complete_trips'] / stats['trips'] * 100):.1f}%",
                'avg_trip_duration_min': f"{stats['total_duration'] / stats['trips']:.1f}",
                'avg_travel_time_min': f"{stats['total_travel'] / stats['trips']:.1f}",
                'avg_dwell_time_min': f"{avg_dwell:.1f}",
                'total_distance_km': f"{stats['total_distance']:.1f}",
                'avg_distance_per_trip_km': f"{stats['total_distance'] / stats['trips']:.1f}"
            })
        
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('vehicle_ref')
        
        summary_file = output_path / 'trip_timeline_summary.csv'
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        
        logger.info(f"Saved trip timeline summary for {len(vehicle_stats)} vehicles")
    
    def _format_time(self, dt: Optional[datetime]) -> str:
        """Format datetime for display in local timezone."""
        if not dt:
            return ''
        
        # Ensure timezone aware
        if dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        
        # Convert to local timezone
        local_dt = dt.astimezone(self.tz)
        
        return local_dt.strftime('%Y-%m-%d %H:%M:%S %Z')
    
    def _dict_to_trip(self, trip_dict: Dict[str, Any]) -> Trip:
        """Convert trip dictionary back to Trip object."""
        from models import Trip, TripSegment
        
        # Parse timestamps
        start_time = datetime.fromisoformat(trip_dict['start_time'].replace('Z', '+00:00'))
        end_time = datetime.fromisoformat(trip_dict['end_time'].replace('Z', '+00:00'))
        
        # Recreate segments if present
        segments = []
        for seg_data in trip_dict.get('trip_segments', []):
            if isinstance(seg_data, dict) and 'segment_id' in seg_data:
                # Parse segment-specific timestamps if available
                if 'start_time' in seg_data and 'end_time' in seg_data:
                    seg_start_time = datetime.fromisoformat(seg_data['start_time'].replace('Z', '+00:00'))
                    seg_end_time = datetime.fromisoformat(seg_data['end_time'].replace('Z', '+00:00'))
                else:
                    # Fallback to trip times (this is the current bug)
                    logger.warning(f"Segment {seg_data['segment_id']} missing timestamps, using trip times")
                    seg_start_time = start_time
                    seg_end_time = end_time
                
                segment = TripSegment(
                    segment_id=seg_data['segment_id'],
                    from_waypoint=seg_data.get('from', ''),
                    to_waypoint=seg_data.get('to', ''),
                    start_time=seg_start_time,
                    end_time=seg_end_time,
                    duration_minutes=seg_data.get('duration_minutes', 0),
                    distance_m=seg_data.get('distance_m', 0),
                    route_points=[]
                )
                segments.append(segment)
        
        return Trip(
            trip_id=trip_dict['trip_id'],
            vehicle_id=trip_dict['vehicle_id'],
            driver_id=trip_dict.get('driver_id'),
            start_place=trip_dict['start_place'],
            end_place=trip_dict['end_place'],
            start_time=start_time,
            end_time=end_time,
            duration_minutes=trip_dict['duration_minutes'],
            distance_m=trip_dict['distance_m'],
            route=trip_dict.get('route', []),
            is_round_trip=trip_dict.get('is_round_trip', False),
            trip_type=trip_dict.get('trip_type', 'simple'),
            trip_segments=segments,
            waypoints_visited=trip_dict.get('waypoints_visited', []),
            is_complete_round_trip=trip_dict.get('is_complete_round_trip', False),
            target_waypoint=trip_dict.get('target_waypoint')
        )
    
    def generate_driver_shift_performance_report(self, fleet_analyses: List[Dict[str, Any]], 
                                               output_path: Path) -> Path:
        """
        Generate driver shift performance report with overtime analysis.
        
        Args:
            fleet_analyses: List of vehicle analysis results
            output_path: Output directory path
            
        Returns:
            Path to generated report
        """
        driver_data = []
        
        for vehicle_analysis in fleet_analyses:
            vehicle_id = vehicle_analysis.get('vehicle_id')
            vehicle_name = vehicle_analysis.get('vehicle_name')
            vehicle_ref = vehicle_analysis.get('vehicle_ref')
            
            for shift_analysis in vehicle_analysis.get('shift_analyses', []):
                metrics = shift_analysis.get('metrics', {})
                shift = shift_analysis.get('shift', {})
                
                # Skip if no driver info
                driver_id = metrics.get('driver_id') or shift.get('driver_id')
                if not driver_id or driver_id == 'UNKNOWN':
                    continue
                
                # Calculate shift times
                shift_start = shift.get('start_time')
                shift_end = shift.get('actual_end_time') or shift.get('end_time')
                
                if shift_start and shift_end:
                    # Parse timestamps
                    if isinstance(shift_start, str):
                        shift_start = datetime.fromisoformat(shift_start.replace('Z', '+00:00'))
                    if isinstance(shift_end, str):
                        shift_end = datetime.fromisoformat(shift_end.replace('Z', '+00:00'))
                    
                    # Convert to local timezone
                    local_start = shift_start.astimezone(self.tz)
                    local_end = shift_end.astimezone(self.tz)
                    
                    driver_data.append({
                        'shift_date': local_start.strftime('%Y-%m-%d'),
                        'driver_id': driver_id,
                        'driver_name': metrics.get('driver_name') or 'Unknown',
                        'driver_ref': metrics.get('driver_ref') or driver_id,
                        'vehicle_id': vehicle_id,
                        'vehicle_ref': vehicle_ref,
                        'vehicle_name': vehicle_name,
                        'shift_start': local_start.strftime('%Y-%m-%d %H:%M'),
                        'shift_end': local_end.strftime('%Y-%m-%d %H:%M'),
                        'shift_duration_hours': metrics.get('shift_duration_hours', 0),
                        'overtime_hours': metrics.get('overtime_hours', 0),
                        'overtime_cost': metrics.get('overtime_cost', 0),
                        'trips_completed': metrics.get('trips_completed', 0),
                        'trips_target': metrics.get('trips_target', 0),
                        'avg_trip_duration': metrics.get('avg_trip_duration', 0),
                        'can_complete_target': shift_analysis.get('can_complete_target', False),
                        'risk_level': shift_analysis.get('risk_level', 'low'),
                        'incomplete_round_trips': metrics.get('incomplete_round_trips', 0)
                    })
        
        # Create DataFrame
        df = pd.DataFrame(driver_data)
        
        if not df.empty:
            # Sort by date and driver
            df = df.sort_values(['shift_date', 'driver_name', 'shift_start'])
            
            # Calculate summary statistics
            summary_stats = {
                'total_drivers': df['driver_id'].nunique(),
                'total_shifts': len(df),
                'total_overtime_hours': df['overtime_hours'].sum(),
                'total_overtime_cost': df['overtime_cost'].sum(),
                'drivers_with_overtime': len(df[df['overtime_hours'] > 0]['driver_id'].unique()),
                'avg_shift_duration': df['shift_duration_hours'].mean(),
                'completion_rate': len(df[df['can_complete_target']]) / len(df) if len(df) > 0 else 0
            }
            
            # Save detailed report
            report_file = output_path / f'driver_shift_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            df.to_csv(report_file, index=False, encoding='utf-8')
            
            # Save summary
            summary_file = output_path / f'driver_shift_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            with open(summary_file, 'w') as f:
                f.write("Driver Shift Performance Summary\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Total Drivers: {summary_stats['total_drivers']}\n")
                f.write(f"Total Shifts: {summary_stats['total_shifts']}\n")
                f.write(f"Average Shift Duration: {summary_stats['avg_shift_duration']:.1f} hours\n")
                f.write(f"Target Completion Rate: {summary_stats['completion_rate']:.1%}\n\n")
                f.write(f"Overtime Analysis:\n")
                f.write(f"  Drivers with Overtime: {summary_stats['drivers_with_overtime']}\n")
                f.write(f"  Total Overtime Hours: {summary_stats['total_overtime_hours']:.1f}\n")
                f.write(f"  Total Overtime Cost: ${summary_stats['total_overtime_cost']:.2f}\n")
            
            logger.info(f"Saved driver shift performance report with {len(df)} shifts")
            
            return report_file
        else:
            logger.warning("No driver shift data found")
            return None