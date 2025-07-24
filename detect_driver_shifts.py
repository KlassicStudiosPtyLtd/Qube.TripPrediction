#!/usr/bin/env python3
"""
Driver Shift Detection System

Detects driver shifts based on driver ID changes in vehicle history data,
with configurable parameters and comprehensive reporting.
"""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytz
from dataclasses import dataclass, asdict

from core.mtdata_api_client import MTDataApiClient
from core.utils import setup_logging, ensure_dir


@dataclass
class ShiftDetectionConfig:
    """Configuration for driver shift detection."""
    fleet_id: int
    start_date: datetime
    end_date: datetime
    timezone: str = "Australia/Perth"
    min_shift_duration_minutes: int = 30
    max_event_gap_hours: int = 4
    min_events_per_shift: int = 5
    vehicle_ref: Optional[str] = None
    cache_dir: Optional[str] = None
    output_dir: str = "./driver_shift_analysis"
    include_null_driver_shifts: bool = False  # Option to include idle/maintenance periods


@dataclass
class ShiftData:
    """Data class representing a detected shift."""
    shift_id: str
    vehicle_id: int
    vehicle_ref: str
    driver_id: Optional[str]
    driver_name: Optional[str]
    driver_ref: Optional[str]
    shift_date: str
    start_time: str
    end_time: str
    duration_hours: float
    total_events: int
    distance_km: float
    start_location_lat: float
    start_location_lon: float
    end_location_lat: float
    end_location_lon: float
    start_place: str
    end_place: str
    max_speed_kmh: float
    total_fuel_used: Optional[float]
    engine_hours: Optional[float]
    shift_type: str = "driver_change"


class DriverShiftDetector:
    """Main class for detecting driver shifts."""
    
    def __init__(self, config: ShiftDetectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.api_client = None
        self.timezone = pytz.timezone(config.timezone)
        
        # Initialize API client if needed
        if not config.cache_dir or not os.path.exists(config.cache_dir):
            self.api_client = MTDataApiClient(self.logger)
    
    def detect_shifts(self) -> Tuple[List[ShiftData], Dict[str, Any]]:
        """
        Main method to detect driver shifts.
        
        Returns:
            Tuple of (shift_list, summary_stats)
        """
        self.logger.info(f"Starting driver shift detection for fleet {self.config.fleet_id}")
        self.logger.info(f"Date range: {self.config.start_date} to {self.config.end_date}")
        
        # Get vehicles to analyze
        vehicles = self._get_vehicles_to_analyze()
        if not vehicles:
            self.logger.warning("No vehicles found to analyze")
            return [], {}
        
        self.logger.info(f"Found {len(vehicles)} vehicles to analyze")
        
        # Process each vehicle
        all_shifts = []
        processed_vehicles = 0
        
        for vehicle in vehicles:
            try:
                vehicle_shifts = self._process_vehicle(vehicle)
                all_shifts.extend(vehicle_shifts)
                processed_vehicles += 1
                self.logger.info(f"Processed vehicle {vehicle['vehicleRef']}: found {len(vehicle_shifts)} shifts")
            except Exception as e:
                self.logger.error(f"Error processing vehicle {vehicle.get('vehicleRef', 'unknown')}: {str(e)}")
                continue
        
        # Generate summary statistics
        summary_stats = self._generate_summary_stats(all_shifts, processed_vehicles)
        
        self.logger.info(f"Completed shift detection: {len(all_shifts)} shifts found across {processed_vehicles} vehicles")
        
        return all_shifts, summary_stats
    
    def _get_vehicles_to_analyze(self) -> List[Dict[str, Any]]:
        """Get list of vehicles to analyze."""
        if self.config.vehicle_ref:
            # Single vehicle mode - look for cache files matching the vehicle
            if self.config.cache_dir:
                cache_files = self._find_cache_files_for_vehicle(self.config.vehicle_ref)
                if cache_files:
                    # Extract vehicle info from cache filename
                    cache_file = cache_files[0]
                    vehicle_id = self._extract_vehicle_id_from_cache(cache_file)
                    return [{"vehicleId": vehicle_id, "vehicleRef": self.config.vehicle_ref}]
            
            # Fallback to API
            if self.api_client:
                vehicles = self.api_client.get_vehicles(self.config.fleet_id)
                return [v for v in vehicles if v.get('vehicleRef') == self.config.vehicle_ref]
        else:
            # Multi-vehicle mode
            if self.config.cache_dir:
                return self._get_vehicles_from_cache()
            
            # Fallback to API
            if self.api_client:
                return self.api_client.get_vehicles(self.config.fleet_id)
        
        return []
    
    def _find_cache_files_for_vehicle(self, vehicle_ref: str) -> List[str]:
        """Find cache files for a specific vehicle."""
        if not self.config.cache_dir or not os.path.exists(self.config.cache_dir):
            return []
        
        cache_files = []
        for root, dirs, files in os.walk(self.config.cache_dir):
            for file in files:
                if file.endswith('.json') and vehicle_ref in root:
                    cache_files.append(os.path.join(root, file))
        
        return cache_files
    
    def _extract_vehicle_id_from_cache(self, cache_file: str) -> int:
        """Extract vehicle ID from cache filename."""
        filename = os.path.basename(cache_file)
        parts = filename.split('_')
        for i, part in enumerate(parts):
            if part == 'vehicle' and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    continue
        return 0
    
    def _get_vehicles_from_cache(self) -> List[Dict[str, Any]]:
        """Get vehicle list from cache directory."""
        if not self.config.cache_dir or not os.path.exists(self.config.cache_dir):
            return []
        
        vehicles = []
        vehicle_ids_seen = set()
        
        for root, dirs, files in os.walk(self.config.cache_dir):
            for file in files:
                if file.endswith('.json') and 'fleet_' in file and 'vehicle_' in file:
                    vehicle_id = self._extract_vehicle_id_from_cache(os.path.join(root, file))
                    if vehicle_id and vehicle_id not in vehicle_ids_seen:
                        vehicle_ids_seen.add(vehicle_id)
                        # Try to determine vehicle ref from directory structure
                        vehicle_ref = self._extract_vehicle_ref_from_path(root)
                        vehicles.append({
                            "vehicleId": vehicle_id,
                            "vehicleRef": vehicle_ref or f"VEH_{vehicle_id}"
                        })
        
        return vehicles
    
    def _extract_vehicle_ref_from_path(self, path: str) -> Optional[str]:
        """Extract vehicle reference from cache directory path."""
        parts = path.split(os.sep)
        for part in parts:
            if part.startswith('PRM') or part.startswith('VEH'):
                return part
        return None
    
    def _process_vehicle(self, vehicle: Dict[str, Any]) -> List[ShiftData]:
        """Process a single vehicle to detect shifts."""
        vehicle_id = vehicle['vehicleId']
        vehicle_ref = vehicle.get('vehicleRef', f'VEH_{vehicle_id}')
        
        # Load vehicle history data
        history_data = self._load_vehicle_history(vehicle_id, vehicle_ref)
        if not history_data:
            self.logger.warning(f"No history data found for vehicle {vehicle_ref}")
            return []
        
        # Sort by timestamp
        history_data.sort(key=lambda x: x.get('deviceTimeUtc', ''))
        
        # Detect shifts based on driver changes
        shifts = self._detect_shifts_from_history(history_data, vehicle_id, vehicle_ref)
        
        return shifts
    
    def _load_vehicle_history(self, vehicle_id: int, vehicle_ref: str) -> List[Dict[str, Any]]:
        """Load vehicle history data from cache or API."""
        # Try cache first
        if self.config.cache_dir:
            cache_data = self._load_from_cache(vehicle_id, vehicle_ref)
            if cache_data:
                return cache_data
        
        # Fallback to API
        if self.api_client:
            try:
                return self.api_client.get_vehicle_history(
                    self.config.fleet_id,
                    vehicle_id,
                    self.config.start_date,
                    self.config.end_date
                )
            except Exception as e:
                self.logger.error(f"Error fetching history from API for vehicle {vehicle_ref}: {str(e)}")
        
        return []
    
    def _load_from_cache(self, vehicle_id: int, vehicle_ref: str) -> List[Dict[str, Any]]:
        """Load vehicle history from cache files."""
        if not self.config.cache_dir or not os.path.exists(self.config.cache_dir):
            return []
        
        # Look for cache files matching this vehicle and date range
        cache_files = []
        for root, dirs, files in os.walk(self.config.cache_dir):
            for file in files:
                if (file.endswith('.json') and 
                    f'vehicle_{vehicle_id}_' in file and
                    f'fleet_{self.config.fleet_id}_' in file):
                    cache_files.append(os.path.join(root, file))
        
        if not cache_files:
            return []
        
        # Load and combine data from all matching cache files
        all_data = []
        for cache_file in cache_files:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    elif isinstance(data, dict) and 'events' in data:
                        all_data.extend(data['events'])
                    elif isinstance(data, dict) and 'history' in data:
                        all_data.extend(data['history'])
            except Exception as e:
                self.logger.error(f"Error loading cache file {cache_file}: {str(e)}")
                continue
        
        return all_data
    
    def _detect_shifts_from_history(self, history: List[Dict[str, Any]], vehicle_id: int, vehicle_ref: str) -> List[ShiftData]:
        """Detect shifts from vehicle history based on driver changes."""
        if not history:
            return []
        
        shifts = []
        current_shift_events = []
        current_driver_id = None
        shift_counter = 1
        
        for event in history:
            event_time = self._parse_timestamp(event.get('deviceTimeUtc'))
            if not event_time:
                continue
            
            # Check if this event is within our date range
            if not (self.config.start_date <= event_time <= self.config.end_date):
                continue
            
            driver_id = event.get('driverId')
            
            # Handle driver change or large time gap
            if self._should_start_new_shift(current_driver_id, driver_id, current_shift_events, event):
                # Finalize current shift if it exists and is valid
                if current_shift_events:
                    shift = self._create_shift_from_events(
                        current_shift_events, vehicle_id, vehicle_ref, 
                        current_driver_id, shift_counter
                    )
                    if shift:
                        shifts.append(shift)
                        shift_counter += 1
                
                # Start new shift
                current_shift_events = [event]
                current_driver_id = driver_id
            else:
                # Continue current shift
                current_shift_events.append(event)
        
        # Handle final shift
        if current_shift_events:
            shift = self._create_shift_from_events(
                current_shift_events, vehicle_id, vehicle_ref, 
                current_driver_id, shift_counter
            )
            if shift:
                shifts.append(shift)
        
        return shifts
    
    def _should_start_new_shift(self, current_driver_id: Optional[str], new_driver_id: Optional[str], 
                               current_events: List[Dict[str, Any]], new_event: Dict[str, Any]) -> bool:
        """Determine if a new shift should be started."""
        # No current shift
        if not current_events:
            return True
        
        # Driver change (primary trigger)
        if current_driver_id != new_driver_id:
            return True
        
        # Large time gap (secondary trigger)
        last_event = current_events[-1]
        last_time = self._parse_timestamp(last_event.get('deviceTimeUtc'))
        new_time = self._parse_timestamp(new_event.get('deviceTimeUtc'))
        
        if last_time and new_time:
            time_gap = (new_time - last_time).total_seconds() / 3600  # hours
            if time_gap > self.config.max_event_gap_hours:
                return True
        
        return False
    
    def _create_shift_from_events(self, events: List[Dict[str, Any]], vehicle_id: int, 
                                 vehicle_ref: str, driver_id: Optional[str], shift_counter: int) -> Optional[ShiftData]:
        """Create a ShiftData object from a list of events."""
        # Skip shifts with null driver IDs unless explicitly requested
        if driver_id is None and not self.config.include_null_driver_shifts:
            self.logger.debug(f"Skipping null driver period for vehicle {vehicle_ref} "
                            f"({len(events)} events, likely idle/maintenance period)")
            return None
            
        if len(events) < self.config.min_events_per_shift:
            return None
        
        # Sort events by time
        events.sort(key=lambda x: x.get('deviceTimeUtc', ''))
        
        first_event = events[0]
        last_event = events[-1]
        
        start_time = self._parse_timestamp(first_event.get('deviceTimeUtc'))
        end_time = self._parse_timestamp(last_event.get('deviceTimeUtc'))
        
        if not start_time or not end_time:
            return None
        
        # Check minimum duration
        duration_minutes = (end_time - start_time).total_seconds() / 60
        if duration_minutes < self.config.min_shift_duration_minutes:
            return None
        
        # Calculate metrics
        duration_hours = duration_minutes / 60
        distance_km = self._calculate_distance(events)
        max_speed = max((e.get('speedKmh', 0) or 0 for e in events), default=0)
        
        # Get fuel and engine data
        total_fuel_used = self._calculate_fuel_used(events)
        engine_hours = self._calculate_engine_hours(events)
        
        # Convert times to local timezone
        local_start = self.timezone.normalize(start_time.replace(tzinfo=pytz.UTC).astimezone(self.timezone))
        local_end = self.timezone.normalize(end_time.replace(tzinfo=pytz.UTC).astimezone(self.timezone))
        
        # Generate shift ID based on driver status
        shift_type_prefix = "IDLE" if driver_id is None else "SHIFT"
        shift_id = f"{vehicle_ref}_{shift_type_prefix}_{local_start.strftime('%Y%m%d_%H%M%S')}_{shift_counter:03d}"
        
        return ShiftData(
            shift_id=shift_id,
            vehicle_id=vehicle_id,
            vehicle_ref=vehicle_ref,
            driver_id=driver_id,
            driver_name=first_event.get('driver'),
            driver_ref=first_event.get('driverRef'),
            shift_date=local_start.strftime('%Y-%m-%d'),
            start_time=local_start.strftime('%Y-%m-%d %H:%M:%S %Z'),
            end_time=local_end.strftime('%Y-%m-%d %H:%M:%S %Z'),
            duration_hours=round(duration_hours, 2),
            total_events=len(events),
            distance_km=round(distance_km, 2),
            start_location_lat=first_event.get('latitude', 0),
            start_location_lon=first_event.get('longitude', 0),
            end_location_lat=last_event.get('latitude', 0),
            end_location_lon=last_event.get('longitude', 0),
            start_place=first_event.get('placeName', ''),
            end_place=last_event.get('placeName', ''),
            max_speed_kmh=round(max_speed, 2),
            total_fuel_used=total_fuel_used,
            engine_hours=engine_hours,
            shift_type="driver_change" if driver_id is not None else "idle_period"
        )
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse timestamp string to datetime object."""
        if not timestamp_str:
            return None
        
        try:
            # Handle various timestamp formats
            for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S']:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue
            return None
        except Exception:
            return None
    
    def _calculate_distance(self, events: List[Dict[str, Any]]) -> float:
        """Calculate distance traveled from odometer readings."""
        odometer_readings = [e.get('odometer') for e in events if e.get('odometer')]
        if len(odometer_readings) < 2:
            return 0.0
        
        return max(odometer_readings) - min(odometer_readings)
    
    def _calculate_fuel_used(self, events: List[Dict[str, Any]]) -> Optional[float]:
        """Calculate total fuel used during the shift."""
        fuel_readings = []
        for event in events:
            engine_data = event.get('engineData', {})
            if engine_data and engine_data.get('totalFuelUsed'):
                fuel_readings.append(engine_data['totalFuelUsed'])
        
        if len(fuel_readings) < 2:
            return None
        
        return round(max(fuel_readings) - min(fuel_readings), 2)
    
    def _calculate_engine_hours(self, events: List[Dict[str, Any]]) -> Optional[float]:
        """Calculate engine hours during the shift."""
        engine_hours = []
        for event in events:
            engine_data = event.get('engineData', {})
            if engine_data and engine_data.get('totalEngineHours'):
                engine_hours.append(engine_data['totalEngineHours'])
        
        if len(engine_hours) < 2:
            return None
        
        return round(max(engine_hours) - min(engine_hours), 2)
    
    def _generate_summary_stats(self, shifts: List[ShiftData], processed_vehicles: int) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not shifts:
            return {
                "total_shifts": 0,
                "total_vehicles": processed_vehicles,
                "date_range": f"{self.config.start_date} to {self.config.end_date}",
                "unique_drivers": 0,
                "average_shift_duration_hours": 0,
                "total_shift_hours": 0
            }
        
        total_hours = sum(s.duration_hours for s in shifts)
        unique_drivers = len(set(s.driver_id for s in shifts if s.driver_id))
        
        return {
            "total_shifts": len(shifts),
            "total_vehicles": processed_vehicles,
            "date_range": f"{self.config.start_date} to {self.config.end_date}",
            "unique_drivers": unique_drivers,
            "average_shift_duration_hours": round(total_hours / len(shifts), 2),
            "total_shift_hours": round(total_hours, 2)
        }


class DriverShiftReporter:
    """Handles report generation for driver shifts."""
    
    def __init__(self, config: ShiftDetectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def _generate_filename_context(self, base_name: str, extension: str) -> str:
        """Generate a contextual filename with timestamp."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Add context based on configuration
        context_parts = []
        if self.config.vehicle_ref:
            context_parts.append(f"vehicle_{self.config.vehicle_ref}")
        context_parts.append(f"fleet_{self.config.fleet_id}")
        
        # Add analysis date range
        date_range = f"{self.config.start_date.strftime('%Y%m%d')}_to_{self.config.end_date.strftime('%Y%m%d')}"
        context_parts.append(date_range)
        
        # Add null driver inclusion status
        if self.config.include_null_driver_shifts:
            context_parts.append("with_idle")
        else:
            context_parts.append("active_only")
        
        context = "_".join(context_parts)
        filename = f"{base_name}_{context}_{timestamp}.{extension}"
        
        return os.path.join(self.config.output_dir, filename)
    
    def generate_reports(self, shifts: List[ShiftData], summary_stats: Dict[str, Any]) -> None:
        """Generate all reports."""
        ensure_dir(self.config.output_dir)
        
        # Console summary
        self._print_console_summary(summary_stats)
        
        # CSV report
        csv_path = self._generate_csv_report(shifts)
        self.logger.info(f"CSV report saved to: {csv_path}")
        
        # JSON summary
        json_path = self._generate_json_summary(summary_stats, shifts)
        self.logger.info(f"JSON summary saved to: {json_path}")
    
    def _print_console_summary(self, summary_stats: Dict[str, Any]) -> None:
        """Print summary to console."""
        print("\n" + "="*60)
        print("DRIVER SHIFT DETECTION SUMMARY")
        print("="*60)
        print(f"Total shifts detected: {summary_stats['total_shifts']}")
        print(f"Total vehicles analyzed: {summary_stats['total_vehicles']}")
        print(f"Date range: {summary_stats['date_range']}")
        print(f"Number of unique drivers: {summary_stats['unique_drivers']}")
        print(f"Average shift duration: {summary_stats['average_shift_duration_hours']} hours")
        print(f"Total shift hours: {summary_stats['total_shift_hours']} hours")
        print("="*60)
    
    def _generate_csv_report(self, shifts: List[ShiftData]) -> str:
        """Generate detailed CSV report with timestamp and context."""
        csv_path = self._generate_filename_context("driver_shifts_detailed", "csv")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            if not shifts:
                csvfile.write("No shifts detected\n")
                return csv_path
            
            fieldnames = [
                'shift_id', 'vehicle_id', 'vehicle_ref', 'driver_id', 'driver_name', 'driver_ref',
                'shift_date', 'start_time', 'end_time', 'duration_hours', 'total_events',
                'distance_km', 'start_location_lat', 'start_location_lon', 'end_location_lat',
                'end_location_lon', 'start_place', 'end_place', 'max_speed_kmh',
                'total_fuel_used', 'engine_hours', 'shift_type'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for shift in shifts:
                writer.writerow(asdict(shift))
        
        return csv_path
    
    def _generate_json_summary(self, summary_stats: Dict[str, Any], shifts: List[ShiftData]) -> str:
        """Generate JSON summary file with timestamp and context."""
        json_path = self._generate_filename_context("driver_shifts_summary", "json")
        
        # Group shifts by vehicle
        vehicle_summaries = {}
        for shift in shifts:
            vehicle_ref = shift.vehicle_ref
            if vehicle_ref not in vehicle_summaries:
                vehicle_summaries[vehicle_ref] = []
            vehicle_summaries[vehicle_ref].append(asdict(shift))
        
        summary_data = {
            "analysis_metadata": {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "timezone": self.config.timezone,
                "fleet_id": self.config.fleet_id,
                "date_range": {
                    "start": self.config.start_date.isoformat(),
                    "end": self.config.end_date.isoformat()
                },
                "configuration": {
                    "min_shift_duration_minutes": self.config.min_shift_duration_minutes,
                    "max_event_gap_hours": self.config.max_event_gap_hours,
                    "min_events_per_shift": self.config.min_events_per_shift,
                    "vehicle_filter": self.config.vehicle_ref
                }
            },
            "summary_statistics": summary_stats,
            "vehicle_summaries": vehicle_summaries
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        return json_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Detect driver shifts based on driver ID changes")
    
    parser.add_argument("--fleet-id", type=int, required=True, help="Fleet ID to analyze")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--timezone", default="Australia/Perth", help="Timezone for date handling")
    parser.add_argument("--output-dir", default="./driver_shift_analysis", help="Output directory")
    parser.add_argument("--cache-dir", help="Cache directory to use")
    parser.add_argument("--vehicle-ref", help="Filter to specific vehicle reference")
    parser.add_argument("--min-shift-duration", type=int, default=30, help="Minimum shift duration in minutes")
    parser.add_argument("--max-event-gap", type=int, default=4, help="Maximum event gap in hours")
    parser.add_argument("--min-events", type=int, default=5, help="Minimum events per shift")
    parser.add_argument("--include-null-drivers", action="store_true", 
                       help="Include periods with null driver IDs (idle/maintenance periods)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Parse dates
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        
        # Create configuration
        config = ShiftDetectionConfig(
            fleet_id=args.fleet_id,
            start_date=start_date,
            end_date=end_date,
            timezone=args.timezone,
            min_shift_duration_minutes=args.min_shift_duration,
            max_event_gap_hours=args.max_event_gap,
            min_events_per_shift=args.min_events,
            vehicle_ref=args.vehicle_ref,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            include_null_driver_shifts=args.include_null_drivers
        )
        
        # Run shift detection
        detector = DriverShiftDetector(config)
        shifts, summary_stats = detector.detect_shifts()
        
        # Generate reports
        reporter = DriverShiftReporter(config)
        reporter.generate_reports(shifts, summary_stats)
        
        logger.info("Driver shift detection completed successfully")
        
    except Exception as e:
        logger.error(f"Error during driver shift detection: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()