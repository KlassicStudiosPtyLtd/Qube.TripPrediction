#!/usr/bin/env python3
"""
Improved Driver Shift Detection System

Enhanced version that better handles null driver IDs and distinguishes
between active work shifts and idle/maintenance periods.
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
class ImprovedShiftDetectionConfig:
    """Enhanced configuration for driver shift detection."""
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
    
    # New parameters for improved detection
    include_null_driver_shifts: bool = False  # Whether to include null driver periods as shifts
    min_active_shift_duration_minutes: int = 60  # Minimum duration for active work shifts
    require_movement_for_shift: bool = True  # Require vehicle movement to consider it a work shift
    min_shift_distance_km: float = 1.0  # Minimum distance traveled to be considered active shift
    separate_idle_periods: bool = True  # Create separate output for idle periods


@dataclass 
class ShiftData:
    """Enhanced shift data class."""
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
    shift_category: str = "active"  # "active", "idle", "maintenance"
    has_movement: bool = True
    primary_location: str = ""


class ImprovedDriverShiftDetector:
    """Enhanced driver shift detector with better null driver handling."""
    
    def __init__(self, config: ImprovedShiftDetectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.api_client = None
        self.timezone = pytz.timezone(config.timezone)
        
        # Initialize API client if needed
        if not config.cache_dir or not os.path.exists(config.cache_dir):
            self.api_client = MTDataApiClient(self.logger)
    
    def detect_shifts(self) -> Tuple[List[ShiftData], List[ShiftData], Dict[str, Any]]:
        """
        Enhanced shift detection that separates active shifts from idle periods.
        
        Returns:
            Tuple of (active_shifts, idle_periods, summary_stats)
        """
        self.logger.info(f"Starting enhanced driver shift detection for fleet {self.config.fleet_id}")
        
        # Get vehicles to analyze
        vehicles = self._get_vehicles_to_analyze()
        if not vehicles:
            self.logger.warning("No vehicles found to analyze")
            return [], [], {}
        
        self.logger.info(f"Found {len(vehicles)} vehicles to analyze")
        
        # Process each vehicle
        all_active_shifts = []
        all_idle_periods = []
        processed_vehicles = 0
        
        for vehicle in vehicles:
            try:
                active_shifts, idle_periods = self._process_vehicle_enhanced(vehicle)
                all_active_shifts.extend(active_shifts)
                all_idle_periods.extend(idle_periods)
                processed_vehicles += 1
                
                self.logger.info(f"Processed vehicle {vehicle['vehicleRef']}: "
                               f"{len(active_shifts)} active shifts, {len(idle_periods)} idle periods")
                
            except Exception as e:
                self.logger.error(f"Error processing vehicle {vehicle.get('vehicleRef', 'unknown')}: {str(e)}")
                continue
        
        # Generate summary statistics
        summary_stats = self._generate_enhanced_summary_stats(
            all_active_shifts, all_idle_periods, processed_vehicles
        )
        
        self.logger.info(f"Completed enhanced shift detection: "
                        f"{len(all_active_shifts)} active shifts, "
                        f"{len(all_idle_periods)} idle periods")
        
        return all_active_shifts, all_idle_periods, summary_stats
    
    def _process_vehicle_enhanced(self, vehicle: Dict[str, Any]) -> Tuple[List[ShiftData], List[ShiftData]]:
        """Process a single vehicle with enhanced logic."""
        vehicle_id = vehicle['vehicleId']
        vehicle_ref = vehicle.get('vehicleRef', f'VEH_{vehicle_id}')
        
        # Load vehicle history data (reuse existing method)
        history_data = self._load_vehicle_history(vehicle_id, vehicle_ref)
        if not history_data:
            self.logger.warning(f"No history data found for vehicle {vehicle_ref}")
            return [], []
        
        # Sort by timestamp
        history_data.sort(key=lambda x: x.get('deviceTimeUtc', ''))
        
        # Enhanced shift detection
        active_shifts, idle_periods = self._detect_shifts_enhanced(history_data, vehicle_id, vehicle_ref)
        
        return active_shifts, idle_periods
    
    def _detect_shifts_enhanced(self, history: List[Dict[str, Any]], vehicle_id: int, 
                               vehicle_ref: str) -> Tuple[List[ShiftData], List[ShiftData]]:
        """Enhanced shift detection that separates active work from idle periods."""
        if not history:
            return [], []
        
        # First, group events by driver sessions
        driver_sessions = self._group_events_by_driver_sessions(history)
        
        active_shifts = []
        idle_periods = []
        shift_counter = 1
        idle_counter = 1
        
        for session in driver_sessions:
            events = session['events']
            driver_id = session['driver_id']
            
            # Analyze the session to determine if it's active work or idle period
            session_analysis = self._analyze_session(events)
            
            # Create shift data
            if driver_id is not None and session_analysis['is_active_work']:
                # This is an active work shift
                shift = self._create_enhanced_shift(
                    events, vehicle_id, vehicle_ref, driver_id, 
                    shift_counter, "active", session_analysis
                )
                if shift:
                    active_shifts.append(shift)
                    shift_counter += 1
                    
            elif self.config.include_null_driver_shifts or session_analysis['has_significant_activity']:
                # This is an idle period or maintenance activity
                category = "maintenance" if session_analysis['has_significant_activity'] else "idle"
                
                shift = self._create_enhanced_shift(
                    events, vehicle_id, vehicle_ref, driver_id,
                    idle_counter, category, session_analysis
                )
                if shift:
                    idle_periods.append(shift)
                    idle_counter += 1
        
        return active_shifts, idle_periods
    
    def _group_events_by_driver_sessions(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group events into driver sessions, handling logon/logoff properly."""
        sessions = []
        current_session_events = []
        current_driver_id = None
        
        for event in history:
            event_time = self._parse_timestamp(event.get('deviceTimeUtc'))
            if not event_time:
                continue
            
            # Check if this event is within our date range
            if not (self.config.start_date <= event_time <= self.config.end_date):
                continue
            
            driver_id = event.get('driverId')
            reason_code = event.get('reasonCode', '')
            
            # Handle driver logon/logoff events
            if reason_code == 'Logon' and driver_id is not None:
                # Start new driver session
                if current_session_events:
                    sessions.append({
                        'driver_id': current_driver_id,
                        'events': current_session_events.copy(),
                        'start_time': current_session_events[0].get('deviceTimeUtc'),
                        'end_time': current_session_events[-1].get('deviceTimeUtc')
                    })
                
                current_session_events = [event]
                current_driver_id = driver_id
                
            elif reason_code == 'Logoff':
                # End current session
                if current_session_events:
                    current_session_events.append(event)
                    sessions.append({
                        'driver_id': current_driver_id,
                        'events': current_session_events.copy(),
                        'start_time': current_session_events[0].get('deviceTimeUtc'),
                        'end_time': current_session_events[-1].get('deviceTimeUtc')
                    })
                
                current_session_events = []
                current_driver_id = None
                
            else:
                # Handle driver change or time gap
                if self._should_start_new_session(current_driver_id, driver_id, current_session_events, event):
                    # Finalize current session
                    if current_session_events:
                        sessions.append({
                            'driver_id': current_driver_id,
                            'events': current_session_events.copy(),
                            'start_time': current_session_events[0].get('deviceTimeUtc'),
                            'end_time': current_session_events[-1].get('deviceTimeUtc')
                        })
                    
                    # Start new session
                    current_session_events = [event]
                    current_driver_id = driver_id
                else:
                    # Continue current session
                    current_session_events.append(event)
        
        # Handle final session
        if current_session_events:
            sessions.append({
                'driver_id': current_driver_id,
                'events': current_session_events.copy(),
                'start_time': current_session_events[0].get('deviceTimeUtc'),
                'end_time': current_session_events[-1].get('deviceTimeUtc')
            })
        
        return sessions
    
    def _should_start_new_session(self, current_driver_id: Optional[str], new_driver_id: Optional[str],
                                 current_events: List[Dict[str, Any]], new_event: Dict[str, Any]) -> bool:
        """Determine if a new session should be started."""
        # No current session
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
    
    def _analyze_session(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a session to determine its characteristics."""
        if not events:
            return {
                'is_active_work': False,
                'has_significant_activity': False,
                'has_movement': False,
                'total_distance': 0.0,
                'max_speed': 0.0,
                'primary_location': '',
                'activity_type': 'idle'
            }
        
        # Calculate metrics
        distances = [e.get('odometer', 0) for e in events if e.get('odometer')]
        total_distance = max(distances) - min(distances) if len(distances) >= 2 else 0.0
        
        speeds = [e.get('speedKmh', 0) or 0 for e in events]
        max_speed = max(speeds)
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        
        # Get primary location
        locations = [e.get('placeName', '') for e in events if e.get('placeName')]
        primary_location = max(set(locations), key=locations.count) if locations else ''
        
        # Determine activity characteristics
        has_movement = total_distance >= self.config.min_shift_distance_km or max_speed > 5.0
        has_significant_activity = (
            has_movement or 
            len(events) > 50 or  # Many events suggest activity
            any(e.get('reasonCode') in ['DepWayPoint', 'ArrWayPoint'] for e in events)  # Waypoint activity
        )
        
        # Determine if this is active work
        session_duration = 0
        if len(events) >= 2:
            start_time = self._parse_timestamp(events[0].get('deviceTimeUtc'))
            end_time = self._parse_timestamp(events[-1].get('deviceTimeUtc'))
            if start_time and end_time:
                session_duration = (end_time - start_time).total_seconds() / 60  # minutes
        
        is_active_work = (
            has_movement and
            session_duration >= self.config.min_active_shift_duration_minutes and
            len(events) >= self.config.min_events_per_shift
        )
        
        # Determine activity type
        if is_active_work:
            activity_type = 'active_work'
        elif has_significant_activity:
            activity_type = 'maintenance' if 'laydown' not in primary_location.lower() else 'preparation'
        else:
            activity_type = 'idle'
        
        return {
            'is_active_work': is_active_work,
            'has_significant_activity': has_significant_activity,
            'has_movement': has_movement,
            'total_distance': total_distance,
            'max_speed': max_speed,
            'avg_speed': avg_speed,
            'primary_location': primary_location,
            'activity_type': activity_type,
            'session_duration_minutes': session_duration
        }
    
    def _create_enhanced_shift(self, events: List[Dict[str, Any]], vehicle_id: int,
                              vehicle_ref: str, driver_id: Optional[str], counter: int,
                              category: str, analysis: Dict[str, Any]) -> Optional[ShiftData]:
        """Create enhanced shift data with additional categorization."""
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
        
        duration_minutes = (end_time - start_time).total_seconds() / 60
        
        # Apply different minimum duration based on category
        min_duration = (self.config.min_active_shift_duration_minutes if category == 'active' 
                       else self.config.min_shift_duration_minutes)
        
        if duration_minutes < min_duration:
            return None
        
        # Calculate metrics (reuse existing methods)
        duration_hours = duration_minutes / 60
        distance_km = analysis['total_distance']
        max_speed = analysis['max_speed']
        
        # Get fuel and engine data
        total_fuel_used = self._calculate_fuel_used(events)
        engine_hours = self._calculate_engine_hours(events)
        
        # Convert times to local timezone
        local_start = self.timezone.normalize(start_time.replace(tzinfo=pytz.UTC).astimezone(self.timezone))
        local_end = self.timezone.normalize(end_time.replace(tzinfo=pytz.UTC).astimezone(self.timezone))
        
        # Generate appropriate shift ID
        prefix = "SHIFT" if category == 'active' else category.upper()
        shift_id = f"{vehicle_ref}_{prefix}_{local_start.strftime('%Y%m%d_%H%M%S')}_{counter:03d}"
        
        return ShiftData(
            shift_id=shift_id,
            vehicle_id=vehicle_id,
            vehicle_ref=vehicle_ref,
            driver_id=str(driver_id) if driver_id is not None else None,
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
            shift_type="driver_session",
            shift_category=category,
            has_movement=analysis['has_movement'],
            primary_location=analysis['primary_location']
        )
    
    # Reuse utility methods from original implementation
    def _get_vehicles_to_analyze(self):
        """Reuse from original implementation."""
        # Implementation would be same as original
        pass
    
    def _load_vehicle_history(self, vehicle_id: int, vehicle_ref: str):
        """Reuse from original implementation.""" 
        # Implementation would be same as original
        pass
    
    def _parse_timestamp(self, timestamp_str: str):
        """Reuse from original implementation."""
        # Implementation would be same as original
        pass
    
    def _calculate_fuel_used(self, events: List[Dict[str, Any]]):
        """Reuse from original implementation."""
        # Implementation would be same as original
        pass
    
    def _calculate_engine_hours(self, events: List[Dict[str, Any]]):
        """Reuse from original implementation."""
        # Implementation would be same as original
        pass
    
    def _generate_enhanced_summary_stats(self, active_shifts: List[ShiftData], 
                                        idle_periods: List[ShiftData], processed_vehicles: int) -> Dict[str, Any]:
        """Generate enhanced summary statistics."""
        active_hours = sum(s.duration_hours for s in active_shifts)
        idle_hours = sum(s.duration_hours for s in idle_periods)
        
        unique_active_drivers = len(set(s.driver_id for s in active_shifts if s.driver_id))
        
        return {
            "total_active_shifts": len(active_shifts),
            "total_idle_periods": len(idle_periods),
            "total_vehicles": processed_vehicles,
            "date_range": f"{self.config.start_date} to {self.config.end_date}",
            "unique_active_drivers": unique_active_drivers,
            "average_active_shift_duration_hours": round(active_hours / len(active_shifts), 2) if active_shifts else 0,
            "average_idle_period_duration_hours": round(idle_hours / len(idle_periods), 2) if idle_periods else 0,
            "total_active_shift_hours": round(active_hours, 2),
            "total_idle_hours": round(idle_hours, 2),
            "vehicle_utilization_ratio": round(active_hours / (active_hours + idle_hours), 2) if (active_hours + idle_hours) > 0 else 0
        }


# Main function would be similar but call the enhanced detector
def main():
    """Enhanced main entry point."""
    # Similar argument parsing but with new options
    parser = argparse.ArgumentParser(description="Enhanced driver shift detection with better null driver handling")
    
    # Add existing arguments plus new ones
    parser.add_argument("--fleet-id", type=int, required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--include-idle-periods", action="store_true", 
                       help="Include idle periods in output")
    parser.add_argument("--min-active-duration", type=int, default=60,
                       help="Minimum duration for active work shifts (minutes)")
    parser.add_argument("--require-movement", action="store_true", default=True,
                       help="Require vehicle movement for active shifts")
    
    # ... rest of argument parsing
    
    # Use enhanced detector
    detector = ImprovedDriverShiftDetector(config)
    active_shifts, idle_periods, summary = detector.detect_shifts()
    
    # Generate enhanced reports
    # ... implementation


if __name__ == "__main__":
    main()