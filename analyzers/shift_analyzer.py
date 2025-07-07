"""
Shift analysis logic.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import numpy as np

from config import ShiftConfig
from models import Trip, Shift, ShiftAnalysis, TripPrediction
from predictors.trip_predictor import TripPredictor

logger = logging.getLogger(__name__)


class ShiftAnalyzer:
    """Analyzes shifts for completion prediction."""
    
    def __init__(self, config: ShiftConfig):
        self.config = config
        self.predictor = TripPredictor(config)
        
    def analyze_shifts(self, vehicle_id: int, vehicle_name: str, 
                      trips: List[Trip]) -> List[ShiftAnalysis]:
        """
        Analyze all shifts for a vehicle.
        
        Args:
            vehicle_id: Vehicle ID
            vehicle_name: Vehicle display name
            trips: List of trips
            
        Returns:
            List of ShiftAnalysis objects
        """
        if not trips:
            return []
        
        # Group trips into shifts
        shifts = self._group_trips_into_shifts(vehicle_id, trips)
        
        # Analyze each shift
        shift_analyses = []
        for shift in shifts:
            analysis = self._analyze_single_shift(shift)
            shift_analyses.append(analysis)
        
        logger.info(f"Analyzed {len(shift_analyses)} shifts for vehicle {vehicle_name}")
        
        return shift_analyses
    
    def _group_trips_into_shifts(self, vehicle_id: int, trips: List[Trip]) -> List[Shift]:
        """Group trips into shifts based on time windows."""
        if not trips:
            return []
        
        # Sort trips by start time
        sorted_trips = sorted(trips, key=lambda t: t.start_time)
        
        shifts = []
        current_shift_trips = []
        shift_start = None
        shift_count = 0
        
        for trip in sorted_trips:
            if not current_shift_trips:
                # Start new shift
                current_shift_trips = [trip]
                shift_start = trip.start_time
            else:
                # Check if trip belongs to current shift
                hours_since_shift_start = (trip.start_time - shift_start).total_seconds() / 3600
                
                if hours_since_shift_start <= self.config.shift_duration_hours:
                    current_shift_trips.append(trip)
                else:
                    # Create shift and start new one
                    shift_count += 1
                    shift = self._create_shift(
                        vehicle_id, shift_start, current_shift_trips, shift_count
                    )
                    shifts.append(shift)
                    
                    current_shift_trips = [trip]
                    shift_start = trip.start_time
        
        # Add last shift
        if current_shift_trips:
            shift_count += 1
            shift = self._create_shift(
                vehicle_id, shift_start, current_shift_trips, shift_count
            )
            shifts.append(shift)
        
        return shifts
    
    def _create_shift(self, vehicle_id: int, start_time: datetime,
                     trips: List[Trip], shift_number: int) -> Shift:
        """Create a Shift object."""
        end_time = start_time + timedelta(hours=self.config.shift_duration_hours)
        driver_id = trips[0].driver_id if trips else None
        
        return Shift(
            shift_id=f"{vehicle_id}_{start_time.strftime('%Y%m%d')}_{shift_number}",
            vehicle_id=vehicle_id,
            driver_id=driver_id,
            start_time=start_time,
            end_time=end_time,
            trips=trips
        )
    
    def _analyze_single_shift(self, shift: Shift) -> ShiftAnalysis:
        """Analyze a single shift for completion prediction."""
        trips_completed = shift.trips_completed
        trips_remaining = max(0, self.config.target_trips - trips_completed)
        
        # Get current status
        if trips_remaining == 0:
            return ShiftAnalysis(
                shift=shift,
                predictions=[],
                can_complete_target=True,
                risk_level='none',
                alert_required=False,
                recommendation='Target achieved',
                metrics={
                    'trips_completed': trips_completed,
                    'trips_target': self.config.target_trips,
                    'completion_rate': 1.0
                }
            )
        
        # Get last trip end time
        last_trip_end = shift.trips[-1].end_time if shift.trips else shift.start_time
        
        # Calculate remaining time
        remaining_time = (shift.end_time - last_trip_end).total_seconds() / 60
        
        # Predict remaining trips
        predictions = []
        current_time = last_trip_end
        
        for i in range(trips_remaining):
            prediction = self.predictor.predict_next_trip(
                shift, current_time, trips_completed + i
            )
            predictions.append(prediction)
            current_time = prediction.estimated_completion_time
        
        # Determine if target can be completed
        total_estimated_time = sum(p.estimated_duration_minutes for p in predictions)
        buffer_adjusted_time = remaining_time - self.config.buffer_time_minutes
        can_complete = total_estimated_time <= buffer_adjusted_time
        
        # Determine risk level
        risk_level = self._calculate_risk_level(
            total_estimated_time, buffer_adjusted_time
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            trips_remaining, can_complete, risk_level
        )
        
        return ShiftAnalysis(
            shift=shift,
            predictions=predictions,
            can_complete_target=can_complete,
            risk_level=risk_level,
            alert_required=risk_level in ['high', 'critical'],
            recommendation=recommendation,
            metrics={
                'trips_completed': trips_completed,
                'trips_target': self.config.target_trips,
                'trips_remaining': trips_remaining,
                'avg_trip_duration': np.mean([t.duration_minutes for t in shift.trips]) if shift.trips else 0,
                'total_estimated_time': total_estimated_time,
                'remaining_time': remaining_time,
                'buffer_adjusted_time': buffer_adjusted_time,
                'completion_probability': 1.0 - (total_estimated_time / buffer_adjusted_time) if buffer_adjusted_time > 0 else 0
            }
        )
    
    def _calculate_risk_level(self, estimated_time: float, available_time: float) -> str:
        """Calculate risk level based on time estimates."""
        if available_time <= 0:
            return 'critical'
        
        time_ratio = estimated_time / available_time
        
        if time_ratio <= 0.7:
            return 'low'
        elif time_ratio <= 0.9:
            return 'medium'
        elif time_ratio <= 1.0:
            return 'high'
        else:
            return 'critical'
    
    def _generate_recommendation(self, trips_remaining: int, 
                               can_complete: bool, risk_level: str) -> str:
        """Generate actionable recommendation."""
        if can_complete and risk_level == 'low':
            return f"Continue with remaining {trips_remaining} trips. Ample time available."
        
        elif can_complete and risk_level == 'medium':
            return f"Continue with remaining {trips_remaining} trips but maintain steady pace."
        
        elif risk_level in ['high', 'critical']:
            if trips_remaining == 1:
                return "DO NOT START final trip. High risk of exceeding shift duration."
            else:
                return f"HIGH RISK: Consider completing only {trips_remaining-1} trips."
        
        return "Monitor closely. Situation requires attention."