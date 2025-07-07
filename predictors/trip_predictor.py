"""
Trip completion prediction logic with timezone support.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pytz

import numpy as np

from config import ShiftConfig
from models import Shift, Trip, TripPrediction

logger = logging.getLogger(__name__)


class TripPredictor:
    """Predicts trip completion times and probabilities with timezone awareness."""
    
    def __init__(self, config: ShiftConfig):
        self.config = config
        self.historical_data = []
        
    def predict_next_trip(self, shift: Shift, current_time: datetime,
                         trips_already_completed: int) -> TripPrediction:
        """
        Predict the next trip duration and completion time.
        
        Args:
            shift: Current shift
            current_time: Current time (should be timezone-aware)
            trips_already_completed: Number of trips already completed
            
        Returns:
            TripPrediction object
        """
        # Ensure current_time is timezone-aware
        if current_time.tzinfo is None:
            current_time = pytz.UTC.localize(current_time)
        
        # Extract prediction factors
        factors = self._extract_prediction_factors(
            shift, current_time, trips_already_completed
        )
        
        # Base prediction on historical average if available
        if shift.trips:
            base_duration = np.mean([t.duration_minutes for t in shift.trips])
            confidence = 0.8
        else:
            # Default estimate
            base_duration = 45.0  # Default 45 minutes
            confidence = 0.5
        
        # Apply modifiers
        adjusted_duration = self._apply_prediction_modifiers(
            base_duration, factors
        )
        
        # Calculate completion time (maintains timezone)
        completion_time = current_time + timedelta(minutes=adjusted_duration)
        
        # Check if completion is within shift
        will_complete = completion_time <= shift.end_time
        
        # Determine risk level
        time_to_shift_end = (shift.end_time - completion_time).total_seconds() / 60
        
        if time_to_shift_end > self.config.buffer_time_minutes:
            risk_level = 'low'
        elif time_to_shift_end > 0:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        return TripPrediction(
            estimated_duration_minutes=adjusted_duration,
            confidence_score=confidence,
            estimated_completion_time=completion_time,
            will_complete_in_shift=will_complete,
            risk_level=risk_level,
            recommendation=self._generate_trip_recommendation(
                will_complete, risk_level, time_to_shift_end
            ),
            factors=factors
        )
    
    def _extract_prediction_factors(self, shift: Shift, current_time: datetime,
                                  trips_completed: int) -> Dict[str, Any]:
        """Extract factors affecting trip duration."""
        # Time-based factors (use local hour if timezone info available)
        if hasattr(current_time, 'hour'):
            hour_of_day = current_time.hour
        else:
            hour_of_day = 12  # Default to noon
            
        is_peak_hour = 6 <= hour_of_day <= 9 or 16 <= hour_of_day <= 19
        
        # Shift progress factors
        shift_elapsed_hours = (current_time - shift.start_time).total_seconds() / 3600
        shift_progress_ratio = shift_elapsed_hours / self.config.shift_duration_hours
        
        # Historical factors
        avg_duration = np.mean([t.duration_minutes for t in shift.trips]) if shift.trips else 0
        duration_variance = np.std([t.duration_minutes for t in shift.trips]) if len(shift.trips) > 1 else 0
        
        return {
            'hour_of_day': hour_of_day,
            'is_peak_hour': is_peak_hour,
            'trips_completed': trips_completed,
            'shift_elapsed_hours': shift_elapsed_hours,
            'shift_progress_ratio': shift_progress_ratio,
            'avg_trip_duration': avg_duration,
            'duration_variance': duration_variance,
            'fatigue_factor': self._calculate_fatigue_factor(trips_completed)
        }
    
    def _apply_prediction_modifiers(self, base_duration: float,
                                  factors: Dict[str, Any]) -> float:
        """Apply modifiers to base duration prediction."""
        modified_duration = base_duration
        
        # Peak hour modifier
        if factors['is_peak_hour']:
            modified_duration *= 1.15
        
        # Fatigue modifier
        modified_duration *= factors['fatigue_factor']
        
        # Variance modifier (add uncertainty)
        if factors['duration_variance'] > 0:
            modified_duration += factors['duration_variance'] * 0.5
        
        return modified_duration
    
    def _calculate_fatigue_factor(self, trips_completed: int) -> float:
        """Calculate fatigue factor based on trips completed."""
        # Progressive fatigue model
        base_factor = 1.0
        
        if trips_completed == 0:
            return base_factor
        elif trips_completed <= 2:
            return base_factor + (trips_completed * self.config.fatigue_factor_base)
        else:
            # Accelerating fatigue after 2 trips
            return base_factor + (2 * self.config.fatigue_factor_base) + \
                   ((trips_completed - 2) * self.config.fatigue_factor_base * 1.5)
    
    def _generate_trip_recommendation(self, will_complete: bool,
                                    risk_level: str, time_buffer: float) -> str:
        """Generate recommendation for the trip."""
        if will_complete and risk_level == 'low':
            return "Safe to proceed with trip"
        elif will_complete and risk_level == 'medium':
            return f"Proceed with caution. Only {time_buffer:.0f} minutes buffer remaining"
        else:
            return "High risk. Consider skipping this trip"