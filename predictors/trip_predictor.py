"""
Trip completion prediction logic with timezone support and detailed calculations.
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
    """Predicts trip completion times and probabilities with detailed calculations."""
    
    def __init__(self, config: ShiftConfig):
        self.config = config
        self.historical_data = []
        
    def predict_next_trip(self, shift: Shift, current_time: datetime,
                         trips_already_completed: int) -> TripPrediction:
        """
        Predict the next trip duration and completion time with detailed factors.
        
        Args:
            shift: Current shift
            current_time: Current time (should be timezone-aware)
            trips_already_completed: Number of trips already completed
            
        Returns:
            TripPrediction object with detailed calculations
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
            calculation_method = 'historical_average'
        else:
            # Use configurable default estimate
            base_duration = self.config.default_trip_duration_minutes
            confidence = 0.5
            calculation_method = 'default_estimate'
        
        # Store base duration in factors
        factors['base_duration'] = base_duration
        factors['calculation_method'] = calculation_method
        
        # Apply modifiers and track each adjustment
        adjusted_duration, adjustment_details = self._apply_prediction_modifiers_detailed(
            base_duration, factors
        )
        
        # Add adjustment details to factors
        factors['adjustments'] = adjustment_details
        factors['final_duration'] = adjusted_duration
        
        # Calculate completion time (maintains timezone)
        completion_time = current_time + timedelta(minutes=adjusted_duration)
        
        # Check if completion is within shift
        will_complete = completion_time <= shift.end_time
        
        # Determine risk level with detailed explanation
        time_to_shift_end = (shift.end_time - completion_time).total_seconds() / 60
        risk_level, risk_details = self._calculate_trip_risk_level(
            time_to_shift_end, adjusted_duration, trips_already_completed
        )
        
        # Add risk details to factors
        factors['time_to_shift_end'] = time_to_shift_end
        factors['risk_details'] = risk_details
        
        return TripPrediction(
            estimated_duration_minutes=adjusted_duration,
            confidence_score=confidence,
            estimated_completion_time=completion_time,
            will_complete_in_shift=will_complete,
            risk_level=risk_level,
            recommendation=self._generate_detailed_trip_recommendation(
                will_complete, risk_level, time_to_shift_end, 
                adjusted_duration, factors
            ),
            factors=factors
        )
    
    def _extract_prediction_factors(self, shift: Shift, current_time: datetime,
                                  trips_completed: int) -> Dict[str, Any]:
        """Extract factors affecting trip duration with detailed calculations."""
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
        if shift.trips:
            durations = [t.duration_minutes for t in shift.trips]
            avg_duration = np.mean(durations)
            duration_variance = np.std(durations) if len(durations) > 1 else 0
            min_duration = np.min(durations)
            max_duration = np.max(durations)
            
            # Calculate trend (are trips getting longer or shorter?)
            if len(durations) >= 2:
                # Simple linear trend
                x = np.arange(len(durations))
                trend_coefficient = np.polyfit(x, durations, 1)[0]
                trend_direction = "increasing" if trend_coefficient > 0 else "decreasing"
            else:
                trend_coefficient = 0
                trend_direction = "insufficient_data"
        else:
            avg_duration = 0
            duration_variance = 0
            min_duration = 0
            max_duration = 0
            trend_coefficient = 0
            trend_direction = "no_data"
        
        # Calculate detailed fatigue factor
        fatigue_factor, fatigue_details = self._calculate_detailed_fatigue_factor(trips_completed)
        
        return {
            'hour_of_day': hour_of_day,
            'is_peak_hour': is_peak_hour,
            'trips_completed': trips_completed,
            'shift_elapsed_hours': shift_elapsed_hours,
            'shift_progress_ratio': shift_progress_ratio,
            'avg_trip_duration': avg_duration,
            'duration_variance': duration_variance,
            'min_duration': min_duration,
            'max_duration': max_duration,
            'trend_coefficient': trend_coefficient,
            'trend_direction': trend_direction,
            'fatigue_factor': fatigue_factor,
            'fatigue_details': fatigue_details,
            'default_duration_used': avg_duration == 0
        }
    
    def _apply_prediction_modifiers_detailed(self, base_duration: float,
                                          factors: Dict[str, Any]) -> tuple:
        """Apply modifiers to base duration prediction with detailed tracking."""
        modified_duration = base_duration
        adjustments = []
        
        # Peak hour modifier
        if factors['is_peak_hour']:
            peak_adjustment = base_duration * 0.15
            modified_duration += peak_adjustment
            adjustments.append({
                'type': 'peak_hour',
                'description': f'Peak hour traffic (hours {6}-{9}, {16}-{19})',
                'adjustment_minutes': peak_adjustment,
                'multiplier': 1.15
            })
        
        # Fatigue modifier
        fatigue_factor = factors['fatigue_factor']
        if fatigue_factor > 1.0:
            fatigue_adjustment = base_duration * (fatigue_factor - 1.0)
            modified_duration = base_duration * fatigue_factor  # Apply as multiplier
            adjustments.append({
                'type': 'fatigue',
                'description': factors['fatigue_details'],
                'adjustment_minutes': fatigue_adjustment,
                'multiplier': fatigue_factor
            })
        
        # Variance modifier (add uncertainty)
        if factors['duration_variance'] > 0:
            # Add half the standard deviation as safety margin
            variance_adjustment = factors['duration_variance'] * 0.5
            modified_duration += variance_adjustment
            adjustments.append({
                'type': 'variance',
                'description': f'Historical variance safety margin (±{factors["duration_variance"]:.1f} min std dev)',
                'adjustment_minutes': variance_adjustment,
                'multiplier': 1.0 + (variance_adjustment / base_duration)
            })
        
        # Trend adjustment
        if factors['trend_coefficient'] != 0 and factors['trips_completed'] > 0:
            # Project the trend forward
            trend_adjustment = factors['trend_coefficient'] * factors['trips_completed']
            if abs(trend_adjustment) > base_duration * 0.2:  # Cap at 20% adjustment
                trend_adjustment = base_duration * 0.2 * np.sign(trend_adjustment)
            
            modified_duration += trend_adjustment
            adjustments.append({
                'type': 'trend',
                'description': f'Trip duration trend ({factors["trend_direction"]})',
                'adjustment_minutes': trend_adjustment,
                'multiplier': 1.0 + (trend_adjustment / base_duration)
            })
        
        # Calculate total adjustment
        total_adjustment = modified_duration - base_duration
        total_multiplier = modified_duration / base_duration if base_duration > 0 else 1.0
        
        # Add summary
        adjustments.append({
            'type': 'total',
            'description': 'Combined adjustments',
            'adjustment_minutes': total_adjustment,
            'multiplier': total_multiplier
        })
        
        return modified_duration, adjustments
    
    def _calculate_detailed_fatigue_factor(self, trips_completed: int) -> tuple:
        """Calculate fatigue factor with detailed explanation."""
        # Progressive fatigue model
        base_factor = 1.0
        
        if trips_completed == 0:
            return base_factor, "No fatigue (first trip)"
        elif trips_completed <= 2:
            factor = base_factor + (trips_completed * self.config.fatigue_factor_base)
            percentage = (factor - 1.0) * 100
            return factor, f"Early shift fatigue: +{percentage:.0f}% (trip {trips_completed + 1})"
        else:
            # Accelerating fatigue after 2 trips
            linear_component = 2 * self.config.fatigue_factor_base
            accelerated_component = (trips_completed - 2) * self.config.fatigue_factor_base * 1.5
            factor = base_factor + linear_component + accelerated_component
            percentage = (factor - 1.0) * 100
            return factor, f"Accumulated fatigue: +{percentage:.0f}% (trip {trips_completed + 1}, accelerated after trip 2)"
    
    def _calculate_trip_risk_level(self, time_buffer: float, duration: float, 
                                  trips_completed: int) -> tuple:
        """Calculate risk level for individual trip with explanation."""
        if time_buffer < 0:
            return 'critical', f"Trip would exceed shift by {abs(time_buffer):.0f} minutes"
        
        # Calculate buffer ratio
        buffer_ratio = time_buffer / duration if duration > 0 else 0
        
        # Determine risk level
        if buffer_ratio > 2.0:
            risk = 'low'
            explanation = f"Comfortable buffer: {time_buffer:.0f} min after trip ({buffer_ratio:.1f}x trip duration)"
        elif buffer_ratio > 1.0:
            risk = 'medium'
            explanation = f"Moderate buffer: {time_buffer:.0f} min after trip ({buffer_ratio:.1f}x trip duration)"
        elif buffer_ratio > 0.5:
            risk = 'high'
            explanation = f"Tight buffer: {time_buffer:.0f} min after trip ({buffer_ratio:.1f}x trip duration)"
        else:
            risk = 'critical'
            explanation = f"Minimal buffer: {time_buffer:.0f} min after trip (only {buffer_ratio:.1f}x trip duration)"
        
        return risk, explanation
    
    def _generate_detailed_trip_recommendation(self, will_complete: bool,
                                             risk_level: str, time_buffer: float,
                                             duration: float, factors: Dict[str, Any]) -> str:
        """Generate detailed recommendation for the trip."""
        details = []
        
        # Duration calculation summary
        if factors['calculation_method'] == 'historical_average':
            details.append(f"Based on {factors['trips_completed']} completed trips (avg: {factors['base_duration']:.1f} min)")
        else:
            details.append(f"Using default estimate: {factors['base_duration']:.1f} min")
        
        # Adjustments applied
        adjustments = factors.get('adjustments', [])
        if adjustments:
            details.append("\nAdjustments applied:")
            for adj in adjustments[:-1]:  # Skip 'total' summary
                if adj['adjustment_minutes'] != 0:
                    details.append(f"• {adj['description']}: {adj['adjustment_minutes']:+.1f} min")
        
        # Final duration
        details.append(f"\nEstimated duration: {duration:.1f} min")
        
        # Risk assessment
        details.append(f"\n{factors['risk_details']}")
        
        # Recommendation
        if will_complete and risk_level == 'low':
            action = "Safe to proceed with trip"
        elif will_complete and risk_level == 'medium':
            action = f"Proceed with caution. Monitor progress closely"
        elif risk_level == 'high':
            action = "High risk - consider if trip is essential"
        else:
            action = "Do not start - will likely exceed shift time"
        
        details.append(f"\nRecommendation: {action}")
        
        return "\n".join(details)
    
    def _generate_trip_recommendation(self, will_complete: bool,
                                    risk_level: str, time_buffer: float) -> str:
        """Generate simple recommendation for the trip (backward compatibility)."""
        if will_complete and risk_level == 'low':
            return "Safe to proceed with trip"
        elif will_complete and risk_level == 'medium':
            return f"Proceed with caution. Only {time_buffer:.0f} minutes buffer remaining"
        else:
            return "High risk. Consider skipping this trip"