"""
Trip completion prediction logic with three-point round trip support.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pytz

import numpy as np

from config import ShiftConfig
from models import Shift, Trip, TripPrediction

logger = logging.getLogger(__name__)


class TripPredictor:
    """Predicts trip completion times with support for three-point round trips."""
    
    def __init__(self, config: ShiftConfig):
        self.config = config
        self.historical_data = []
        
    def predict_next_trip(self, shift: Shift, current_time: datetime,
                         trips_already_completed: int) -> TripPrediction:
        """
        Predict the next trip duration and completion time.
        
        Handles both simple trips and three-point round trips based on configuration.
        
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
        
        # Check if we're in three-point mode
        if self.config.round_trip_mode == 'three_point':
            return self._predict_three_point_round_trip(
                shift, current_time, trips_already_completed
            )
        else:
            # Use existing simple prediction logic
            return self._predict_simple_trip(
                shift, current_time, trips_already_completed
            )
    
    def _predict_three_point_round_trip(self, shift: Shift, current_time: datetime,
                                      trips_already_completed: int) -> TripPrediction:
        """
        Predict completion time for a three-point round trip.
        
        Args:
            shift: Current shift
            current_time: Current time (timezone-aware)
            trips_already_completed: Number of complete round trips already done
            
        Returns:
            TripPrediction with segment-by-segment predictions
        """
        # Extract prediction factors
        factors = self._extract_prediction_factors(
            shift, current_time, trips_already_completed
        )
        
        # Analyze historical segment durations
        segment_durations = self._analyze_segment_durations(shift.trips)
        
        # Predict each segment
        segment_predictions = {}
        prediction_time = current_time
        total_duration = 0
        
        segments = ['start_to_target', 'target_to_end']
        for i, segment in enumerate(segments):
            # Get base duration for this segment
            if segment in segment_durations and segment_durations[segment]['count'] > 0:
                base_duration = segment_durations[segment]['avg']
                confidence = min(0.9, 0.5 + 0.1 * segment_durations[segment]['count'])
                calculation_method = f'historical_average ({segment_durations[segment]["count"]} samples)'
            else:
                # Use configured default for this segment
                base_duration = self.config.default_segment_durations.get(
                    segment, self.config.default_trip_duration_minutes
                )
                confidence = 0.4
                calculation_method = 'default_estimate'
            
            # Apply modifiers for this segment
            adjusted_duration, adjustment_details = self._apply_segment_modifiers(
                base_duration, segment, prediction_time, trips_already_completed, i, factors
            )
            
            # Calculate completion time for this segment
            segment_end_time = prediction_time + timedelta(minutes=adjusted_duration)
            
            # Store segment prediction
            segment_predictions[segment] = {
                'base_duration': base_duration,
                'adjusted_duration': adjusted_duration,
                'start_time': prediction_time.isoformat(),
                'end_time': segment_end_time.isoformat(),
                'confidence': confidence,
                'calculation_method': calculation_method,
                'adjustments': adjustment_details
            }
            
            # Update for next segment
            prediction_time = segment_end_time
            total_duration += adjusted_duration
        
        # Overall confidence is minimum of segment confidences
        overall_confidence = min(seg['confidence'] for seg in segment_predictions.values())
        
        # Check if completion is within shift
        completion_time = prediction_time
        will_complete = completion_time <= shift.end_time
        
        # Determine risk level
        time_to_shift_end = (shift.end_time - completion_time).total_seconds() / 60
        risk_level, risk_details = self._calculate_round_trip_risk_level(
            time_to_shift_end, total_duration, trips_already_completed, segment_predictions
        )
        
        # Add additional factors
        factors['segment_durations'] = segment_durations
        factors['time_to_shift_end'] = time_to_shift_end
        factors['risk_details'] = risk_details
        factors['round_trip_type'] = 'three_point'
        factors['total_segments'] = len(segments)
        
        return TripPrediction(
            estimated_duration_minutes=total_duration,
            confidence_score=overall_confidence,
            estimated_completion_time=completion_time,
            will_complete_in_shift=will_complete,
            risk_level=risk_level,
            recommendation=self._generate_three_point_recommendation(
                will_complete, risk_level, time_to_shift_end, 
                total_duration, factors, segment_predictions
            ),
            factors=factors,
            segment_predictions=segment_predictions,
            total_segments=len(segments)
        )
    
    def _analyze_segment_durations(self, trips: List[Trip]) -> Dict[str, Dict[str, float]]:
        """
        Analyze historical durations for each segment type.
        
        Args:
            trips: Historical trips from the shift
            
        Returns:
            Dictionary with statistics for each segment type
        """
        segment_stats = {
            'start_to_target': {'durations': [], 'avg': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0},
            'target_to_end': {'durations': [], 'avg': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
        }
        
        # Collect durations from three-point trips
        for trip in trips:
            if trip.trip_type == 'three_point_round' and trip.is_complete_round_trip:
                # Extract segment durations
                if len(trip.trip_segments) >= 2:
                    segment_stats['start_to_target']['durations'].append(
                        trip.trip_segments[0].duration_minutes
                    )
                    segment_stats['target_to_end']['durations'].append(
                        trip.trip_segments[1].duration_minutes
                    )
        
        # Calculate statistics
        for segment, data in segment_stats.items():
            if data['durations']:
                data['avg'] = np.mean(data['durations'])
                data['std'] = np.std(data['durations']) if len(data['durations']) > 1 else 0
                data['min'] = np.min(data['durations'])
                data['max'] = np.max(data['durations'])
                data['count'] = len(data['durations'])
        
        return segment_stats
    
    def _apply_segment_modifiers(self, base_duration: float, segment: str,
                               start_time: datetime, trips_completed: int,
                               segment_index: int, factors: Dict[str, Any]) -> Tuple[float, List[Dict]]:
        """
        Apply modifiers to segment duration prediction.
        
        Args:
            base_duration: Base duration for the segment
            segment: Segment type ('start_to_target' or 'target_to_end')
            start_time: When this segment will start
            trips_completed: Number of complete round trips done
            segment_index: Index of this segment (0 or 1)
            factors: General prediction factors
            
        Returns:
            Tuple of (adjusted_duration, adjustment_details)
        """
        modified_duration = base_duration
        adjustments = []
        
        # Peak hour modifier (check the hour when this segment will occur)
        if hasattr(start_time, 'hour'):
            hour_of_day = start_time.hour
            is_peak = 6 <= hour_of_day <= 9 or 16 <= hour_of_day <= 19
            
            if is_peak:
                peak_adjustment = base_duration * 0.15
                modified_duration += peak_adjustment
                adjustments.append({
                    'type': 'peak_hour',
                    'description': f'Peak hour traffic for {segment} segment',
                    'adjustment_minutes': peak_adjustment,
                    'multiplier': 1.15
                })
        
        # Fatigue modifier (increases more for second segment)
        fatigue_base = factors.get('fatigue_factor', 1.0)
        segment_fatigue = fatigue_base * (1 + 0.1 * segment_index)  # Additional 10% for second segment
        
        if segment_fatigue > 1.0:
            fatigue_adjustment = base_duration * (segment_fatigue - 1.0)
            modified_duration = base_duration * segment_fatigue
            adjustments.append({
                'type': 'fatigue',
                'description': f'Driver fatigue for {segment} segment (trip {trips_completed + 1}, segment {segment_index + 1})',
                'adjustment_minutes': fatigue_adjustment,
                'multiplier': segment_fatigue
            })
        
        # Segment-specific adjustments
        if segment == 'target_to_end' and self.config.start_waypoint == self.config.end_waypoint:
            # Return journey might be faster (empty vehicle)
            return_adjustment = -base_duration * 0.05  # 5% faster
            modified_duration += return_adjustment
            adjustments.append({
                'type': 'return_journey',
                'description': 'Return journey adjustment (potentially empty)',
                'adjustment_minutes': return_adjustment,
                'multiplier': 0.95
            })
        
        # Add uncertainty buffer for segments with high variance
        segment_stats = factors.get('segment_durations', {}).get(segment, {})
        if segment_stats.get('std', 0) > base_duration * 0.2:  # High variance
            variance_adjustment = segment_stats['std'] * 0.5
            modified_duration += variance_adjustment
            adjustments.append({
                'type': 'variance',
                'description': f'High variance in {segment} segment times',
                'adjustment_minutes': variance_adjustment,
                'multiplier': 1.0 + (variance_adjustment / base_duration)
            })
        
        # Summary
        total_adjustment = modified_duration - base_duration
        total_multiplier = modified_duration / base_duration if base_duration > 0 else 1.0
        
        adjustments.append({
            'type': 'total',
            'description': f'Combined adjustments for {segment}',
            'adjustment_minutes': total_adjustment,
            'multiplier': total_multiplier
        })
        
        return modified_duration, adjustments
    
    def _calculate_round_trip_risk_level(self, time_buffer: float, total_duration: float,
                                       trips_completed: int, 
                                       segment_predictions: Dict[str, Dict]) -> Tuple[str, str]:
        """
        Calculate risk level for a three-point round trip.
        
        Args:
            time_buffer: Minutes remaining after trip completion
            total_duration: Total predicted trip duration
            trips_completed: Number of trips already completed
            segment_predictions: Predictions for each segment
            
        Returns:
            Tuple of (risk_level, explanation)
        """
        if time_buffer < 0:
            return 'critical', f"Round trip would exceed shift by {abs(time_buffer):.0f} minutes"
        
        # Calculate buffer ratio
        buffer_ratio = time_buffer / total_duration if total_duration > 0 else 0
        
        # Consider segment-specific risks
        segment_risks = []
        for segment, pred in segment_predictions.items():
            if pred['confidence'] < 0.6:
                segment_risks.append(f"{segment}: low confidence ({pred['confidence']:.1f})")
        
        # Determine overall risk
        if buffer_ratio > 1.5 and not segment_risks:
            risk = 'low'
            explanation = f"Comfortable buffer: {time_buffer:.0f} min after round trip"
        elif buffer_ratio > 0.75:
            risk = 'medium'
            explanation = f"Moderate buffer: {time_buffer:.0f} min after round trip"
            if segment_risks:
                explanation += f" (Concerns: {', '.join(segment_risks)})"
        elif buffer_ratio > 0.25:
            risk = 'high'
            explanation = f"Tight buffer: only {time_buffer:.0f} min after round trip"
        else:
            risk = 'critical'
            explanation = f"Minimal buffer: {time_buffer:.0f} min (high risk of overrun)"
        
        return risk, explanation
    
    def _generate_three_point_recommendation(self, will_complete: bool, risk_level: str,
                                           time_buffer: float, total_duration: float,
                                           factors: Dict[str, Any],
                                           segment_predictions: Dict[str, Dict]) -> str:
        """Generate detailed recommendation for three-point round trip."""
        details = []
        
        # Header
        details.append("THREE-POINT ROUND TRIP ANALYSIS")
        details.append(f"Route: {self.config.start_waypoint} → {self.config.target_waypoint} → {self.config.end_waypoint}")
        details.append("")
        
        # Segment breakdown
        details.append("SEGMENT PREDICTIONS:")
        for segment, pred in segment_predictions.items():
            segment_name = segment.replace('_', ' ').title()
            details.append(f"\n{segment_name}:")
            details.append(f"• Base duration: {pred['base_duration']:.1f} min ({pred['calculation_method']})")
            details.append(f"• Adjusted duration: {pred['adjusted_duration']:.1f} min")
            details.append(f"• Confidence: {pred['confidence']:.1%}")
            
            # Show key adjustments
            key_adjustments = [adj for adj in pred['adjustments'] if adj['type'] != 'total']
            if key_adjustments:
                for adj in key_adjustments[:2]:  # Show top 2 adjustments
                    details.append(f"  - {adj['description']}: {adj['adjustment_minutes']:+.1f} min")
        
        # Total duration
        details.append(f"\nTOTAL ROUND TRIP DURATION: {total_duration:.1f} minutes")
        
        # Risk assessment
        details.append(f"\nRISK ASSESSMENT:")
        details.append(f"• {factors['risk_details']}")
        details.append(f"• Time remaining in shift: {time_buffer + total_duration:.0f} minutes")
        details.append(f"• Buffer after completion: {time_buffer:.0f} minutes")
        
        # Historical context
        segment_stats = factors.get('segment_durations', {})
        if any(stats.get('count', 0) > 0 for stats in segment_stats.values()):
            details.append("\nHISTORICAL CONTEXT:")
            for segment, stats in segment_stats.items():
                if stats.get('count', 0) > 0:
                    segment_name = segment.replace('_', ' ').title()
                    details.append(f"• {segment_name}: {stats['count']} trips, "
                                 f"avg {stats['avg']:.1f} min (±{stats['std']:.1f})")
        
        # Recommendation
        details.append("\nRECOMMENDATION:")
        if will_complete and risk_level == 'low':
            action = "Safe to start round trip. All segments can be completed comfortably within shift."
        elif will_complete and risk_level == 'medium':
            action = "Proceed with round trip but maintain steady pace. Monitor progress at target waypoint."
        elif risk_level == 'high':
            action = "HIGH RISK: Consider partial completion only (skip return segment if running late)."
        else:
            action = "DO NOT START: Round trip cannot be completed within shift time."
        
        details.append(action)
        
        # Special warnings
        if factors.get('fatigue_factor', 1.0) > 1.2:
            details.append("\n⚠️ FATIGUE WARNING: Driver fatigue is significantly affecting trip durations.")
        
        incomplete_count = sum(1 for trip in factors.get('recent_trips', []) 
                             if trip.trip_type == 'three_point_round' and not trip.is_complete_round_trip)
        if incomplete_count > 0:
            details.append(f"\n⚠️ PATTERN WARNING: {incomplete_count} incomplete round trips in recent history.")
        
        return "\n".join(details)
    
    def _predict_simple_trip(self, shift: Shift, current_time: datetime,
                           trips_already_completed: int) -> TripPrediction:
        """Original simple trip prediction logic (backward compatibility)."""
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
        
        # Add recent trips for pattern analysis
        recent_trips = shift.trips[-5:] if shift.trips else []
        
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
            'default_duration_used': avg_duration == 0,
            'recent_trips': recent_trips
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