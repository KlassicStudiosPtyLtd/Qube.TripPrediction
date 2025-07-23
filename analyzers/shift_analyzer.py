"""
Shift analysis logic with three-point round trip support.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pytz

import numpy as np

from config import ShiftConfig
from models import Trip, Shift, ShiftAnalysis, TripPrediction
from predictors.trip_predictor import TripPredictor

logger = logging.getLogger(__name__)


class ShiftAnalyzer:
    """Analyzes shifts for completion prediction with three-point round trip support."""
    
    def __init__(self, config: ShiftConfig):
        self.config = config
        self.predictor = TripPredictor(config)
        self.analysis_end_time = None  # Will be set by analyze_shifts
        
    def analyze_shifts(self, vehicle_id: int, vehicle_name: str, 
                      trips: List[Trip], timezone: str = 'UTC',
                      analysis_end_time: Optional[datetime] = None,
                      history_data: Optional[List[Dict[str, Any]]] = None) -> List[ShiftAnalysis]:
        """
        Analyze all shifts for a vehicle.
        
        Args:
            vehicle_id: Vehicle ID
            vehicle_name: Vehicle display name
            trips: List of trips (with UTC times)
            timezone: Timezone for shift boundaries
            analysis_end_time: End time of the analysis period (UTC)
            history_data: Raw history data for driver detection
            
        Returns:
            List of ShiftAnalysis objects
        """
        if not trips and not analysis_end_time:
            return []
        
        # Store analysis end time for shift boundary calculations
        self.analysis_end_time = analysis_end_time
        
        # Group trips into shifts based on detection mode
        if self.config.shift_detection_mode == 'driver_based':
            shifts = self._group_trips_by_driver(vehicle_id, trips, timezone, history_data)
        else:
            shifts = self._group_trips_into_shifts(vehicle_id, trips, timezone)
        
        # Analyze each shift
        shift_analyses = []
        for shift in shifts:
            analysis = self._analyze_single_shift(shift)
            shift_analyses.append(analysis)
        
        logger.info(f"Analyzed {len(shift_analyses)} shifts for vehicle {vehicle_name}")
        
        return shift_analyses
    
    def _group_trips_into_shifts(self, vehicle_id: int, trips: List[Trip], 
                                timezone: str) -> List[Shift]:
        """Group trips into shifts based on fixed shift boundaries (6am-6pm, 6pm-6am)."""
        if not trips and not self.analysis_end_time:
            return []
        
        # Sort trips by start time
        sorted_trips = sorted(trips, key=lambda t: t.start_time) if trips else []
        
        # Get timezone object
        tz = pytz.timezone(timezone)
        
        # Dictionary to store trips by shift period
        shifts_dict = {}
        
        for trip in sorted_trips:
            # Determine which shift this trip belongs to
            shift_start_utc, shift_end_utc = self._identify_shift_period(
                trip.start_time, timezone
            )
            
            # Create shift key
            shift_key = (shift_start_utc, shift_end_utc)
            
            # Add trip to appropriate shift
            if shift_key not in shifts_dict:
                shifts_dict[shift_key] = []
            shifts_dict[shift_key].append(trip)
        
        # If no trips but we have analysis_end_time, create empty shift for current period
        if not sorted_trips and self.analysis_end_time:
            shift_start_utc, shift_end_utc = self._identify_shift_period(
                self.analysis_end_time, timezone
            )
            # Only create empty shift if we're within the shift period
            if shift_start_utc <= self.analysis_end_time <= shift_end_utc:
                shift_key = (shift_start_utc, shift_end_utc)
                shifts_dict[shift_key] = []
        
        # Create Shift objects
        shifts = []
        
        for (shift_start, shift_end), shift_trips in sorted(shifts_dict.items()):
            # Get driver ID from first trip if available
            driver_id = shift_trips[0].driver_id if shift_trips else None
            
            shift = Shift(
                shift_id=f"{vehicle_id}_{shift_start.strftime('%Y%m%d_%H')}",
                vehicle_id=vehicle_id,
                driver_id=driver_id,
                start_time=shift_start,
                end_time=shift_end,
                trips=shift_trips
            )
            shifts.append(shift)
        
        return sorted(shifts, key=lambda s: s.start_time)
    
    def _identify_shift_period(self, check_time: datetime, timezone: str) -> Tuple[datetime, datetime]:
        """
        Identify which shift period a given time belongs to.
        Returns (shift_start_utc, shift_end_utc)
        
        Shift periods:
        - Day shift: 6am - 6pm local time
        - Night shift: 6pm - 6am next day local time
        """
        # Ensure check_time is timezone-aware
        if check_time.tzinfo is None:
            check_time = pytz.UTC.localize(check_time)
        
        # Convert to local timezone
        tz = pytz.timezone(timezone)
        local_time = check_time.astimezone(tz)
        
        # Determine shift boundaries in local time
        if 6 <= local_time.hour < 18:
            # Day shift: 6am - 6pm same day
            shift_start_local = local_time.replace(hour=6, minute=0, second=0, microsecond=0)
            shift_end_local = local_time.replace(hour=18, minute=0, second=0, microsecond=0)
        else:
            # Night shift: 6pm - 6am next day
            if local_time.hour >= 18:
                # Evening part of night shift
                shift_start_local = local_time.replace(hour=18, minute=0, second=0, microsecond=0)
                shift_end_local = (local_time + timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0)
            else:
                # Morning part of night shift (midnight to 6am)
                shift_start_local = (local_time - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
                shift_end_local = local_time.replace(hour=6, minute=0, second=0, microsecond=0)
        
        # Convert back to UTC for storage
        shift_start_utc = shift_start_local.astimezone(pytz.UTC)
        shift_end_utc = shift_end_local.astimezone(pytz.UTC)
        
        return shift_start_utc, shift_end_utc
    
    def _analyze_single_shift(self, shift: Shift) -> ShiftAnalysis:
        """Analyze a single shift for completion prediction with three-point support."""
        # Calculate overtime if in driver-based mode
        overtime_hours = 0.0
        overtime_cost = 0.0
        if self.config.shift_detection_mode == 'driver_based' and shift.actual_end_time:
            overtime_hours = max(0, shift.actual_duration_hours - self.config.shift_duration_hours)
            # Assuming overtime rate is 1.5x normal rate (can be configured later)
            overtime_cost = overtime_hours * 1.5 * 50  # Assuming $50/hour base rate
        
        # Determine what to count based on mode
        if self.config.round_trip_mode == 'three_point':
            # Count complete round trips
            trips_completed = shift.complete_round_trips
            incomplete_count = len(shift.incomplete_round_trips)
        else:
            # Count all trips
            trips_completed = shift.trips_completed
            incomplete_count = 0
        
        trips_remaining = max(0, self.config.target_trips - trips_completed)
        
        # Calculate how far into the shift we are
        if hasattr(self, 'analysis_end_time') and self.analysis_end_time:
            current_time = min(self.analysis_end_time, datetime.now(pytz.UTC))
        else:
            current_time = datetime.now(pytz.UTC)
        
        # Ensure current_time is timezone-aware
        if current_time.tzinfo is None:
            current_time = pytz.UTC.localize(current_time)
        
        time_into_shift = (current_time - shift.start_time).total_seconds() / 3600
        
        # Get current status
        if trips_remaining == 0:
            return self._create_target_achieved_analysis(
                shift, trips_completed, time_into_shift, incomplete_count
            )
        
        # For new shifts with no trips, use more appropriate messaging
        if trips_completed == 0 and time_into_shift < 2.0:
            return self._create_early_shift_analysis(
                shift, current_time, time_into_shift, trips_remaining
            )
        
        # Get last trip end time or current time if no trips
        if shift.trips:
            last_trip_end = shift.trips[-1].end_time
        else:
            last_trip_end = current_time
        
        # Ensure last_trip_end is timezone-aware
        if last_trip_end.tzinfo is None:
            last_trip_end = pytz.UTC.localize(last_trip_end)
        
        # Calculate remaining time
        remaining_time = (shift.end_time - last_trip_end).total_seconds() / 60
        
        # Predict remaining trips
        predictions = []
        prediction_current_time = last_trip_end
        
        for i in range(trips_remaining):
            prediction = self.predictor.predict_next_trip(
                shift, prediction_current_time, trips_completed + i
            )
            predictions.append(prediction)
            prediction_current_time = prediction.estimated_completion_time
        
        # Determine if target can be completed
        total_estimated_time = sum(p.estimated_duration_minutes for p in predictions)
        buffer_adjusted_time = remaining_time - self.config.buffer_time_minutes
        can_complete = total_estimated_time <= buffer_adjusted_time
        
        # Determine risk level (consider overtime in driver-based mode)
        if self.config.shift_detection_mode == 'driver_based':
            # Check if completing all remaining trips would cause overtime
            projected_end_time = prediction_current_time if predictions else last_trip_end
            projected_duration = (projected_end_time - shift.start_time).total_seconds() / 3600
            
            if projected_duration > self.config.shift_duration_hours:
                # Increase risk level if overtime is projected
                base_risk = self._calculate_risk_level(total_estimated_time, buffer_adjusted_time, time_into_shift)
                if base_risk == 'low':
                    risk_level = 'medium'
                elif base_risk == 'medium':
                    risk_level = 'high'
                else:
                    risk_level = 'critical'
            else:
                risk_level = self._calculate_risk_level(total_estimated_time, buffer_adjusted_time, time_into_shift)
        else:
            risk_level = self._calculate_risk_level(total_estimated_time, buffer_adjusted_time, time_into_shift)
        
        # Generate detailed recommendation
        recommendation = self._generate_detailed_recommendation(
            trips_completed, trips_remaining, remaining_time, 
            total_estimated_time, buffer_adjusted_time,
            can_complete, risk_level, time_into_shift, shift, predictions,
            incomplete_count, overtime_hours
        )
        
        # Build metrics
        metrics = {
            'trips_completed': trips_completed,
            'trips_target': self.config.target_trips,
            'trips_remaining': trips_remaining,
            'avg_trip_duration': self._calculate_average_trip_duration(shift),
            'total_estimated_time': total_estimated_time,
            'remaining_time': remaining_time,
            'buffer_adjusted_time': buffer_adjusted_time,
            'completion_probability': 1.0 - (total_estimated_time / buffer_adjusted_time) if buffer_adjusted_time > 0 else 0,
            'time_into_shift_hours': time_into_shift,
            'shift_duration_hours': shift.duration_hours,
            'incomplete_round_trips': incomplete_count
        }
        
        # Add driver-based metrics
        if self.config.shift_detection_mode == 'driver_based':
            metrics['driver_id'] = shift.driver_id
            metrics['driver_name'] = shift.driver_name
            metrics['driver_ref'] = shift.driver_ref
            metrics['overtime_hours'] = overtime_hours
            metrics['overtime_cost'] = overtime_cost
            if predictions:
                projected_end_time = predictions[-1].estimated_completion_time
                projected_duration = (projected_end_time - shift.start_time).total_seconds() / 3600
                metrics['projected_shift_duration'] = projected_duration
                metrics['projected_overtime_hours'] = max(0, projected_duration - self.config.shift_duration_hours)
        
        # Add three-point specific metrics
        if self.config.round_trip_mode == 'three_point':
            metrics['complete_round_trips'] = shift.complete_round_trips
            metrics['round_trip_completion_rate'] = (
                shift.complete_round_trips / len(shift.trips) 
                if shift.trips else 0
            )
        
        return ShiftAnalysis(
            shift=shift,
            predictions=predictions,
            can_complete_target=can_complete,
            risk_level=risk_level,
            alert_required=risk_level in ['high', 'critical'] and time_into_shift >= 1.0,
            recommendation=recommendation,
            metrics=metrics
        )
    
    def _calculate_average_trip_duration(self, shift: Shift) -> float:
        """Calculate average trip duration based on mode."""
        if not shift.trips:
            return 0
        
        if self.config.round_trip_mode == 'three_point':
            # For three-point mode, calculate average of complete round trips
            complete_trips = [t for t in shift.trips if t.is_complete_round_trip]
            if complete_trips:
                return np.mean([t.duration_minutes for t in complete_trips])
            else:
                # Fall back to all trips if no complete round trips
                return np.mean([t.duration_minutes for t in shift.trips])
        else:
            # Simple mode: average of all trips
            return np.mean([t.duration_minutes for t in shift.trips])
    
    def _create_target_achieved_analysis(self, shift: Shift, trips_completed: int,
                                       time_into_shift: float, incomplete_count: int) -> ShiftAnalysis:
        """Create analysis for shifts that have achieved their target."""
        metrics = {
            'trips_completed': trips_completed,
            'trips_target': self.config.target_trips,
            'completion_rate': 1.0,
            'time_into_shift_hours': time_into_shift,
            'incomplete_round_trips': incomplete_count
        }
        
        recommendation = f"TARGET ACHIEVED - Algorithm Status:\n"
        recommendation += f"• Completed: {trips_completed}/{self.config.target_trips} "
        
        if self.config.round_trip_mode == 'three_point':
            recommendation += "complete round trips\n"
            if incomplete_count > 0:
                recommendation += f"• Incomplete round trips: {incomplete_count}\n"
        else:
            recommendation += "trips\n"
        
        recommendation += f"• Time used: {time_into_shift:.1f} hours\n"
        recommendation += f"• Recommendation: Target met. Additional trips optional."
        
        return ShiftAnalysis(
            shift=shift,
            predictions=[],
            can_complete_target=True,
            risk_level='none',
            alert_required=False,
            recommendation=recommendation,
            metrics=metrics
        )
    
    def _create_early_shift_analysis(self, shift: Shift, current_time: datetime,
                                   time_into_shift: float, trips_remaining: int) -> ShiftAnalysis:
        """Create analysis for early in shift with no trips."""
        remaining_time = (shift.end_time - current_time).total_seconds() / 60
        
        metrics = {
            'trips_completed': 0,
            'trips_target': self.config.target_trips,
            'completion_rate': 0.0,
            'time_into_shift_hours': time_into_shift,
            'shift_duration_hours': shift.duration_hours,
            'remaining_time': remaining_time,
            'incomplete_round_trips': 0
        }
        
        recommendation = f"SHIFT START - Algorithm Status:\n"
        recommendation += f"• Time into shift: {time_into_shift:.1f} hours\n"
        recommendation += f"• Remaining shift time: {remaining_time / 60:.1f} hours\n"
        recommendation += f"• Target: {self.config.target_trips} "
        
        if self.config.round_trip_mode == 'three_point':
            recommendation += f"three-point round trips\n"
            recommendation += f"• Route: {self.config.start_waypoint} → {self.config.target_waypoint} → {self.config.end_waypoint}\n"
            total_est = sum(self.config.default_segment_durations.values())
            recommendation += f"• Estimated round trip duration: {total_est} min\n"
        else:
            recommendation += f"trips\n"
            recommendation += f"• Default trip duration: {self.config.default_trip_duration_minutes} min\n"
        
        recommendation += f"• Recommendation: Begin first trip when ready."
        
        return ShiftAnalysis(
            shift=shift,
            predictions=[],
            can_complete_target=True,
            risk_level='low',
            alert_required=False,
            recommendation=recommendation,
            metrics=metrics
        )
    
    def _calculate_risk_level(self, estimated_time: float, available_time: float, 
                             time_into_shift: float) -> str:
        """Calculate risk level based on time estimates and shift progress."""
        # Early in shift, be less aggressive with risk levels
        if time_into_shift < 2.0:  # First 2 hours
            if available_time <= 0:
                return 'medium'  # Downgrade from critical
            
            time_ratio = estimated_time / available_time if available_time > 0 else 2.0
            
            if time_ratio <= 0.8:
                return 'low'
            elif time_ratio <= 1.1:
                return 'medium'
            else:
                return 'high'
        
        # Normal risk calculation for later in shift
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
    
    def _generate_detailed_recommendation(self, trips_completed: int, trips_remaining: int,
                                        remaining_time: float, total_estimated_time: float,
                                        buffer_adjusted_time: float, can_complete: bool,
                                        risk_level: str, time_into_shift: float,
                                        shift: Shift, predictions: List[TripPrediction],
                                        incomplete_count: int, overtime_hours: float = 0.0) -> str:
        """Generate detailed recommendation with algorithmic explanation."""
        
        # Format time values
        remaining_hours = remaining_time / 60
        buffer_hours = self.config.buffer_time_minutes / 60
        
        # Build detailed algorithm explanation
        calculation_details = []
        
        # Mode-specific header
        if self.config.round_trip_mode == 'three_point':
            calculation_details.append(f"THREE-POINT ROUND TRIP ANALYSIS - Algorithm Status:")
            calculation_details.append(f"Route: {self.config.start_waypoint} → {self.config.target_waypoint} → {self.config.end_waypoint}")
        else:
            calculation_details.append(f"TRIP ANALYSIS - Algorithm Status:")
        
        # Historical data section
        if shift.trips:
            avg_duration = self._calculate_average_trip_duration(shift)
            std_duration = np.std([t.duration_minutes for t in shift.trips]) if len(shift.trips) > 1 else 0
            calculation_details.append(
                f"\nHISTORICAL DATA:\n"
                f"• Trips completed: {trips_completed}\n"
                f"• Average duration: {avg_duration:.1f} min\n"
                f"• Standard deviation: {std_duration:.1f} min"
            )
            
            if self.config.round_trip_mode == 'three_point' and incomplete_count > 0:
                calculation_details.append(f"• Incomplete round trips: {incomplete_count}")
        else:
            calculation_details.append(
                f"\nHISTORICAL DATA:\n"
                f"• No trips completed yet\n"
                f"• Using default duration: {self.config.default_trip_duration_minutes} min"
            )
        
        # Time calculations
        calculation_details.append(
            f"\nTIME CALCULATIONS:\n"
            f"• Current time into shift: {time_into_shift:.1f} hours\n"
            f"• Remaining shift time: {remaining_hours:.1f} hours ({remaining_time:.0f} min)\n"
            f"• Buffer time: {buffer_hours:.1f} hours ({self.config.buffer_time_minutes} min)\n"
            f"• Effective time available: {buffer_adjusted_time:.0f} min"
        )
        
        # Trip predictions
        if predictions:
            pred_details = "\nTRIP PREDICTIONS:"
            for i, pred in enumerate(predictions):
                trip_num = trips_completed + i + 1
                
                if self.config.round_trip_mode == 'three_point' and pred.segment_predictions:
                    pred_details += f"\n• Round Trip {trip_num}:"
                    for segment, seg_pred in pred.segment_predictions.items():
                        segment_name = segment.replace('_', ' ').title()
                        pred_details += f"\n  - {segment_name}: {seg_pred['adjusted_duration']:.1f} min"
                    pred_details += f"\n  - Total: {pred.estimated_duration_minutes:.1f} min"
                else:
                    pred_details += f"\n• Trip {trip_num}: {pred.estimated_duration_minutes:.1f} min"
            
            calculation_details.append(pred_details)
            calculation_details.append(
                f"\n• Total estimated time: {total_estimated_time:.0f} min"
                f"\n• Time utilization: {(total_estimated_time / buffer_adjusted_time * 100):.1f}%" if buffer_adjusted_time > 0 else ""
            )
        
        # Risk assessment
        risk_explanation = self._get_risk_explanation(total_estimated_time, buffer_adjusted_time, time_into_shift)
        calculation_details.append(f"\nRISK ASSESSMENT:\n{risk_explanation}")
        
        # Final recommendation
        if can_complete and risk_level == 'low':
            action = f"Continue with all {trips_remaining} remaining trips. Ample time available."
        elif can_complete and risk_level == 'medium':
            action = f"Continue with {trips_remaining} trips but maintain steady pace. Monitor progress."
        elif risk_level == 'high':
            if self.config.round_trip_mode == 'three_point':
                action = "HIGH RISK: Consider skipping final round trip or completing partial trip only."
            else:
                action = "HIGH RISK: Consider skipping final trip."
        else:  # critical
            safe_trips = max(0, trips_remaining - 1)
            action = f"CRITICAL: Complete maximum {safe_trips} trips. Multiple trips at risk."
        
        calculation_details.append(f"\nRECOMMENDATION: {action}")
        
        # Add fatigue warning if applicable
        if predictions and any(p.factors.get('fatigue_factor', 1.0) > 1.1 for p in predictions):
            calculation_details.append(
                f"\n⚠️ FATIGUE WARNING: Driver fatigue factor is increasing trip durations. "
                f"Consider additional breaks."
            )
        
        # Add overtime warnings for driver-based mode
        if self.config.shift_detection_mode == 'driver_based':
            if overtime_hours > 0:
                calculation_details.append(
                    f"\n⚠️ OVERTIME ALERT: Driver has worked {overtime_hours:.1f} hours of overtime. "
                    f"Cost impact: ${overtime_hours * 1.5 * 50:.2f}"
                )
            
            if predictions:
                # Calculate projected overtime
                projected_end_time = predictions[-1].estimated_completion_time
                projected_duration = (projected_end_time - shift.start_time).total_seconds() / 3600
                proj_overtime = max(0, projected_duration - self.config.shift_duration_hours)
                if proj_overtime > 0:
                    calculation_details.append(
                        f"\n⚠️ OVERTIME PROJECTION: Completing all trips will result in {proj_overtime:.1f} hours overtime. "
                        f"Estimated cost: ${proj_overtime * 1.5 * 50:.2f}. Consider reducing trips to avoid overtime."
                    )
        
        # Three-point specific warnings
        if self.config.round_trip_mode == 'three_point' and incomplete_count > 1:
            calculation_details.append(
                f"\n⚠️ PATTERN WARNING: {incomplete_count} incomplete round trips detected. "
                f"Consider investigating causes."
            )
        
        return "\n".join(calculation_details)
    
    def _get_risk_explanation(self, estimated_time: float, available_time: float, 
                             time_into_shift: float) -> str:
        """Generate detailed risk level explanation."""
        if available_time <= 0:
            return "• Risk Level: CRITICAL (No time remaining)"
        
        time_ratio = estimated_time / available_time
        percentage = time_ratio * 100
        
        # Early shift adjustments
        if time_into_shift < 2.0:
            thresholds = "• Thresholds (early shift): Low < 80%, Medium 80-110%, High > 110%"
        else:
            thresholds = "• Thresholds: Low < 70%, Medium 70-90%, High 90-100%, Critical > 100%"
        
        risk_level = self._calculate_risk_level(estimated_time, available_time, time_into_shift)
        
        return (f"• Estimated time needed: {estimated_time:.0f} min\n"
                f"• Time available: {available_time:.0f} min\n"
                f"• Utilization: {percentage:.1f}%\n"
                f"• Risk Level: {risk_level.upper()}\n"
                f"{thresholds}")
    
    def _group_trips_by_driver(self, vehicle_id: int, trips: List[Trip], 
                              timezone: str, history_data: Optional[List[Dict[str, Any]]] = None) -> List[Shift]:
        """Group trips into shifts based on driver changes."""
        if not trips and not history_data:
            return []
        
        # Detect driver changes
        driver_changes = self._detect_driver_changes(trips, history_data)
        
        # Sort trips by start time
        sorted_trips = sorted(trips, key=lambda t: t.start_time) if trips else []
        
        # Group trips by driver shifts
        shifts = []
        current_shift_trips = []
        current_driver_id = None
        current_driver_name = None
        current_driver_ref = None
        shift_start_time = None
        
        for i, trip in enumerate(sorted_trips):
            trip_driver_id = trip.driver_id or 'UNKNOWN'
            
            # Check if this is a new driver/shift
            is_new_shift = False
            if current_driver_id is None:
                # First trip
                is_new_shift = True
            elif trip_driver_id != current_driver_id:
                # Driver changed
                is_new_shift = True
            else:
                # Same driver - check if it's a continuation of the same shift
                if current_shift_trips:
                    last_trip_end = current_shift_trips[-1].end_time
                    time_gap_hours = (trip.start_time - last_trip_end).total_seconds() / 3600
                    
                    # If driver returns after max shift duration, it's a new shift
                    if time_gap_hours > self.config.max_shift_duration_hours:
                        is_new_shift = True
                    # Also check if total shift duration would exceed max
                    elif shift_start_time:
                        potential_duration = (trip.end_time - shift_start_time).total_seconds() / 3600
                        if potential_duration > self.config.max_shift_duration_hours:
                            is_new_shift = True
            
            if is_new_shift and current_shift_trips:
                # Create shift for previous driver
                shift_end_time = current_shift_trips[-1].end_time
                
                # Check if shift duration exceeds normal shift hours (for overtime)
                actual_duration = (shift_end_time - shift_start_time).total_seconds() / 3600
                actual_end_time = shift_end_time if actual_duration > self.config.shift_duration_hours else None
                
                shift = Shift(
                    shift_id=f"{vehicle_id}_{shift_start_time.strftime('%Y%m%d_%H%M')}_{current_driver_id}",
                    vehicle_id=vehicle_id,
                    driver_id=current_driver_id,
                    start_time=shift_start_time,
                    end_time=shift_end_time,
                    trips=current_shift_trips,
                    driver_name=current_driver_name,
                    driver_ref=current_driver_ref,
                    actual_end_time=actual_end_time
                )
                shifts.append(shift)
                
                # Reset for new shift
                current_shift_trips = []
            
            if is_new_shift:
                # Start new shift
                current_driver_id = trip_driver_id
                current_driver_name = self._get_driver_name_from_history(trip_driver_id, history_data)
                current_driver_ref = trip_driver_id if trip_driver_id != 'UNKNOWN' else None
                shift_start_time = trip.start_time
            
            current_shift_trips.append(trip)
        
        # Don't forget the last shift
        if current_shift_trips and shift_start_time:
            shift_end_time = current_shift_trips[-1].end_time
            
            # Check for overtime
            actual_duration = (shift_end_time - shift_start_time).total_seconds() / 3600
            actual_end_time = shift_end_time if actual_duration > self.config.shift_duration_hours else None
            
            shift = Shift(
                shift_id=f"{vehicle_id}_{shift_start_time.strftime('%Y%m%d_%H%M')}_{current_driver_id}",
                vehicle_id=vehicle_id,
                driver_id=current_driver_id,
                start_time=shift_start_time,
                end_time=shift_end_time,
                trips=current_shift_trips,
                driver_name=current_driver_name,
                driver_ref=current_driver_ref,
                actual_end_time=actual_end_time
            )
            shifts.append(shift)
        
        # Handle case where we have no trips but need to create a shift for current analysis time
        if not sorted_trips and self.analysis_end_time and history_data:
            # Look for the most recent driver in history
            recent_driver = self._get_most_recent_driver(history_data, self.analysis_end_time)
            if recent_driver:
                # Create an empty shift for the current driver
                shift_start_time = self.analysis_end_time.replace(hour=6, minute=0, second=0, microsecond=0)
                if self.analysis_end_time.hour < 6:
                    shift_start_time -= timedelta(days=1)
                
                shift = Shift(
                    shift_id=f"{vehicle_id}_{shift_start_time.strftime('%Y%m%d_%H%M')}_{recent_driver['id']}",
                    vehicle_id=vehicle_id,
                    driver_id=recent_driver['id'],
                    start_time=shift_start_time,
                    end_time=shift_start_time + timedelta(hours=self.config.shift_duration_hours),
                    trips=[],
                    driver_name=recent_driver.get('name'),
                    driver_ref=recent_driver.get('ref')
                )
                shifts.append(shift)
        
        return sorted(shifts, key=lambda s: s.start_time)
    
    def _detect_driver_changes(self, trips: List[Trip], 
                              history_data: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Detect driver changes from trips and history data."""
        changes = []
        
        if not trips and not history_data:
            return changes
        
        # First, extract driver info from trips
        trip_drivers = []
        for trip in sorted(trips, key=lambda t: t.start_time):
            driver_id = trip.driver_id or 'UNKNOWN'
            trip_drivers.append({
                'time': trip.start_time,
                'driver_id': driver_id,
                'source': 'trip'
            })
        
        # Then, extract driver info from history data if available
        if history_data:
            for record in history_data:
                if 'driverId' in record and record['driverId']:
                    # Convert timestamp to datetime if needed
                    timestamp = record.get('timestamp') or record.get('time')
                    if timestamp:
                        if isinstance(timestamp, str):
                            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        elif isinstance(timestamp, (int, float)):
                            timestamp = datetime.fromtimestamp(timestamp, pytz.UTC)
                        
                        if timestamp.tzinfo is None:
                            timestamp = pytz.UTC.localize(timestamp)
                        
                        trip_drivers.append({
                            'time': timestamp,
                            'driver_id': record['driverId'],
                            'driver_name': record.get('driverName'),
                            'source': 'history'
                        })
        
        # Sort all driver records by time
        trip_drivers.sort(key=lambda x: x['time'])
        
        # Detect changes
        prev_driver = None
        for record in trip_drivers:
            if prev_driver is None or record['driver_id'] != prev_driver:
                changes.append({
                    'time': record['time'],
                    'from_driver': prev_driver,
                    'to_driver': record['driver_id'],
                    'driver_name': record.get('driver_name')
                })
                prev_driver = record['driver_id']
        
        return changes
    
    def _get_driver_name_from_history(self, driver_id: str, 
                                     history_data: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """Get driver name from history data."""
        if not history_data or driver_id == 'UNKNOWN':
            return None
        
        for record in history_data:
            if record.get('driverId') == driver_id and record.get('driverName'):
                return record['driverName']
        
        return None
    
    def _get_most_recent_driver(self, history_data: List[Dict[str, Any]], 
                               before_time: datetime) -> Optional[Dict[str, Any]]:
        """Get the most recent driver from history data before a given time."""
        if not history_data:
            return None
        
        # Sort history by time descending
        sorted_history = []
        for record in history_data:
            if 'driverId' in record and record['driverId']:
                timestamp = record.get('timestamp') or record.get('time')
                if timestamp:
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    elif isinstance(timestamp, (int, float)):
                        timestamp = datetime.fromtimestamp(timestamp, pytz.UTC)
                    
                    if timestamp.tzinfo is None:
                        timestamp = pytz.UTC.localize(timestamp)
                    
                    if timestamp < before_time:
                        sorted_history.append({
                            'time': timestamp,
                            'id': record['driverId'],
                            'name': record.get('driverName'),
                            'ref': record['driverId'] if record['driverId'] != 'UNKNOWN' else None
                        })
        
        sorted_history.sort(key=lambda x: x['time'], reverse=True)
        
        return sorted_history[0] if sorted_history else None