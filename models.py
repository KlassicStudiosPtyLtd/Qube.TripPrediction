"""
Data models for fleet shift analyzer with timezone support.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pytz


@dataclass
class Trip:
    """Represents a single trip with timezone-aware timestamps."""
    trip_id: str
    vehicle_id: int
    driver_id: Optional[str]
    start_place: str
    end_place: str
    start_time: datetime  # Should be timezone-aware
    end_time: datetime    # Should be timezone-aware
    duration_minutes: float
    distance_m: float
    route: List[Dict[str, Any]]
    is_round_trip: bool
    
    def __post_init__(self):
        """Ensure timestamps are timezone-aware."""
        # If timestamps are naive, assume they're UTC
        if self.start_time.tzinfo is None:
            self.start_time = pytz.UTC.localize(self.start_time)
        if self.end_time.tzinfo is None:
            self.end_time = pytz.UTC.localize(self.end_time)
    
    @property
    def distance_km(self) -> float:
        """Get distance in kilometers."""
        return self.distance_m / 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trip_id': self.trip_id,
            'vehicle_id': self.vehicle_id,
            'driver_id': self.driver_id,
            'start_place': self.start_place,
            'end_place': self.end_place,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_minutes': self.duration_minutes,
            'distance_m': self.distance_m,
            'route': self.route,
            'is_round_trip': self.is_round_trip
        }


@dataclass
class Shift:
    """Represents a driver shift with timezone-aware timestamps."""
    shift_id: str
    vehicle_id: int
    driver_id: Optional[str]
    start_time: datetime  # Should be timezone-aware
    end_time: datetime    # Should be timezone-aware
    trips: List[Trip] = field(default_factory=list)
    
    def __post_init__(self):
        """Ensure timestamps are timezone-aware."""
        # If timestamps are naive, assume they're UTC
        if self.start_time.tzinfo is None:
            self.start_time = pytz.UTC.localize(self.start_time)
        if self.end_time.tzinfo is None:
            self.end_time = pytz.UTC.localize(self.end_time)
    
    @property
    def duration_hours(self) -> float:
        """Get shift duration in hours."""
        return (self.end_time - self.start_time).total_seconds() / 3600
    
    @property
    def trips_completed(self) -> int:
        """Get number of completed trips."""
        return len(self.trips)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'shift_id': self.shift_id,
            'vehicle_id': self.vehicle_id,
            'driver_id': self.driver_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_hours': self.duration_hours,
            'trips': [trip.to_dict() for trip in self.trips],
            'trips_completed': self.trips_completed
        }


@dataclass
class TripPrediction:
    """Prediction result for a trip with timezone-aware timestamps."""
    estimated_duration_minutes: float
    confidence_score: float
    estimated_completion_time: datetime  # Should be timezone-aware
    will_complete_in_shift: bool
    risk_level: str  # 'low', 'medium', 'high'
    recommendation: str
    factors: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure timestamp is timezone-aware."""
        if self.estimated_completion_time.tzinfo is None:
            self.estimated_completion_time = pytz.UTC.localize(self.estimated_completion_time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'estimated_duration_minutes': self.estimated_duration_minutes,
            'confidence_score': self.confidence_score,
            'estimated_completion_time': self.estimated_completion_time.isoformat(),
            'will_complete_in_shift': self.will_complete_in_shift,
            'risk_level': self.risk_level,
            'recommendation': self.recommendation,
            'factors': self.factors
        }


@dataclass
class ShiftAnalysis:
    """Analysis result for a shift."""
    shift: Shift
    predictions: List[TripPrediction]
    can_complete_target: bool
    risk_level: str
    alert_required: bool
    recommendation: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'shift': self.shift.to_dict(),
            'predictions': [p.to_dict() for p in self.predictions],
            'can_complete_target': self.can_complete_target,
            'risk_level': self.risk_level,
            'alert_required': self.alert_required,
            'recommendation': self.recommendation,
            'metrics': self.metrics
        }


@dataclass
class Alert:
    """Represents an alert with timezone-aware timestamps."""
    alert_id: str
    timestamp: datetime  # Should be timezone-aware
    vehicle_id: int
    vehicle_name: str
    driver_id: Optional[str]
    shift_date: str
    severity: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    actions_required: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Ensure timestamp is timezone-aware."""
        if self.timestamp.tzinfo is None:
            self.timestamp = pytz.UTC.localize(self.timestamp)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'vehicle_id': self.vehicle_id,
            'vehicle_name': self.vehicle_name,
            'driver_id': self.driver_id,
            'shift_date': self.shift_date,
            'severity': self.severity,
            'message': self.message,
            'details': self.details,
            'actions_required': self.actions_required
        }