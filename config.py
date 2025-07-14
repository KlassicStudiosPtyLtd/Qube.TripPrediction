"""
Configuration module for fleet shift analyzer with three-point round trip support.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ShiftConfig:
    """Configuration for driver shifts with three-point round trip support."""
    shift_duration_hours: float = 12.0
    target_trips: int = 4
    buffer_time_minutes: int = 30
    min_confidence_threshold: float = 0.8
    min_trip_duration_minutes: int = 10
    min_trip_distance_m: float = 500
    fatigue_factor_base: float = 0.05
    default_trip_duration_minutes: float = 45.0
    
    # Waypoint configuration for three-point round trips
    start_waypoint: Optional[str] = None
    target_waypoint: Optional[str] = None  # NEW: intermediate waypoint for three-point trips
    end_waypoint: Optional[str] = None
    waypoint_matching: str = 'exact'  # 'exact', 'contains', 'normalized'
    
    # Round trip configuration
    round_trip_mode: str = 'simple'  # 'simple' or 'three_point'
    require_waypoint_order: bool = True  # Must visit waypoints in Start->Target->End order
    allow_partial_trips: bool = False  # Whether to count incomplete round trips
    
    # Segment-specific durations (for three-point trips)
    default_segment_durations: Dict[str, float] = field(default_factory=lambda: {
        'start_to_target': 45.0,  # Default duration for Start -> Target segment
        'target_to_end': 45.0     # Default duration for Target -> End segment
    })
    
    # Timezone configuration
    timezone: str = 'UTC'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'shift_duration_hours': self.shift_duration_hours,
            'target_trips': self.target_trips,
            'buffer_time_minutes': self.buffer_time_minutes,
            'min_confidence_threshold': self.min_confidence_threshold,
            'min_trip_duration_minutes': self.min_trip_duration_minutes,
            'min_trip_distance_m': self.min_trip_distance_m,
            'fatigue_factor_base': self.fatigue_factor_base,
            'default_trip_duration_minutes': self.default_trip_duration_minutes,
            'start_waypoint': self.start_waypoint,
            'target_waypoint': self.target_waypoint,
            'end_waypoint': self.end_waypoint,
            'waypoint_matching': self.waypoint_matching,
            'round_trip_mode': self.round_trip_mode,
            'require_waypoint_order': self.require_waypoint_order,
            'allow_partial_trips': self.allow_partial_trips,
            'default_segment_durations': self.default_segment_durations,
            'timezone': self.timezone
        }


@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""
    enable_ml_predictions: bool = False
    save_individual_analyses: bool = True
    generate_visualizations: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
    

@dataclass
class AlertConfig:
    """Configuration for alert system."""
    high_risk_threshold: float = 0.8
    medium_risk_threshold: float = 0.6
    enable_notifications: bool = True
    notification_channels: list = field(default_factory=lambda: ['dashboard', 'log'])
    alert_cooldown_minutes: int = 30