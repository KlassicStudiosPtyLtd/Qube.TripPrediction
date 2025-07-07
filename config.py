"""
Configuration module for fleet shift analyzer with timezone support.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ShiftConfig:
    """Configuration for driver shifts."""
    shift_duration_hours: float = 12.0
    target_trips: int = 4
    buffer_time_minutes: int = 30
    min_confidence_threshold: float = 0.8
    min_trip_duration_minutes: int = 10
    min_trip_distance_m: float = 500
    fatigue_factor_base: float = 0.05
    
    # Waypoint configuration
    start_waypoint: Optional[str] = None
    end_waypoint: Optional[str] = None
    waypoint_matching: str = 'exact'  # 'exact', 'contains', 'normalized'
    
    # Timezone configuration
    timezone: str = 'UTC'  # Default to UTC, can be overridden
    
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
            'start_waypoint': self.start_waypoint,
            'end_waypoint': self.end_waypoint,
            'waypoint_matching': self.waypoint_matching,
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