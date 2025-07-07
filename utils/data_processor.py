"""
Data processing utilities for fleet shift analyzer.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
from geopy.distance import geodesic

logger = logging.getLogger(__name__)


class DataProcessor:
    """Common data processing utilities."""
    
    @staticmethod
    def calculate_distance_between_points(point1: Tuple[float, float], 
                                        point2: Tuple[float, float]) -> float:
        """
        Calculate distance between two GPS points in meters.
        
        Args:
            point1: (latitude, longitude)
            point2: (latitude, longitude)
            
        Returns:
            Distance in meters
        """
        try:
            return geodesic(point1, point2).meters
        except Exception as e:
            logger.error(f"Error calculating distance: {str(e)}")
            return 0.0
    
    @staticmethod
    def calculate_route_distance(coordinates: List[Tuple[float, float]]) -> float:
        """
        Calculate total distance of a route.
        
        Args:
            coordinates: List of (latitude, longitude) tuples
            
        Returns:
            Total distance in meters
        """
        if len(coordinates) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(coordinates)):
            total_distance += DataProcessor.calculate_distance_between_points(
                coordinates[i-1], coordinates[i]
            )
        
        return total_distance
    
    @staticmethod
    def group_by_time_windows(data: List[Dict[str, Any]], 
                            window_hours: float,
                            time_field: str = 'timestamp') -> List[List[Dict[str, Any]]]:
        """
        Group data points by time windows.
        
        Args:
            data: List of dictionaries with timestamp field
            window_hours: Size of time window in hours
            time_field: Name of timestamp field
            
        Returns:
            List of grouped data
        """
        if not data:
            return []
        
        # Sort by timestamp
        sorted_data = sorted(data, key=lambda x: x.get(time_field))
        
        groups = []
        current_group = []
        window_start = None
        
        for item in sorted_data:
            item_time = item.get(time_field)
            if isinstance(item_time, str):
                item_time = pd.to_datetime(item_time)
            
            if not current_group:
                current_group = [item]
                window_start = item_time
            else:
                hours_diff = (item_time - window_start).total_seconds() / 3600
                
                if hours_diff <= window_hours:
                    current_group.append(item)
                else:
                    groups.append(current_group)
                    current_group = [item]
                    window_start = item_time
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    @staticmethod
    def filter_outliers_iqr(values: List[float], 
                           multiplier: float = 1.5) -> Tuple[List[float], List[int]]:
        """
        Filter outliers using IQR method.
        
        Args:
            values: List of numeric values
            multiplier: IQR multiplier for outlier bounds
            
        Returns:
            Tuple of (filtered_values, outlier_indices)
        """
        if len(values) < 4:
            return values, []
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        filtered_values = []
        outlier_indices = []
        
        for i, value in enumerate(values):
            if lower_bound <= value <= upper_bound:
                filtered_values.append(value)
            else:
                outlier_indices.append(i)
        
        return filtered_values, outlier_indices
    
    @staticmethod
    def interpolate_missing_values(values: List[Optional[float]]) -> List[float]:
        """
        Interpolate missing values in a series.
        
        Args:
            values: List of values with possible None entries
            
        Returns:
            List with interpolated values
        """
        if not values:
            return []
        
        # Convert to pandas series for easy interpolation
        series = pd.Series(values)
        
        # Interpolate missing values
        if series.isna().any():
            series = series.interpolate(method='linear', limit_direction='both')
            
            # If still has NaN (at edges), fill with nearest valid value
            series = series.fillna(method='ffill').fillna(method='bfill')
        
        return series.tolist()
    
    @staticmethod
    def calculate_time_statistics(timestamps: List[datetime]) -> Dict[str, Any]:
        """
        Calculate statistics about time intervals.
        
        Args:
            timestamps: List of datetime objects
            
        Returns:
            Dictionary with time statistics
        """
        if len(timestamps) < 2:
            return {}
        
        # Calculate intervals
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds() / 60  # minutes
            intervals.append(interval)
        
        return {
            'total_duration_minutes': (timestamps[-1] - timestamps[0]).total_seconds() / 60,
            'num_intervals': len(intervals),
            'avg_interval_minutes': np.mean(intervals),
            'min_interval_minutes': np.min(intervals),
            'max_interval_minutes': np.max(intervals),
            'std_interval_minutes': np.std(intervals)
        }
    
    @staticmethod
    def normalize_place_names(place_name: str) -> str:
        """
        Normalize place names for consistent matching.
        
        Args:
            place_name: Original place name
            
        Returns:
            Normalized place name
        """
        if not place_name:
            return "Unknown"
        
        # Convert to uppercase and strip whitespace
        normalized = place_name.upper().strip()
        
        # Remove common suffixes/prefixes
        for pattern in ['_WAYPOINT', '_WP', 'WAYPOINT_', 'WP_']:
            normalized = normalized.replace(pattern, '')
        
        # Replace multiple spaces with single space
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    @staticmethod
    def extract_shift_periods(history_data: List[Dict[str, Any]], 
                            shift_hours: float = 12) -> List[Dict[str, Any]]:
        """
        Extract potential shift periods from vehicle history.
        
        Args:
            history_data: Vehicle history records
            shift_hours: Expected shift duration
            
        Returns:
            List of shift periods with start/end times
        """
        if not history_data:
            return []
        
        # Sort by timestamp
        df = pd.DataFrame(history_data)
        if 'deviceTimeUtc' in df.columns:
            df['timestamp'] = pd.to_datetime(df['deviceTimeUtc'])
            df = df.sort_values('timestamp')
        else:
            return []
        
        shifts = []
        current_shift_start = None
        last_activity = None
        
        # Look for patterns of activity/inactivity
        for idx, row in df.iterrows():
            current_time = row['timestamp']
            
            if last_activity is None:
                # First record
                current_shift_start = current_time
                last_activity = current_time
            else:
                # Check for long gaps (potential shift break)
                gap_hours = (current_time - last_activity).total_seconds() / 3600
                
                if gap_hours > shift_hours * 0.5:  # Gap > half shift duration
                    # End current shift
                    shifts.append({
                        'start_time': current_shift_start,
                        'end_time': last_activity,
                        'duration_hours': (last_activity - current_shift_start).total_seconds() / 3600
                    })
                    # Start new shift
                    current_shift_start = current_time
                
                last_activity = current_time
        
        # Add final shift
        if current_shift_start and last_activity:
            shifts.append({
                'start_time': current_shift_start,
                'end_time': last_activity,
                'duration_hours': (last_activity - current_shift_start).total_seconds() / 3600
            })
        
        return shifts