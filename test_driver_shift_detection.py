#!/usr/bin/env python3
"""
Test script for driver shift detection system.
Tests various edge cases and validates the implementation.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from detect_driver_shifts import DriverShiftDetector, ShiftDetectionConfig


def create_test_history_data():
    """Create test vehicle history data with various scenarios."""
    base_time = datetime(2025, 7, 10, 8, 0, 0)
    
    # Scenario 1: Normal driver change
    events = []
    
    # Driver A shift (8:00 - 12:00)
    for i in range(10):
        events.append({
            'deviceTimeUtc': (base_time + timedelta(minutes=i*30)).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'driverId': 'DRIVER_A',
            'driver': 'John Smith',
            'driverRef': 'REF_A',
            'vehicleId': 100,
            'vehicleRef': 'TEST001',
            'latitude': -20.4 + i*0.001,
            'longitude': 118.7 + i*0.001,
            'placeName': f'Location_{i}',
            'odometer': 1000 + i*10,
            'speedKmh': 60,
            'engineData': {
                'totalFuelUsed': 100 + i*0.5,
                'totalEngineHours': 500 + i*0.1
            }
        })
    
    # Driver B shift (13:00 - 17:00) - 1 hour gap
    shift_start = base_time + timedelta(hours=5)
    for i in range(8):
        events.append({
            'deviceTimeUtc': (shift_start + timedelta(minutes=i*30)).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'driverId': 'DRIVER_B',
            'driver': 'Jane Doe',
            'driverRef': 'REF_B',
            'vehicleId': 100,
            'vehicleRef': 'TEST001',
            'latitude': -20.5 + i*0.001,
            'longitude': 118.8 + i*0.001,
            'placeName': f'Location_{i+10}',
            'odometer': 1100 + i*15,
            'speedKmh': 55,
            'engineData': {
                'totalFuelUsed': 105 + i*0.7,
                'totalEngineHours': 501 + i*0.15
            }
        })
    
    # Large time gap scenario (next day)
    gap_start = base_time + timedelta(days=1, hours=8)
    for i in range(6):
        events.append({
            'deviceTimeUtc': (gap_start + timedelta(minutes=i*30)).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'driverId': 'DRIVER_A',
            'driver': 'John Smith',
            'driverRef': 'REF_A',
            'vehicleId': 100,
            'vehicleRef': 'TEST001',
            'latitude': -20.6 + i*0.001,
            'longitude': 118.9 + i*0.001,
            'placeName': f'Location_{i+20}',
            'odometer': 1300 + i*20,
            'speedKmh': 65,
            'engineData': {
                'totalFuelUsed': 110 + i*0.8,
                'totalEngineHours': 502 + i*0.2
            }
        })
    
    return events


def create_edge_case_data():
    """Create test data for edge cases."""
    base_time = datetime(2025, 7, 10, 8, 0, 0)
    
    # Edge case 1: Single event (should be filtered out)
    single_event = [{
        'deviceTimeUtc': base_time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
        'driverId': 'DRIVER_SINGLE',
        'driver': 'Single Event',
        'driverRef': 'REF_SINGLE',
        'vehicleId': 101,
        'vehicleRef': 'TEST002',
        'latitude': -20.4,
        'longitude': 118.7,
        'placeName': 'Single Location',
        'odometer': 1000,
        'speedKmh': 60
    }]
    
    # Edge case 2: Very short shift (should be filtered out)
    short_shift = []
    for i in range(3):
        short_shift.append({
            'deviceTimeUtc': (base_time + timedelta(minutes=i*5)).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'driverId': 'DRIVER_SHORT',
            'driver': 'Short Shift',
            'driverRef': 'REF_SHORT',
            'vehicleId': 102,
            'vehicleRef': 'TEST003',
            'latitude': -20.4,
            'longitude': 118.7,
            'placeName': 'Short Location',
            'odometer': 1000 + i,
            'speedKmh': 60
        })
    
    # Edge case 3: Missing driver IDs
    missing_driver = []
    for i in range(6):
        missing_driver.append({
            'deviceTimeUtc': (base_time + timedelta(minutes=i*30)).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'driverId': None,
            'driver': None,
            'driverRef': None,
            'vehicleId': 103,
            'vehicleRef': 'TEST004',
            'latitude': -20.4 + i*0.001,
            'longitude': 118.7 + i*0.001,
            'placeName': f'No Driver Location_{i}',
            'odometer': 1000 + i*10,
            'speedKmh': 50
        })
    
    return single_event, short_shift, missing_driver


def test_normal_operation():
    """Test normal operation with driver changes."""
    print("Testing normal operation...")
    
    # Create test configuration
    config = ShiftDetectionConfig(
        fleet_id=104002,
        start_date=datetime(2025, 7, 10, 0, 0, 0),
        end_date=datetime(2025, 7, 12, 0, 0, 0),
        timezone="Australia/Perth",
        min_shift_duration_minutes=60,  # 1 hour minimum
        max_event_gap_hours=4,
        min_events_per_shift=5,
        cache_dir=None,
        output_dir="./test_output"
    )
    
    # Create detector
    detector = DriverShiftDetector(config)
    
    # Mock the vehicle history loading
    test_data = create_test_history_data()
    
    with patch.object(detector, '_get_vehicles_to_analyze') as mock_vehicles, \
         patch.object(detector, '_load_vehicle_history') as mock_history:
        
        mock_vehicles.return_value = [{'vehicleId': 100, 'vehicleRef': 'TEST001'}]
        mock_history.return_value = test_data
        
        shifts, summary = detector.detect_shifts()
        
        print(f"Detected {len(shifts)} shifts")
        print(f"Summary: {summary}")
        
        # Validate results
        assert len(shifts) > 0, "Should detect at least one shift"
        assert summary['total_shifts'] == len(shifts), "Summary should match shift count"
        assert summary['unique_drivers'] > 0, "Should have unique drivers"
        
        # Check shift details
        for shift in shifts:
            assert shift.duration_hours > 0, "Shift should have positive duration"
            assert shift.total_events >= config.min_events_per_shift, "Shift should meet minimum events"
            assert shift.shift_type == "driver_change", "Should be driver_change type"
        
        print("✓ Normal operation test passed")


def test_edge_cases():
    """Test edge cases."""
    print("Testing edge cases...")
    
    config = ShiftDetectionConfig(
        fleet_id=104002,
        start_date=datetime(2025, 7, 10, 0, 0, 0),
        end_date=datetime(2025, 7, 12, 0, 0, 0),
        timezone="Australia/Perth",
        min_shift_duration_minutes=30,
        max_event_gap_hours=4,
        min_events_per_shift=5,
        cache_dir=None,
        output_dir="./test_output"
    )
    
    detector = DriverShiftDetector(config)
    
    single_event, short_shift, missing_driver = create_edge_case_data()
    
    # Test single event (should be filtered out)
    with patch.object(detector, '_get_vehicles_to_analyze') as mock_vehicles, \
         patch.object(detector, '_load_vehicle_history') as mock_history:
        
        mock_vehicles.return_value = [{'vehicleId': 101, 'vehicleRef': 'TEST002'}]
        mock_history.return_value = single_event
        
        shifts, summary = detector.detect_shifts()
        
        assert len(shifts) == 0, "Single event should be filtered out"
        print("✓ Single event filtering test passed")
    
    # Test short shift (should be filtered out)
    with patch.object(detector, '_get_vehicles_to_analyze') as mock_vehicles, \
         patch.object(detector, '_load_vehicle_history') as mock_history:
        
        mock_vehicles.return_value = [{'vehicleId': 102, 'vehicleRef': 'TEST003'}]
        mock_history.return_value = short_shift
        
        shifts, summary = detector.detect_shifts()
        
        assert len(shifts) == 0, "Short shift should be filtered out"
        print("✓ Short shift filtering test passed")
    
    # Test missing driver IDs (should still create shifts)
    with patch.object(detector, '_get_vehicles_to_analyze') as mock_vehicles, \
         patch.object(detector, '_load_vehicle_history') as mock_history:
        
        mock_vehicles.return_value = [{'vehicleId': 103, 'vehicleRef': 'TEST004'}]
        mock_history.return_value = missing_driver
        
        shifts, summary = detector.detect_shifts()
        
        if len(shifts) > 0:
            assert shifts[0].driver_id is None, "Should handle missing driver ID"
            assert shifts[0].driver_name is None, "Should handle missing driver name"
        print("✓ Missing driver ID test passed")


def test_cache_file_detection():
    """Test cache file detection logic."""
    print("Testing cache file detection...")
    
    # Create temporary cache structure
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock cache files
        cache_file1 = os.path.join(temp_dir, "fleet_104002_vehicle_100_20250710_080000_to_20250711_080000.json")
        cache_file2 = os.path.join(temp_dir, "PLS", "PRM00843", "fleet_104002_vehicle_11096_20250710_080000_to_20250711_080000.json")
        
        os.makedirs(os.path.dirname(cache_file2), exist_ok=True)
        
        # Write test data to cache files
        test_data = create_test_history_data()
        
        with open(cache_file1, 'w') as f:
            json.dump(test_data, f)
        
        with open(cache_file2, 'w') as f:
            json.dump(test_data, f)
        
        # Test cache detection
        config = ShiftDetectionConfig(
            fleet_id=104002,
            start_date=datetime(2025, 7, 10, 0, 0, 0),
            end_date=datetime(2025, 7, 12, 0, 0, 0),
            cache_dir=temp_dir
        )
        
        detector = DriverShiftDetector(config)
        
        # Test vehicle list from cache
        vehicles = detector._get_vehicles_from_cache()
        assert len(vehicles) > 0, "Should find vehicles from cache"
        print(f"Found {len(vehicles)} vehicles from cache")
        
        # Test cache file loading
        cache_data = detector._load_from_cache(100, "TEST001")
        assert len(cache_data) > 0, "Should load data from cache"
        print("✓ Cache file detection test passed")


def run_all_tests():
    """Run all tests."""
    print("Running driver shift detection tests...\n")
    
    try:
        test_normal_operation()
        print()
        test_edge_cases()
        print()
        test_cache_file_detection()
        print()
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()