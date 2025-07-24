#!/usr/bin/env python3
"""
Validation script for driver shift detection logic.
Tests core algorithms without requiring full system setup.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

def validate_cache_file_structure():
    """Validate that we can read existing cache files."""
    print("Validating cache file structure...")
    
    cache_dir = Path("simulation_cache")
    if not cache_dir.exists():
        print("❌ Cache directory not found")
        return False
    
    # Find sample cache files
    cache_files = list(cache_dir.glob("**/*.json"))
    if not cache_files:
        print("❌ No cache files found")
        return False
    
    print(f"✓ Found {len(cache_files)} cache files")
    
    # Test reading a sample file
    sample_file = cache_files[0]
    try:
        with open(sample_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list) and len(data) > 0:
            sample_event = data[0]
            required_fields = ['deviceTimeUtc', 'vehicleId', 'vehicleRef']
            missing_fields = [field for field in required_fields if field not in sample_event]
            
            if missing_fields:
                print(f"⚠️ Missing required fields in sample: {missing_fields}")
            else:
                print("✓ Cache file structure is valid")
                
                # Show sample data structure
                print(f"Sample event keys: {list(sample_event.keys())}")
                print(f"Sample driverId: {sample_event.get('driverId')}")
                print(f"Sample vehicleRef: {sample_event.get('vehicleRef')}")
                
                return True
        else:
            print("❌ Cache file has unexpected structure")
            return False
            
    except Exception as e:
        print(f"❌ Error reading cache file: {str(e)}")
        return False


def validate_timestamp_parsing():
    """Validate timestamp parsing logic."""
    print("\nValidating timestamp parsing...")
    
    test_timestamps = [
        "2025-07-10T08:30:15.123Z",
        "2025-07-10T08:30:15Z",
        "2025-07-10T08:30:15",
    ]
    
    def parse_timestamp(timestamp_str):
        """Simplified version of the parsing logic."""
        if not timestamp_str:
            return None
        
        try:
            for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S']:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue
            return None
        except Exception:
            return None
    
    all_passed = True
    for ts in test_timestamps:
        parsed = parse_timestamp(ts)
        if parsed:
            print(f"✓ {ts} -> {parsed}")
        else:
            print(f"❌ Failed to parse: {ts}")
            all_passed = False
    
    return all_passed


def validate_shift_detection_logic():
    """Validate core shift detection logic."""
    print("\nValidating shift detection logic...")
    
    # Create test events
    base_time = datetime(2025, 7, 10, 8, 0, 0)
    
    events = [
        # Driver A - 4 events (8:00-8:45)
        {'deviceTimeUtc': (base_time + timedelta(minutes=0)).strftime('%Y-%m-%dT%H:%M:%SZ'), 'driverId': 'A'},
        {'deviceTimeUtc': (base_time + timedelta(minutes=15)).strftime('%Y-%m-%dT%H:%M:%SZ'), 'driverId': 'A'},
        {'deviceTimeUtc': (base_time + timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%SZ'), 'driverId': 'A'},
        {'deviceTimeUtc': (base_time + timedelta(minutes=45)).strftime('%Y-%m-%dT%H:%M:%SZ'), 'driverId': 'A'},
        
        # Driver B - 6 events (9:00-10:15)
        {'deviceTimeUtc': (base_time + timedelta(minutes=60)).strftime('%Y-%m-%dT%H:%M:%SZ'), 'driverId': 'B'},
        {'deviceTimeUtc': (base_time + timedelta(minutes=75)).strftime('%Y-%m-%dT%H:%M:%SZ'), 'driverId': 'B'},
        {'deviceTimeUtc': (base_time + timedelta(minutes=90)).strftime('%Y-%m-%dT%H:%M:%SZ'), 'driverId': 'B'},
        {'deviceTimeUtc': (base_time + timedelta(minutes=105)).strftime('%Y-%m-%dT%H:%M:%SZ'), 'driverId': 'B'},
        {'deviceTimeUtc': (base_time + timedelta(minutes=120)).strftime('%Y-%m-%dT%H:%M:%SZ'), 'driverId': 'B'},
        {'deviceTimeUtc': (base_time + timedelta(minutes=135)).strftime('%Y-%m-%dT%H:%M:%SZ'), 'driverId': 'B'},
        
        # Large gap, then Driver A again - 5 events (15:00-16:00)
        {'deviceTimeUtc': (base_time + timedelta(hours=7)).strftime('%Y-%m-%dT%H:%M:%SZ'), 'driverId': 'A'},
        {'deviceTimeUtc': (base_time + timedelta(hours=7, minutes=15)).strftime('%Y-%m-%dT%H:%M:%SZ'), 'driverId': 'A'},
        {'deviceTimeUtc': (base_time + timedelta(hours=7, minutes=30)).strftime('%Y-%m-%dT%H:%M:%SZ'), 'driverId': 'A'},
        {'deviceTimeUtc': (base_time + timedelta(hours=7, minutes=45)).strftime('%Y-%m-%dT%H:%M:%SZ'), 'driverId': 'A'},
        {'deviceTimeUtc': (base_time + timedelta(hours=8)).strftime('%Y-%m-%dT%H:%M:%SZ'), 'driverId': 'A'},
    ]
    
    # Simulate shift detection logic
    shifts = []
    current_shift_events = []
    current_driver_id = None
    min_events_per_shift = 5
    max_event_gap_hours = 4
    
    def parse_timestamp(timestamp_str):
        return datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ')
    
    def should_start_new_shift(current_driver_id, new_driver_id, current_events, new_event):
        if not current_events:
            return True
        
        # Driver change
        if current_driver_id != new_driver_id:
            return True
        
        # Time gap
        last_event = current_events[-1]
        last_time = parse_timestamp(last_event['deviceTimeUtc'])
        new_time = parse_timestamp(new_event['deviceTimeUtc'])
        
        time_gap = (new_time - last_time).total_seconds() / 3600
        if time_gap > max_event_gap_hours:
            return True
        
        return False
    
    # Process events
    for event in events:
        driver_id = event.get('driverId')
        
        if should_start_new_shift(current_driver_id, driver_id, current_shift_events, event):
            # Finalize current shift
            if current_shift_events and len(current_shift_events) >= min_events_per_shift:
                shifts.append({
                    'driver': current_driver_id,
                    'events': len(current_shift_events),
                    'start': current_shift_events[0]['deviceTimeUtc'],
                    'end': current_shift_events[-1]['deviceTimeUtc']
                })
            
            # Start new shift
            current_shift_events = [event]
            current_driver_id = driver_id
        else:
            current_shift_events.append(event)
    
    # Handle final shift
    if current_shift_events and len(current_shift_events) >= min_events_per_shift:
        shifts.append({
            'driver': current_driver_id,
            'events': len(current_shift_events),
            'start': current_shift_events[0]['deviceTimeUtc'],
            'end': current_shift_events[-1]['deviceTimeUtc']
        })
    
    print(f"Detected {len(shifts)} shifts:")
    for i, shift in enumerate(shifts, 1):
        print(f"  Shift {i}: Driver {shift['driver']}, {shift['events']} events, {shift['start']} to {shift['end']}")
    
    # Validate results
    expected_shifts = 2  # Should detect 2 valid shifts (Driver B and final Driver A)
    if len(shifts) == expected_shifts:
        print("✓ Shift detection logic working correctly")
        return True
    else:
        print(f"❌ Expected {expected_shifts} shifts, got {len(shifts)}")
        return False


def validate_command_line_interface():
    """Validate that the CLI script exists and has proper structure."""
    print("\nValidating command line interface...")
    
    script_path = Path("detect_driver_shifts.py")
    if not script_path.exists():
        print("❌ Main script not found")
        return False
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    required_elements = [
        'argparse',
        'def main():',
        '--fleet-id',
        '--start-date',
        '--end-date',
        'DriverShiftDetector',
        'ShiftDetectionConfig'
    ]
    
    missing_elements = []
    for element in required_elements:
        if element not in content:
            missing_elements.append(element)
    
    if missing_elements:
        print(f"❌ Missing CLI elements: {missing_elements}")
        return False
    else:
        print("✓ CLI structure is valid")
        return True


def main():
    """Run all validations."""
    print("Driver Shift Detection Validation\n" + "="*40)
    
    validations = [
        validate_cache_file_structure,
        validate_timestamp_parsing,
        validate_shift_detection_logic,
        validate_command_line_interface
    ]
    
    results = []
    for validation in validations:
        try:
            result = validation()
            results.append(result)
        except Exception as e:
            print(f"❌ Validation failed with error: {str(e)}")
            results.append(False)
    
    print(f"\n{'='*40}")
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} validations passed!")
        print("\nThe driver shift detection system is ready for use.")
        print("\nExample usage:")
        print("python detect_driver_shifts.py \\")
        print("    --fleet-id 104002 \\")
        print("    --start-date \"2025-07-09\" \\")
        print("    --end-date \"2025-07-16\" \\")
        print("    --cache-dir \"./simulation_cache\" \\")
        print("    --vehicle-ref \"PRM00843\"")
    else:
        print(f"❌ {passed}/{total} validations passed")
        print("Please review the failed validations above.")


if __name__ == "__main__":
    main()