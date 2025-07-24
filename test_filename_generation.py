#!/usr/bin/env python3
"""
Test script to demonstrate the new filename generation for driver shift reports.
"""

from datetime import datetime
from detect_driver_shifts import ShiftDetectionConfig, DriverShiftReporter


def test_filename_generation():
    """Test various filename generation scenarios."""
    print("Driver Shift Report Filename Generation Examples")
    print("=" * 60)
    
    # Test scenarios
    scenarios = [
        {
            "name": "Single Vehicle Analysis (Active Only)",
            "config": {
                "fleet_id": 104002,
                "start_date": datetime(2025, 7, 9),
                "end_date": datetime(2025, 7, 16),
                "vehicle_ref": "PRM00843",
                "include_null_driver_shifts": False,
                "output_dir": "./test_output"
            }
        },
        {
            "name": "Single Vehicle with Idle Periods",
            "config": {
                "fleet_id": 104002,
                "start_date": datetime(2025, 7, 9),
                "end_date": datetime(2025, 7, 16), 
                "vehicle_ref": "PRM00843",
                "include_null_driver_shifts": True,
                "output_dir": "./test_output"
            }
        },
        {
            "name": "Full Fleet Analysis (Active Only)",
            "config": {
                "fleet_id": 104002,
                "start_date": datetime(2025, 7, 9),
                "end_date": datetime(2025, 7, 16),
                "vehicle_ref": None,
                "include_null_driver_shifts": False,
                "output_dir": "./test_output"
            }
        },
        {
            "name": "Full Fleet with Idle Periods",
            "config": {
                "fleet_id": 104002,
                "start_date": datetime(2025, 7, 9),
                "end_date": datetime(2025, 7, 16),
                "vehicle_ref": None,
                "include_null_driver_shifts": True,
                "output_dir": "./test_output"
            }
        },
        {
            "name": "Single Day Analysis",
            "config": {
                "fleet_id": 104002,
                "start_date": datetime(2025, 7, 15),
                "end_date": datetime(2025, 7, 15),
                "vehicle_ref": "PRM00801",
                "include_null_driver_shifts": False,
                "output_dir": "./test_output"
            }
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print("-" * 40)
        
        # Create configuration
        config = ShiftDetectionConfig(**scenario['config'])
        
        # Create reporter
        reporter = DriverShiftReporter(config)
        
        # Generate example filenames
        csv_filename = reporter._generate_filename_context("driver_shifts_detailed", "csv")
        json_filename = reporter._generate_filename_context("driver_shifts_summary", "json")
        
        print(f"CSV:  {csv_filename.split('/')[-1]}")
        print(f"JSON: {json_filename.split('/')[-1]}")
    
    print("\n" + "=" * 60)
    print("Key Benefits of New Filename Format:")
    print("• Timestamps prevent overwriting previous reports")
    print("• Context includes all analysis parameters")
    print("• Easy to identify analysis scope from filename")
    print("• Chronological sorting by timestamp")
    print("• Clear distinction between active-only vs with-idle reports")


if __name__ == "__main__":
    test_filename_generation()