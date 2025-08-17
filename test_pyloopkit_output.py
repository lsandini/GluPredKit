#!/usr/bin/env python3
"""
Test script to investigate the exact output format of pyloopkit's update() function
"""
from pyloopkit import loop_data_manager
from pyloopkit.dose import DoseType
import datetime
import json
import pprint

# Create a minimal test input similar to what loop_v2 uses
current_time = datetime.datetime.now()
test_input = {
    "dose_types": [DoseType.basal, DoseType.bolus],
    "dose_entries": [
        {
            "type": DoseType.basal,
            "start_time": current_time - datetime.timedelta(hours=2),
            "end_time": current_time - datetime.timedelta(hours=1, minutes=55),
            "value": 0.1,  # 0.1 units delivered over 5 minutes
            "unit": "U/hr",
            "scheduled_basal_rate": 1.0
        },
        {
            "type": DoseType.bolus,
            "start_time": current_time - datetime.timedelta(hours=1),
            "end_time": current_time - datetime.timedelta(hours=1),
            "value": 2.0,
            "unit": "U"
        }
    ],
    "carb_entries": [
        {
            "start_time": current_time - datetime.timedelta(hours=1, minutes=30),
            "carb_value": 30,
            "absorption_time": 180
        }
    ],
    "glucose_entries": [
        {
            "date": current_time - datetime.timedelta(minutes=15),
            "value": 120
        },
        {
            "date": current_time - datetime.timedelta(minutes=10),
            "value": 125
        },
        {
            "date": current_time - datetime.timedelta(minutes=5),
            "value": 130
        },
        {
            "date": current_time,
            "value": 135
        }
    ],
    "basal_rate_schedule": [(0, 1.0)],
    "sensitivity_schedule": [(0, 40)],
    "carb_ratio_schedule": [(0, 10)],
    "target_range_schedule": [(0, 100, 110)],
    "correction_range_schedule": [(0, 100, 110)],
    "suspend_threshold": 70,
    "max_basal_rate": 4.0,
    "max_bolus": 10,
    "retrospective_correction_enabled": True,
    "insulin_delay": 10,
    "now_date": current_time,
    "insulin_model": "humalog",
    "momentum_data_interval": 15,
    "default_absorption_times": [120, 150, 180, 240, 300],
    "max_history_age": 60 * 60 * 24,
    "carb_delay": 10,
    "retrospective_correction_grouping_interval": 30
}

print("=" * 80)
print("Testing pyloopkit.loop_data_manager.update() output format")
print("=" * 80)

try:
    # Call update() and examine the output
    output = loop_data_manager.update(test_input)
    
    print("\n1. Output type:", type(output))
    
    if output is None:
        print("   WARNING: Output is None!")
    elif isinstance(output, dict):
        print("\n2. Output keys:")
        for key in sorted(output.keys()):
            print(f"   - {key}")
        
        print("\n3. Detailed output structure:")
        for key, value in output.items():
            if value is None:
                print(f"\n   {key}: None")
            elif isinstance(value, (list, tuple)):
                print(f"\n   {key}: {type(value).__name__} with {len(value)} items")
                if len(value) > 0:
                    print(f"      First item type: {type(value[0])}")
                    if len(value) <= 3:
                        for i, item in enumerate(value):
                            print(f"      [{i}]: {item}")
                    else:
                        print(f"      [0]: {value[0]}")
                        print(f"      [1]: {value[1]}")
                        print(f"      ...")
                        print(f"      [-1]: {value[-1]}")
            elif isinstance(value, dict):
                print(f"\n   {key}: dict with {len(value)} keys")
                for k in list(value.keys())[:5]:
                    print(f"      - {k}")
            else:
                print(f"\n   {key}: {type(value).__name__} = {value}")
        
        # Look specifically for prediction-related fields
        print("\n4. Searching for prediction-related fields:")
        prediction_keywords = ['predict', 'glucose', 'forecast', 'future', 'projection']
        found_predictions = False
        
        for key in output.keys():
            for keyword in prediction_keywords:
                if keyword in key.lower():
                    print(f"\n   Found: {key}")
                    value = output[key]
                    if isinstance(value, (list, tuple)) and len(value) > 0:
                        print(f"      Type: {type(value).__name__} with {len(value)} items")
                        print(f"      First few values: {value[:5]}")
                        found_predictions = True
                    elif value is not None:
                        print(f"      Value: {value}")
                        found_predictions = True
        
        if not found_predictions:
            print("   No prediction-related fields found with common keywords")
    else:
        print(f"\n   Unexpected output type: {type(output)}")
        print(f"   Output: {output}")

except Exception as e:
    print(f"\nERROR calling update(): {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Now checking what the original loop.py model expects...")
print("=" * 80)

# Check the loop.py implementation
from glupredkit.models import loop
print("\nExamining loop.py model's _predict_model method...")
print("The loop.py model returns the full output dictionary from update()")
print("and expects the caller to extract predictions from it.")