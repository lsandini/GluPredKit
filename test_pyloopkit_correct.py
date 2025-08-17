#!/usr/bin/env python3
"""
Test script to investigate the exact output format of pyloopkit's update() function
Using the correct input format from loop.py
"""
from pyloopkit.loop_data_manager import update
from pyloopkit.dose import DoseType
import datetime
import numpy as np

# Create test input using the correct format from loop.py
current_time = datetime.datetime.now()

# Build the input dictionary exactly as loop.py does
input_dict = {
    'carb_value_units': 'g',
    'settings_dictionary': {
        'model': [360, 75],  # DIA=360 minutes, peak=75 minutes
        'momentum_data_interval': 15.0,
        'suspend_threshold': None,
        'dynamic_carb_absorption_enabled': True,
        'retrospective_correction_integration_interval': 30,
        'recency_interval': 15,
        'retrospective_correction_grouping_interval': 30,
        'rate_rounder': 0.05,
        'insulin_delay': 10,
        'carb_delay': 0,
        'default_absorption_times': [120.0, 180.0, 240.0],
        'max_basal_rate': 2.5,
        'max_bolus': 12.0,
        'retrospective_correction_enabled': True
    },
    'sensitivity_ratio_start_times': [datetime.time(0, 0)],
    'sensitivity_ratio_end_times': [datetime.time(0, 0)],
    'sensitivity_ratio_values': [40],  # ISF = 40 mg/dL/U
    'sensitivity_ratio_value_units': 'mg/dL/U',
    'carb_ratio_start_times': [datetime.time(0, 0)],
    'carb_ratio_values': [10],  # CR = 10 g/U
    'carb_ratio_value_units': 'g/U',
    'basal_rate_start_times': [datetime.time(0, 0)],
    'basal_rate_minutes': [1440],
    'basal_rate_values': [1.0],  # 1.0 U/hr
    'basal_rate_value_units': 'U/hour',
    'target_range_start_times': [datetime.time(0, 0)],
    'target_range_end_times': [datetime.time(0, 0)],
    'target_range_minimum_values': [100],
    'target_range_maximum_values': [110],
    'target_range_value_units': 'mg/dL',
    'time_to_calculate_at': current_time,
    
    # Add glucose data
    'glucose_dates': [
        current_time - datetime.timedelta(minutes=15),
        current_time - datetime.timedelta(minutes=10),
        current_time - datetime.timedelta(minutes=5),
        current_time
    ],
    'glucose_values': [120, 125, 130, 135],
    
    # Add dose data
    'dose_types': [DoseType.bolus, DoseType.basal],
    'dose_start_times': [
        current_time - datetime.timedelta(hours=1),
        current_time - datetime.timedelta(minutes=30)
    ],
    'dose_end_times': [
        current_time - datetime.timedelta(hours=1),
        current_time - datetime.timedelta(minutes=25)
    ],
    'dose_values': [2.0, 1.0],  # 2U bolus, 1.0 U/hr basal
    'dose_delivered_units': [None, None],  # PyLoopKit expects None, not strings
    
    # Add carb data
    'carb_dates': [current_time - datetime.timedelta(hours=1, minutes=30)],
    'carb_values': [30],
    'carb_absorption_times': [180]
}

print("=" * 80)
print("Testing pyloopkit.loop_data_manager.update() with correct input format")
print("=" * 80)

try:
    # Call update() and examine the output
    output = update(input_dict)
    
    print("\n1. Output type:", type(output))
    
    if output is None:
        print("   WARNING: Output is None!")
    elif isinstance(output, dict):
        print("\n2. Output keys:")
        for key in sorted(output.keys()):
            print(f"   - {key}")
        
        # Look for prediction fields
        print("\n3. Prediction-related fields:")
        if "predicted_glucose_values" in output:
            values = output["predicted_glucose_values"]
            print(f"   predicted_glucose_values: {type(values).__name__} with {len(values)} items")
            if len(values) > 0:
                print(f"      First 5 values: {values[:5]}")
                print(f"      Last value: {values[-1]}")
        
        if "predicted_glucose_dates" in output:
            dates = output["predicted_glucose_dates"]
            print(f"   predicted_glucose_dates: {type(dates).__name__} with {len(dates)} items")
            if len(dates) > 0:
                print(f"      First date: {dates[0]}")
                print(f"      Last date: {dates[-1]}")
        
        # Check other important fields
        print("\n4. Other important fields:")
        for key in ["input_time", "insulin_on_board", "carbs_on_board", 
                    "recommended_bolus", "recommended_temp_basal"]:
            if key in output:
                value = output[key]
                if isinstance(value, (list, tuple)):
                    print(f"   {key}: {type(value).__name__} with {len(value)} items")
                else:
                    print(f"   {key}: {value}")
        
        print("\n5. Full output keys and types:")
        for key, value in output.items():
            if value is None:
                print(f"   {key}: None")
            elif isinstance(value, (list, tuple)):
                print(f"   {key}: {type(value).__name__}[{len(value)}]")
            elif isinstance(value, dict):
                print(f"   {key}: dict[{len(value)} keys]")
            else:
                print(f"   {key}: {type(value).__name__}")
    else:
        print(f"\n   Unexpected output type: {type(output)}")

except Exception as e:
    print(f"\nERROR calling update(): {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)
print("The loop.py model expects output['predicted_glucose_values'] to contain predictions")
print("Element [0] is the reference value (current glucose)")
print("Elements [1:] are the actual predictions")