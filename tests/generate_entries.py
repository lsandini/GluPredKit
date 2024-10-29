import json
from datetime import datetime, timedelta, timezone
import random
import math

def get_base_glucose(hour):
    """
    Returns base glucose value based on time of day.
    Implements typical patterns:
    - Dawn phenomenon (higher in early morning)
    - Slightly higher during day
    - Lower during night
    """
    if 2 <= hour < 7:  # Dawn phenomenon
        return 130 + (hour - 2) * 5  # Gradual increase
    elif 7 <= hour < 23:  # Daytime
        return 120
    else:  # Night
        return 100

def add_variation(base_value, hour):
    """
    Adds realistic variation to the base glucose value.
    - More variation during day
    - Less variation at night
    - Random walk component for realistic trends
    """
    if 7 <= hour < 23:  # Daytime
        variation = random.gauss(0, 10)  # More variation during day
    else:  # Night
        variation = random.gauss(0, 5)   # Less variation at night
    
    return max(40, min(400, base_value + variation))  # Clamp between 40 and 400

def generate_entries(start_date_str, end_date_str):
    """
    Generate glucose entries for testing the Nightscout parser.
    
    Args:
        start_date_str: Start date in format "YYYY-MM-DD"
        end_date_str: End date in format "YYYY-MM-DD"
    
    Returns:
        List of entries with 5-minute intervals
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    
    entries = []
    current_time = start_date
    last_value = 120  # Starting glucose value
    
    while current_time < end_date:
        hour = current_time.hour
        
        # Get base value for this time of day
        base_value = get_base_glucose(hour)
        
        # Add meal effects (breakfast, lunch, dinner)
        meal_effect = 0
        if hour in [7, 12, 18]:  # Meal times
            if current_time.minute < 30:  # First 30 minutes after meal
                meal_effect = 40 * (current_time.minute / 30)  # Gradual increase
        elif hour in [8, 13, 19]:  # Post-meal
            meal_effect = 40 * (1 - current_time.minute / 60)  # Gradual decrease
            
        # Calculate new value with smooth transition from last value
        target = base_value + meal_effect
        smoothing = 0.7  # Smoothing factor
        new_value = last_value * smoothing + target * (1 - smoothing)
        
        # Add some random variation
        glucose = round(add_variation(new_value, hour), 1)
        last_value = glucose
        
        timestamp_ms = int(current_time.timestamp() * 1000)
        
        entry = {
            "sgv": glucose,
            "type": "sgv",
            "dateString": current_time.strftime("%Y-%m-%dT%H:%M:00.000Z"),
            "date": timestamp_ms,
            "utcOffset": 0,
            "sysTime": current_time.strftime("%Y-%m-%dT%H:%M:00.000Z")
        }
        
        entries.append(entry)
        current_time += timedelta(minutes=5)
    
    return entries

def save_entries(entries, filename="nightscout_entries.json"):
    """Save entries to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(entries, f, indent=2)

def main():
    # Generate 2 days of data
    entries = generate_entries("2024-10-26", "2024-10-28")
    save_entries(entries)
    
    print(f"Generated {len(entries)} entries")
    print("\nFirst few entries:")
    for entry in entries[:5]:
        print(f"Time: {entry['dateString']}, SGV: {entry['sgv']}")
    print("\nLast few entries:")
    for entry in entries[-5:]:
        print(f"Time: {entry['dateString']}, SGV: {entry['sgv']}")
    print(f"\nSaved to nightscout_entries.json")

if __name__ == "__main__":
    main()