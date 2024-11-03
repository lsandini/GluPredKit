import requests
from aiohttp import ClientError, ClientConnectorError, ClientResponseError
import nightscout
from .base_parser import BaseParser
import pandas as pd
import datetime
import json
import os
import urllib.parse

# Monkey patch the Treatment class, so no need to modify models.py in nightscout package
from nightscout.models import Treatment

original_init = Treatment.__init__

def new_init(self, **kwargs):
    # Set up param_defaults with all possible attributes, including new ones for Loop/Trio
    self.param_defaults = {
        'temp': None,
        'enteredBy': None,
        'eventType': None,
        'glucose': None,
        'glucoseType': None,
        'units': None,
        'device': None,
        'created_at': None,
        'timestamp': None,
        'absolute': None,
        'percent': None,
        'percentage': None,
        'rate': None,
        'duration': None,
        'carbs': None,
        'insulin': None,
        'unabsorbed': None,
        'suspended': None,
        'type': None,
        'programmed': None,
        'foodType': None,
        'absorptionTime': None,
        'profile': None,
        'insulinNeedsScaleFactor': None,  # Added for Loop/Trio
        'reason': None,  # Added for Loop/Trio
        'automatic': None  # Added for Loop/Trio
    }
    # Set all default attributes
    for (param, default) in self.param_defaults.items():
        setattr(self, param, kwargs.get(param, default))
    
    # Call original init
    original_init(self, **kwargs)

# Replace the __init__ method
Treatment.__init__ = new_init

class Parser(BaseParser):
    def __init__(self):
        super().__init__()
        self.test_mode = False
        self.test_data_dir = None

    def __call__(self, start_date, end_date, username: str, password: str, test_mode=False, test_data_dir=None):
        """
        Main method to parse Nightscout data.
        In the nightscout parser, the username is the nightscout URL, and the password is the API key.
        """
        try:
            if test_mode:
                # Load test data from local files
                from pathlib import Path
                test_dir = Path(test_data_dir)

                # Load profiles
                with open(test_dir / "nightscout_profiles.json") as f:
                    profiles = json.load(f)

                # Load treatments
                with open(test_dir / "nightscout_treatments.json") as f:
                    treatments_data = json.load(f)
                treatments = []
                for t in treatments_data:
                    treatment = type('Treatment', (), {})()
                    for k, v in t.items():
                        setattr(treatment, k, v)
                    treatments.append(treatment)

                # Load entries
                with open(test_dir / "nightscout_entries.json") as f:
                    entries_data = json.load(f)
                entries = []
                for e in entries_data:
                    entry = type('Entry', (), {})()
                    for k, v in e.items():
                        setattr(entry, k, v)
                    entries.append(entry)

            else:
                api = nightscout.Api(username, api_secret=password)
                api_start_date = start_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
                api_end_date = end_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')

                # Fetch and save profiles data
                profile_query_params = {
                    'count': 0,
                    'find[created_at][$gte]': api_start_date,
                    'find[created_at][$lte]': api_end_date
                }
                base_url = username.rstrip('/')
                profile_url = f"{base_url}/api/v1/profile?{urllib.parse.urlencode(profile_query_params)}"
                profiles_response = requests.get(profile_url, headers=api.request_headers())
                profiles = profiles_response.json()
                self.save_json_profiles(profiles, 'profiles', api_start_date, api_end_date)

                # Fetch treatments
                treatment_query_params = {'count': 0, 'find[created_at][$gte]': api_start_date, 
                                        'find[created_at][$lte]': api_end_date}
                treatments = api.get_treatments(treatment_query_params)
                self.save_json(treatments, 'treatments', api_start_date, api_end_date)

                # Fetch entries
                query_params = {'count': 0, 'find[dateString][$gte]': api_start_date,
                              'find[dateString][$lte]': api_end_date}
                entries = api.get_sgvs(query_params)
                self.save_json(entries, 'entries', api_start_date, api_end_date)

            # Create DataFrame for Glucose (SGV) [mg/dL]
            df_glucose = self.create_dataframe(entries, 'date', 'sgv', 'CGM')
            print("Glucose DataFrame:")
            print(df_glucose)

            # Create DataFrame for Carbs
            df_carbs = self.create_dataframe(treatments, 'created_at', 'carbs', 'carbs', 
                                           event_type=['Carb Correction', 'Meal Bolus', 'Snack Bolus'])
            print("Carbs DataFrame:")
            print(df_carbs)

            # Create DataFrame for Bolus insulin
            df_bolus = self.create_dataframe(treatments, 'created_at', 'insulin', 'bolus', 
                                           event_type=['Bolus', 'Meal Bolus', 'Snack Bolus', 
                                                     'Correction Bolus', 'SMB'])
            print("Bolus DataFrame:")
            print(df_bolus)

            # Create DataFrame for temporary basal rates and durations
            df_temp_basal = self.create_dataframe(treatments, 'created_at', ['absolute', 'rate'], 
                                                'basal', event_type='Temp Basal')
            df_temp_duration = self.create_dataframe(treatments, 'created_at', 'duration', 
                                                   'duration', event_type='Temp Basal')
            print("Temporary Basal DataFrame:")
            print(df_temp_basal)

            # Get profile-based basal rates and create basal DataFrame
            basal_rates = self.get_basal_rates_from_profile(profiles)
            df_basal_profile = self.create_basal_dataframe([start_date, end_date], basal_rates)
            
            # Create and apply profile switches
            df_profile_switches = self.create_profile_switches_df(treatments)
            df_basal_profile = self.apply_profile_switches(df_basal_profile, df_profile_switches, profiles)
            
            print("Profile Basal DataFrame after switches:")
            print(df_basal_profile)

            # Resampling all datatypes into the same time-grid
            df = df_glucose.resample('5min').mean().fillna(0)
            df = self.merge_and_process(df, df_carbs, 'carbs')
            df = self.merge_and_process(df, df_bolus, 'bolus')

            # Merge profile basals and temporary basals
            df = self.merge_basal_rates(df, df_basal_profile, df_temp_basal, df_temp_duration)

            # Process basal rate
            df['basal'] = round(df['basal'] / 60 * 5, 5)  # From U/hr to U (5-minutes)
            df['basal'] = df['basal'].fillna(0)  # Ensure no NaN in basal
            df['bolus'] = df['bolus'].fillna(0)  # Ensure no NaN in bolus
            df['insulin'] = df['bolus'] + df['basal']  # Calculate total insulin

            # Convert timezone to local timezone
            current_timezone = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
            df.index = df.index.tz_convert(current_timezone)

            # Add hour of day and id column
            df['hour'] = df.index.hour
            df['id'] = 1
            df['is_test'] = False

            # Reorder columns to match the desired output exactly
            df = df.reset_index()
            df = df[['date', 'id', 'CGM', 'insulin', 'carbs', 'is_test', 'hour', 'basal', 'bolus']]
            
            # Fill any remaining NaN values with 0 before setting final index
            for col in ['CGM', 'insulin', 'carbs', 'basal', 'bolus']:
                df[col] = df[col].fillna(0)
                
            df.set_index('date', inplace=True)

            print("Final DataFrame:")
            print(df)
            
            # Add verification of treatments
            df = self.verify_treatments(treatments, df)

            return df

        except ClientResponseError as error:
            raise RuntimeError("Received ClientResponseError") from error
        except (ClientError, ClientConnectorError, TimeoutError, OSError) as error:
            raise RuntimeError("Received client error or timeout. Make sure that the username (nightscout URL) and "
                               "password (API key) is correct.") from error
        except Exception as e:
            raise RuntimeError(f"Error fetching data: {str(e)}")

    def create_profile_switches_df(self, treatments):
        """Create DataFrame for profile switches and temporary overrides."""
        switches = []
        for treatment in treatments:
            if not hasattr(treatment, 'eventType'):
                continue
                
            switch = {
                'date': pd.to_datetime(treatment.created_at, utc=True),
                'profile': None,
                'scale_factor': 1.0,  # Default no scaling
                'duration': None,
                'source': None
            }
            
            # Handle AndroidAPS Profile Switches
            if treatment.eventType == 'Profile Switch':
                switch.update({
                    'profile': getattr(treatment, 'profile', None),
                    'duration': getattr(treatment, 'duration', None),
                    'source': 'AndroidAPS'
                })
                if switch['profile'] is not None:
                    switches.append(switch)
                
            # Handle Loop/Trio Temporary Overrides
            elif treatment.eventType == 'Temporary Override':
                scale_factor = getattr(treatment, 'insulinNeedsScaleFactor', None)
                if scale_factor is not None:
                    switch.update({
                        'scale_factor': float(scale_factor),
                        'duration': getattr(treatment, 'duration', None),
                        'source': 'Loop' if 'Loop' in getattr(treatment, 'enteredBy', '') else 'Trio'
                    })
                    switches.append(switch)
        
        if not switches:
            return pd.DataFrame(columns=['profile', 'scale_factor', 'duration', 'source'])
        
        df = pd.DataFrame(switches)
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        return df

    def apply_profile_switches(self, df_basal, profile_switches, profiles):
        """Apply profile switches and temporary overrides to basal rates."""
        if profile_switches.empty:
            return df_basal
        
        store = profiles[0].get('store', {})
        result_df = df_basal.copy()
        
        # Process each switch/override chronologically
        for time, row in profile_switches.iterrows():
            # Calculate end time
            if pd.notnull(row['duration']):
                end_time = time + pd.Timedelta(minutes=float(row['duration']))
            else:
                # If no duration, effective until next switch or end of data
                later_switches = profile_switches.index[profile_switches.index > time]
                end_time = later_switches[0] if len(later_switches) > 0 else df_basal.index[-1]
            
            # Create mask for the time period
            mask = (result_df.index >= time) & (result_df.index < end_time)
            
            if row['source'] == 'AndroidAPS':
                # Handle AndroidAPS Profile Switch
                if row['profile'] and row['profile'] in store:
                    new_profile = store[row['profile']]
                    new_basal_rates = []
                    for entry in new_profile.get('basal', []):
                        seconds = entry.get('timeAsSeconds', 0)
                        rate = entry.get('value', 0)
                        new_basal_rates.append((seconds, rate))
                    new_basal_rates.sort()
                    
                    # Update each timestep in the period
                    for idx in result_df[mask].index:
                        seconds = (idx.hour * 3600 + idx.minute * 60 + idx.second)
                        rate = self.get_basal_rate_for_time(new_basal_rates, seconds)
                        result_df.loc[idx, 'basal'] = rate
                        
            elif row['source'] in ['Loop', 'Trio']:
                # Handle Loop/Trio Temporary Override
                result_df.loc[mask, 'basal'] *= row['scale_factor']
        
        return result_df

    def merge_basal_rates(self, df, df_profile_basal, df_temp_basal, df_temp_duration):
        """Merge profile basal rates with temporary basal overrides."""
        # Start with profile basal rates
        df = df.merge(df_profile_basal, left_index=True, right_index=True, how='left')
        df['basal'] = df['basal'].fillna(0)
        
        if not df_temp_basal.empty:
            temp_basals = pd.concat([df_temp_basal, df_temp_duration], axis=1)
            temp_basals.columns = ['basal', 'percent_x', 'duration', 'percent_y']
            
            for time, row in temp_basals.iterrows():
                end_time = time + pd.Timedelta(minutes=float(row['duration']))
                mask = (df.index >= time) & (df.index < end_time)
                
                basal_value = float(row['basal']) if pd.notnull(row['basal']) else 0
                percent_value = float(row['percent_x']) if pd.notnull(row['percent_x']) else 100
                
                if basal_value > 0:  # Absolute temp basal
                    df.loc[mask, 'basal'] = basal_value
                elif percent_value != 100:  # Percentage temp basal
                    current_rates = df.loc[mask, 'basal']
                    new_rates = current_rates * (percent_value / 100)
                    df.loc[mask, 'basal'] = new_rates
        
        df['basal'] = df['basal'].fillna(0)
        return df

    def get_basal_rates_from_profile(self, profiles):
            """
            Extract basal rates from the default profile in the profiles data.
            Returns a list of tuples (time_seconds, rate).
            """
            if not profiles or len(profiles) == 0:
                return []
                
            # Get the default profile name
            default_profile_name = profiles[0].get('defaultProfile')
            if not default_profile_name:
                return []
                
            # Get the store containing all profiles
            store = profiles[0].get('store', {})
            if not store or default_profile_name not in store:
                return []
                
            # Get the basal schedule from the default profile
            default_profile = store[default_profile_name]
            basal_schedule = default_profile.get('basal', [])
            
            def time_to_seconds(time_str):
                """Convert time string (HH:MM) to seconds since midnight."""
                hours, minutes = map(int, time_str.split(':'))
                return hours * 3600 + minutes * 60
            
            # Convert to list of (seconds, rate) tuples, calculating seconds if needed
            basal_rates = []
            for entry in basal_schedule:
                seconds = entry.get('timeAsSeconds', None)
                if seconds is None:
                    seconds = time_to_seconds(entry['time'])
                rate = entry.get('value', 0)  # Default to 0 if no value provided
                basal_rates.append((seconds, rate))
            
            return sorted(basal_rates)

    def get_basal_rate_for_time(self, basal_rates, seconds_since_midnight):
            """
            Get the appropriate basal rate for a given time (in seconds since midnight).
            """
            if not basal_rates:
                return 0.0
                
            # Find the last basal rate that started before or at this time
            applicable_rate = basal_rates[0][1]  # Default to first rate
            for time_sec, rate in basal_rates:
                if time_sec <= seconds_since_midnight:
                    applicable_rate = rate
                else:
                    break
            return applicable_rate

    def create_basal_dataframe(self, date_range, basal_rates):
            """
            Create a DataFrame with basal rates for every 5 minutes in the date range.
            """
            dates = []
            rates = []
            
            current_date = date_range[0]
            while current_date <= date_range[1]:
                seconds = (current_date.hour * 3600 + 
                        current_date.minute * 60 + 
                        current_date.second)
                
                rate = round(self.get_basal_rate_for_time(basal_rates, seconds), 5)
                
                dates.append(current_date)
                rates.append(rate)
                
                current_date += datetime.timedelta(minutes=5)
            
            df = pd.DataFrame({'date': dates, 'basal': rates})
            df['basal'] = df['basal'].fillna(0)  # Fill any NaN basal rates with 0
            df['date'] = pd.to_datetime(df['date'], utc=True)
            df.set_index('date', inplace=True)
            return df

    def create_dataframe(self, data, date_column, value_column, new_column_name, event_type=None):
        """
        Create a DataFrame from the given data.
        Also extract percent for temp basals when available.
        """
        dates = []
        values = []
        percents = []
        
        for entry in data:
            try:
                if event_type:
                    # Handle treatments (insulin, carbs, etc.)
                    if isinstance(event_type, list):
                        if any(et in entry.eventType for et in event_type):
                            dates.append(pd.to_datetime(getattr(entry, date_column), utc=True))
                            if isinstance(value_column, list):
                                value = getattr(entry, value_column[0], None)
                                if value is None or pd.isna(value):
                                    value = getattr(entry, value_column[1], 0)
                            else:
                                value = getattr(entry, value_column, 0)
                            values.append(value if not pd.isna(value) else 0)
                            percents.append(getattr(entry, 'percent', 0))
                    elif event_type in entry.eventType:
                        dates.append(pd.to_datetime(getattr(entry, date_column), utc=True))
                        if isinstance(value_column, list):
                            value = getattr(entry, value_column[0], None)
                            if value is None or pd.isna(value):
                                value = getattr(entry, value_column[1], 0)
                        else:
                            value = getattr(entry, value_column, 0)
                        values.append(value if not pd.isna(value) else 0)
                        percents.append(getattr(entry, 'percent', 0))
                else:
                    # Handle entries (glucose values)
                    # Try different date attributes
                    if hasattr(entry, 'dateString'):
                        date_value = pd.to_datetime(entry.dateString, utc=True)
                    elif hasattr(entry, 'date'):
                        # Handle both string dates and millisecond timestamps
                        try:
                            date_value = pd.to_datetime(entry.date, utc=True)
                        except (TypeError, ValueError):
                            # If it's a millisecond timestamp
                            date_value = pd.to_datetime(entry.date, unit='ms', utc=True)
                    else:
                        raise AttributeError(f"No valid date field found in entry")

                    dates.append(date_value)
                    value = getattr(entry, value_column, 0)
                    values.append(value if not pd.isna(value) else 0)
                    percents.append(0)
            
            except Exception as e:
                print(f"Error processing entry: {entry}")
                print(f"Error details: {str(e)}")
                raise
        
        df = pd.DataFrame({
            'date': dates,
            new_column_name: values,
            'percent': percents
        })
        
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        return df

    def merge_and_process(self, df, df_to_merge, column_name):
            """
            Merge and process dataframes with better handling of bolus data and detailed logging.
            """
            if not df_to_merge.empty:
                print(f"\nBefore resampling {column_name} data:")
                print(f"Original {column_name} data:")
                print(df_to_merge)
                
                # Convert index to exact 5-minute marks for better alignment
                df_to_merge.index = df_to_merge.index.round('5min')
                print(f"\nAfter rounding to 5min intervals:")
                print(df_to_merge)
                
                # Fill NaN values before resampling
                df_to_merge[column_name] = df_to_merge[column_name].fillna(0)
                
                # For bolus data, use last() instead of sum() to preserve individual boluses
                if column_name == 'bolus':
                    df_to_merge = df_to_merge.resample('5min').last().fillna(0)
                else:
                    df_to_merge = df_to_merge.resample('5min').sum().fillna(0)
                    
                print(f"\nAfter resampling:")
                print(df_to_merge)
                
                # Merge with original dataframe
                print("\nMerging with main dataframe...")
                print(f"Main df times: {df.index.min()} to {df.index.max()}")
                print(f"{column_name} df times: {df_to_merge.index.min()} to {df_to_merge.index.max()}")
                
                df = pd.merge(df, df_to_merge, left_index=True, right_index=True, how='outer')
                
                # Fill all NaN values with 0 after merge
                df[column_name] = df[column_name].fillna(0)
                
                print(f"\nAfter merge and fillna:")
                print(df[df[column_name] > 0])  # Show only rows where we have values
                
            else:
                print(f"\nNo {column_name} data to merge")
                df[column_name] = 0
            
            return df

    def verify_treatments(self, treatments, final_df):
            """
            Verify that all treatments are properly captured in the final dataframe.
            """
            print("\nVerifying treatments capture:")
            
            for treatment in treatments:
                treatment_time = pd.to_datetime(treatment.created_at).tz_convert(final_df.index.tz)
                rounded_time = treatment_time.round('5min')
                
                if hasattr(treatment, 'insulin') and treatment.insulin:
                    insulin_value = float(treatment.insulin) if not pd.isna(treatment.insulin) else 0
                    print(f"\nChecking insulin treatment:")
                    print(f"Treatment time: {treatment_time}")
                    print(f"Rounded time: {rounded_time}")
                    print(f"Treatment insulin: {insulin_value}")
                    if rounded_time in final_df.index:
                        print(f"Found in df: {final_df.loc[rounded_time, 'bolus']}")
                    else:
                        print("Time not found in final dataframe!")
                
                if hasattr(treatment, 'carbs') and treatment.carbs:
                    carbs_value = float(treatment.carbs) if not pd.isna(treatment.carbs) else 0
                    print(f"\nChecking carb treatment:")
                    print(f"Treatment time: {treatment_time}")
                    print(f"Rounded time: {rounded_time}")
                    print(f"Treatment carbs: {carbs_value}")
                    if rounded_time in final_df.index:
                        print(f"Found in df: {final_df.loc[rounded_time, 'carbs']}")
                    else:
                        print("Time not found in final dataframe!")

            return final_df
    
    
    def save_json(self, data, data_type, start_date, end_date):
        """
        Save raw data to JSON file.
        """
        os.makedirs('data/raw', exist_ok=True)
        filename = f'data/raw/{data_type}_{start_date}_{end_date}.json'
        with open(filename, 'w') as f:
            json.dump([self.entry_to_dict(entry) for entry in data], f, indent=2, default=str)

    def save_json_profiles(self, profiles, data_type, start_date, end_date):
        """
        Save profiles data to JSON file.
        """
        os.makedirs('data/raw', exist_ok=True)
        filename = f'data/raw/{data_type}_{start_date}_{end_date}.json'
        with open(filename, 'w') as f:
            json.dump(profiles, f, indent=2)


    def entry_to_dict(self, entry):
        """
        Convert entry object to dictionary with only essential fields.
        For both entries and treatments.
        """
        if hasattr(entry, '_json'):
            # If it's an API response object (SGV or Treatment), return the original JSON
            return entry._json
        elif hasattr(entry, '__dict__'):
            # For test data, get all attributes except internal ones
            data = {}
            for key, value in entry.__dict__.items():
                if not key.startswith('_'):  # Skip internal attributes
                    data[key] = value
            return data
        return dict(entry)