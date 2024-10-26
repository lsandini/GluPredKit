import requests
from aiohttp import ClientError, ClientConnectorError, ClientResponseError
import nightscout
from .base_parser import BaseParser
import pandas as pd
import datetime
import json
import os
import urllib.parse

class Parser(BaseParser):
    def __init__(self):
        super().__init__()

    def __call__(self, start_date, end_date, username: str, password: str):
        """
        Main method to parse Nightscout data.
        In the nightscout parser, the username is the nightscout URL, and the password is the API key.
        """
        try:
            api = nightscout.Api(username, api_secret=password)

            api_start_date = start_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            api_end_date = end_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')

            query_params = {'count': 0, 'find[dateString][$gte]': api_start_date,
                            'find[dateString][$lte]': api_end_date}

            # Fetch and save Sensor Glucose Values (SGV) data
            entries = api.get_sgvs(query_params)
            self.save_json(entries, 'entries', api_start_date, api_end_date)

            # Create DataFrame for Glucose (SGV) [mg/dL]
            df_glucose = self.create_dataframe(entries, 'date', 'sgv', 'CGM')
            print("Glucose DataFrame:")
            print(df_glucose)

            # Fetch treatments data (insulin, carbs, etc.)
            treatment_query_params = {'count': 0, 'find[created_at][$gte]': api_start_date, 'find[created_at][$lte]': api_end_date}
            treatments = api.get_treatments(treatment_query_params)
            self.save_json(treatments, 'treatments', api_start_date, api_end_date)

            # Create DataFrame for Carbs - include both Carb Correction and Meal Bolus
            df_carbs = self.create_dataframe(treatments, 'created_at', 'carbs', 'carbs', 
                                            event_type=['Carb Correction', 'Meal Bolus', 'Snack Bolus'])
            print("Carbs DataFrame:")
            print(df_carbs)

            # Create DataFrame for Bolus insulin - include both Bolus and Meal Bolus
            df_bolus = self.create_dataframe(treatments, 'created_at', 'insulin', 'bolus', 
                                            event_type=['Bolus', 'Meal Bolus', 'Snack Bolus', 'Correction Bolus', 'SMB'])
            print("Bolus DataFrame:")
            print(df_bolus)

            # Create DataFrame for temporary basal rates and durations
            df_temp_basal = self.create_dataframe(treatments, 'created_at', ['absolute', 'rate'], 'basal', 
                                                event_type='Temp Basal')
            df_temp_duration = self.create_dataframe(treatments, 'created_at', 'duration', 'duration', 
                                                   event_type='Temp Basal')
            print("Temporary Basal DataFrame:")
            print(df_temp_basal)

            # Fetch and save profiles data
            profile_query_params = {
                'count': 0,
                'find[created_at][$gte]': api_start_date,
                'find[created_at][$lte]': api_end_date
            }
            base_url = username.rstrip('/')  # Remove trailing slash if present
            profile_url = f"{base_url}/api/v1/profile?{urllib.parse.urlencode(profile_query_params)}"
            profiles_response = requests.get(profile_url, headers=api.request_headers())
            profiles = profiles_response.json()
            self.save_json_profiles(profiles, 'profiles', api_start_date, api_end_date)

            # Get profile-based basal rates and create basal DataFrame
            basal_rates = self.get_basal_rates_from_profile(profiles)
            df_basal_profile = self.create_basal_dataframe([start_date, end_date], basal_rates)
            print("Profile Basal DataFrame:")
            print(df_basal_profile)

            # Resampling all datatypes into the same time-grid
            df = df_glucose.resample('5min').mean().fillna(0)  # Fill NaN after resampling glucose
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
            df['id'] = 1  # You may need to adjust this if you have multiple patients

            # Add is_test column (it will be populated later in the processing pipeline)
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

    def merge_basal_rates(self, df, df_profile_basal, df_temp_basal, df_temp_duration):
        """
        Merge profile basal rates with temporary basal overrides, handling both absolute and percentage temp basals.
        """
        # Fill any NaN in profile basal rates with 0
        df_profile_basal['basal'] = df_profile_basal['basal'].fillna(0)
        
        # Start with profile basal rates
        df = df.merge(df_profile_basal, left_index=True, right_index=True, how='left')
        df['basal'] = df['basal'].fillna(0)  # Fill any NaN after merge
        
        if not df_temp_basal.empty:
            # For each temp basal entry
            temp_basals = pd.concat([df_temp_basal, df_temp_duration], axis=1)
            # Rename duplicate columns to avoid confusion
            temp_basals.columns = ['basal', 'percent_x', 'duration', 'percent_y']
            
            # Fill NaN in temp basals
            temp_basals['basal'] = temp_basals['basal'].fillna(0)
            temp_basals['percent_x'] = temp_basals['percent_x'].fillna(100)  # Default to 100% if not specified
            
            print("\nTemp basals after concat:")
            print(temp_basals)
            
            for time, row in temp_basals.iterrows():
                print(f"\nProcessing temp basal at {time}:")
                print(f"Row data: {row.to_dict()}")
                
                # Calculate end time of temp basal
                end_time = time + pd.Timedelta(minutes=int(row['duration']))
                print(f"End time: {end_time}")
                
                # Get the mask for this time period
                mask = (df.index >= time) & (df.index < end_time)
                affected_rows = df[mask]
                print(f"Number of affected rows: {len(affected_rows)}")
                
                # Check if we have an absolute value or percent
                basal_value = float(row['basal']) if pd.notnull(row['basal']) else 0
                percent_value = float(row['percent_x']) if pd.notnull(row['percent_x']) else 100
                
                print(f"Basal value: {basal_value}")
                print(f"Percent value: {percent_value}")
                
                if basal_value > 0:  # Absolute temp basal
                    print(f"Applying absolute basal: {basal_value}")
                    df.loc[mask, 'basal'] = basal_value
                elif percent_value != 100:  # Percentage temp basal
                    profile_rates = df.loc[mask, 'basal']
                    print(f"Profile rates before adjustment: {profile_rates.head()}")
                    new_rates = profile_rates * (percent_value / 100)
                    print(f"New rates after percentage: {new_rates.head()}")
                    df.loc[mask, 'basal'] = new_rates
                
                print("Sample of affected rows after update:")
                print(df[mask].head())
        
        # Fill any remaining NaN values
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
            if event_type:
                if isinstance(event_type, list):
                    if any(et in entry.eventType for et in event_type):
                        dates.append(getattr(entry, date_column))
                        if isinstance(value_column, list):
                            value = getattr(entry, value_column[0], None)
                            if value is None or pd.isna(value):
                                value = getattr(entry, value_column[1], 0)
                        else:
                            value = getattr(entry, value_column, 0)
                        values.append(value if not pd.isna(value) else 0)
                        percents.append(getattr(entry, 'percent', 0))  # Use 0 instead of None
                elif event_type in entry.eventType:
                    dates.append(getattr(entry, date_column))
                    if isinstance(value_column, list):
                        value = getattr(entry, value_column[0], None)
                        if value is None or pd.isna(value):
                            value = getattr(entry, value_column[1], 0)
                    else:
                        value = getattr(entry, value_column, 0)
                    values.append(value if not pd.isna(value) else 0)
                    percents.append(getattr(entry, 'percent', 0))  # Use 0 instead of None
            else:
                dates.append(getattr(entry, date_column))
                value = getattr(entry, value_column, 0)
                values.append(value if not pd.isna(value) else 0)
                percents.append(0)  # Use 0 instead of None
        
        df = pd.DataFrame({
            'date': dates,
            new_column_name: values,
            'percent': percents
        })
        
        # Fill any remaining NaN values with 0
        df[new_column_name] = df[new_column_name].fillna(0)
        df['percent'] = df['percent'].fillna(0)
        
        df['date'] = pd.to_datetime(df['date'], utc=True)
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
        Convert entry object to dictionary.
        """
        if hasattr(entry, '__dict__'):
            return entry.__dict__
        return dict(entry)