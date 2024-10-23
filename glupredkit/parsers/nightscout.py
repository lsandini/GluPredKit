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
                                            event_type=['Bolus', 'Meal Bolus', 'Snack Bolus', 'Correction Bolus'])
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
            # profile_url = f"{username}api/v1/profile?{urllib.parse.urlencode(profile_query_params)}"
            profiles_response = requests.get(profile_url, headers=api.request_headers())
            profiles = profiles_response.json()
            self.save_json_profiles(profiles, 'profiles', api_start_date, api_end_date)

            # Get profile-based basal rates and create basal DataFrame
            basal_rates = self.get_basal_rates_from_profile(profiles)
            df_basal_profile = self.create_basal_dataframe([start_date, end_date], basal_rates)
            print("Profile Basal DataFrame:")
            print(df_basal_profile)

            # Resampling all datatypes into the same time-grid
            df = df_glucose.resample('5min').mean()
            df = self.merge_and_process(df, df_carbs, 'carbs')
            df = self.merge_and_process(df, df_bolus, 'bolus')

            # Merge profile basals and temporary basals
            df = self.merge_basal_rates(df, df_basal_profile, df_temp_basal, df_temp_duration)

            # Process basal rate
            df['basal'] = round(df['basal'] / 60 * 5, 5)  # From U/hr to U (5-minutes)
            df['insulin'] = df['bolus'].fillna(0) + df['basal'].fillna(0)

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
            df.set_index('date', inplace=True)

            print("Final DataFrame:")
            print(df)

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
        # Start with profile basal rates
        df = df.merge(df_profile_basal, left_index=True, right_index=True, how='left')
        
        if not df_temp_basal.empty:
            # For each temp basal entry
            temp_basals = pd.concat([df_temp_basal, df_temp_duration], axis=1)
            # Rename duplicate columns to avoid confusion
            temp_basals.columns = ['basal', 'percent_x', 'duration', 'percent_y']
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
                
                # Check if we have an absolute value or percent (use percent_x which comes from temp_basal)
                basal_value = float(row['basal']) if pd.notnull(row['basal']) else None
                percent_value = float(row['percent_x']) if pd.notnull(row['percent_x']) else None
                
                print(f"Basal value: {basal_value}")
                print(f"Percent value: {percent_value}")
                
                if pd.notnull(basal_value):  # Absolute temp basal
                    print(f"Applying absolute basal: {basal_value}")
                    df.loc[mask, 'basal'] = basal_value
                elif pd.notnull(percent_value):  # Percentage temp basal
                    # Get the profile basal rates for this period and multiply by percentage
                    profile_rates = df.loc[mask, 'basal']
                    print(f"Profile rates before adjustment: {profile_rates.head()}")
                    new_rates = profile_rates * (percent_value / 100)
                    print(f"New rates after percentage: {new_rates.head()}")
                    df.loc[mask, 'basal'] = new_rates
                
                print("Sample of affected rows after update:")
                print(df[mask].head())
        
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
        
        # Convert to list of (seconds, rate) tuples
        basal_rates = [(entry['timeAsSeconds'], entry['value']) for entry in basal_schedule]
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
            # Calculate seconds since midnight
            seconds = (current_date.hour * 3600 + 
                      current_date.minute * 60 + 
                      current_date.second)
            
            # Get basal rate for this time and round to 5 decimal places
            rate = round(self.get_basal_rate_for_time(basal_rates, seconds), 5)
            
            dates.append(current_date)
            rates.append(rate)
            
            current_date += datetime.timedelta(minutes=5)
        
        df = pd.DataFrame({'date': dates, 'basal': rates})
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
        percents = []  # New list for percentage values
        
        for entry in data:
            if event_type:
                if isinstance(event_type, list):
                    if any(et in entry.eventType for et in event_type):
                        dates.append(getattr(entry, date_column))
                        if isinstance(value_column, list):
                            # For temp basal, try to get absolute first
                            value = getattr(entry, value_column[0], None)
                            if value is None:
                                value = getattr(entry, value_column[1], 0)
                        else:
                            value = getattr(entry, value_column, 0)
                        values.append(value)
                        # Get percent if it exists
                        if hasattr(entry, '_json') and 'percent' in entry._json:
                            percents.append(entry._json['percent'])
                        else:
                            percents.append(None)
                elif event_type in entry.eventType:
                    dates.append(getattr(entry, date_column))
                    if isinstance(value_column, list):
                        value = getattr(entry, value_column[0], None)
                        if value is None:
                            value = getattr(entry, value_column[1], 0)
                    else:
                        value = getattr(entry, value_column, 0)
                    values.append(value)
                    # Get percent if it exists
                    if hasattr(entry, '_json') and 'percent' in entry._json:
                        percents.append(entry._json['percent'])
                    else:
                        percents.append(None)
            else:
                dates.append(getattr(entry, date_column))
                values.append(getattr(entry, value_column, 0))
                percents.append(None)
        
        df = pd.DataFrame({
            'date': dates,
            new_column_name: values,
            'percent': percents  # Add percent column
        })
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        return df

    def merge_and_process(self, df, df_to_merge, column_name):
        """
        Merge and process dataframes.
        """
        if not df_to_merge.empty:
            df_to_merge = df_to_merge.resample('5min').sum()
            df = pd.merge(df, df_to_merge, left_index=True, right_index=True, how='outer')
            df[column_name] = df[column_name].fillna(0)
        else:
            df[column_name] = 0
        return df

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