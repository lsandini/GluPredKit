"""
The Apple Health parser is processing the raw .xml data from Apple Health export and returning the data merged into
the same time grid in a dataframe.
"""
from .base_parser import BaseParser
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


def getDataframeForType(data, type, name):
    df = data[data.type == type]
    df = df.copy()
    df.rename(columns={'value': name}, inplace=True)
    df.rename(columns={'startDate': 'date'}, inplace=True)
    df.drop(['type', 'sourceName', 'sourceVersion', 'unit', 'creationDate', 'device', 'endDate'], axis=1, inplace=True)
    df.set_index('date', inplace=True)
    return df


# TODO: Refactor base_parser and CLI to be more flexible for parsers
class Parser(BaseParser):
    def __init__(self):
        super().__init__()

    def __call__(self, start_date, end_date, file_path: str, file_name: str):
        tree = ET.parse(file_path + file_name)
        root = tree.getroot()
        record_list = [x.attrib for x in root.iter('Record')]
        data = pd.DataFrame(record_list)

        # Dates from string to datetime
        for col in ['creationDate', 'startDate', 'endDate']:
            data[col] = pd.to_datetime(data[col], errors='coerce')

        # value is numeric, NaN if fails
        data['value'] = pd.to_numeric(data['value'], errors='coerce')
        
        # Filter the data on the input start- and enddate
        data['startDate'] = data['startDate'].dt.tz_localize(None)
        data = data[(data['startDate'] >= start_date) & (data['startDate'] <= end_date)]

        # Creating separate dataframes for each data type
        df_glucose = getDataframeForType(data, 'HKQuantityTypeIdentifierBloodGlucose', 'CGM')
        df_insulin = getDataframeForType(data, 'HKQuantityTypeIdentifierInsulinDelivery', 'insulin')
        df_carbs = getDataframeForType(data, 'HKQuantityTypeIdentifierDietaryCarbohydrates', 'carbs')
        df_heartrate = getDataframeForType(data, 'HKQuantityTypeIdentifierHeartRate', 'heartrate')
        df_heartratevariability = getDataframeForType(data, 'HKQuantityTypeIdentifierHeartRateVariabilitySDNN', 'heartratevariability')
        df_caloriesburned = getDataframeForType(data, 'HKQuantityTypeIdentifierActiveEnergyBurned', 'caloriesburned')
        df_respiratoryrate = getDataframeForType(data, 'HKQuantityTypeIdentifierRespiratoryRate', 'respiratoryrate')
        df_steps = getDataframeForType(data, 'HKQuantityTypeIdentifierStepCount', 'steps')
        df_restingheartrate = getDataframeForType(data, 'HKQuantityTypeIdentifierRestingHeartRate', 'restingheartrate')

		# TODO: Add workouts and sleep data
        
        # Resampling all datatypes into the same time-grid
        df = df_glucose.copy()
        df = df.resample('5T', label='right').mean()

        df_carbs = df_carbs.resample('5T', label='right').sum().fillna(value=0)
        df = pd.merge(df, df_carbs, on="date", how='outer')
        df['carbs'] = df['carbs'].fillna(value=0.0)

        df_insulin = df_insulin.resample('5T', label='right').sum()
        df = pd.merge(df, df_insulin, on="date", how='outer')
        
        df_heartrate = df_heartrate.resample('5T', label='right').mean()
        df = pd.merge(df, df_heartrate, on="date", how='outer')
        
        df_heartratevariability = df_heartratevariability.resample('5T', label='right').mean()
        df = pd.merge(df, df_heartratevariability, on="date", how='outer')
        
        df_caloriesburned = df_caloriesburned.resample('5T', label='right').sum()
        df = pd.merge(df, df_caloriesburned, on="date", how='outer')
        
        df_respiratoryrate = df_respiratoryrate.resample('5T', label='right').mean()
        df = pd.merge(df, df_respiratoryrate, on="date", how='outer')
        
        df_steps = df_steps.resample('5T', label='right').sum()
        df = pd.merge(df, df_steps, on="date", how='outer')
        
        df_restingheartrate = df_restingheartrate.resample('5T', label='right').mean()
        df = pd.merge(df, df_restingheartrate, on="date", how='outer')

        return df
