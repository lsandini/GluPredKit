"""
Loop v2 model adapted to work on Linux using pyloopkit instead of loop_to_python_api.
This provides the same functionality as loop_v2.py but works cross-platform.
Fixed version with correct pyloopkit input/output format.
"""
from .base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data
from glupredkit.metrics.rmse import Metric
from pyloopkit.loop_data_manager import update
from pyloopkit.dose import DoseType

import datetime
import numpy as np
import pandas as pd


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)
        
        self.subject_ids = None
        self.basal_rates = []
        self.insulin_sensitivities = []
        self.carb_ratios = []
        self.DIA = 360  # Duration of insulin action in minutes

    def _fit_model(self, x_train, y_train, n_cross_val_samples=200, *args):
        required_columns = ['CGM', 'carbs', 'basal', 'bolus']
        missing_columns = [col for col in required_columns if col not in x_train.columns]
        if missing_columns:
            raise ValueError(
                f"The Loop model requires the following features from the data input: {', '.join(missing_columns)}. "
                f"Please ensure that your dataset and configurations include these features. ")

        self.subject_ids = x_train['id'].unique()
        x_train['insulin'] = x_train['bolus'] + (x_train['basal'] / 12)

        rmse = Metric()

        for subject_id in self.subject_ids:
            
            x_train_filtered = x_train[x_train['id'] == subject_id]
            y_train_filtered = y_train[x_train['id'] == subject_id]

            subset_df_x = x_train_filtered.sample(n=n_cross_val_samples, random_state=42)
            subset_df_y = y_train_filtered.sample(n=n_cross_val_samples, random_state=42)

            # Flattened list of measured values across trajectory
            y_true = subset_df_y.to_numpy().ravel().tolist()

            # Calculate total daily insulin
            daily_insulin_series = x_train_filtered.groupby(pd.Grouper(freq='D')).agg({'insulin': 'sum'})['insulin']
            daily_avg_insulin = np.mean(daily_insulin_series) if len(daily_insulin_series) > 0 else 50
            print(f"Daily average insulin for subject {subject_id}: ", daily_avg_insulin)

            daily_basal_series = subset_df_x.groupby(pd.Grouper(freq='D')).agg({'basal': 'mean'})['basal']
            daily_avg_basal = np.mean(daily_basal_series) if len(daily_basal_series) > 0 else 1.0
            computed_basal = daily_avg_insulin * 0.45 / 24  # Basal 45% of TDI
            print(f"daily average basal is {daily_avg_basal}, while 45% of TDD is {computed_basal}")

            basal = (daily_avg_basal + computed_basal) / 2  # Average between 45% and their original setting
            print(f"Basal for subject {subject_id}: ", basal)

            isf = 1800 / daily_avg_insulin  # ISF 1800 rule
            cr = 500 / daily_avg_insulin  # CR 500 rule

            mult_factors = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]

            best_rmse = np.inf
            best_basal = basal
            best_isf = isf
            best_cr = cr
            
            for i in mult_factors:
                for j in mult_factors:
                    for basal_rate_factor in [0.3, 0.4, 0.5, 0.6]:
                        current_basal = daily_avg_insulin * basal_rate_factor / 24
                        y_pred = self._predict_model(subset_df_x, basal=current_basal, isf=isf*i, cr=cr*j)

                        # Flatten y_pred
                        if isinstance(y_pred, list) and any(isinstance(i, list) for i in y_pred):
                            y_pred = [item for sublist in y_pred for item in sublist]  # Flattening y_pred
                        else:
                            y_pred = y_pred  # Use as is if it's already flat

                        print(f'Factors {i} and {j}, basal {basal_rate_factor}')

                        iteration_result = rmse(y_true, y_pred)
                        print("RMSE: ", iteration_result)

                        if iteration_result < best_rmse and iteration_result != 0:
                            best_rmse = iteration_result
                            best_basal = current_basal
                            best_isf = isf * i
                            best_cr = cr * j
                            print(f"Best results so far with RMSE: {best_rmse}, basal: {best_basal}, "
                                  f"isf: {best_isf}, cr: {best_cr}")

            self.basal_rates += [best_basal]
            self.insulin_sensitivities += [best_isf]
            self.carb_ratios += [best_cr]

        return self

    def _predict_model(self, x_test, basal=None, isf=None, cr=None):
        n_predictions = self.prediction_horizon // 5
        y_pred = []

        for subject_idx, subject_id in enumerate(self.subject_ids):
            df_subset = x_test[x_test['id'] == subject_id]

            # Use provided parameters or defaults
            if basal is None:
                basal = self.basal_rates[subject_idx] if self.basal_rates else 1.0
            if isf is None:
                isf = self.insulin_sensitivities[subject_idx] if self.insulin_sensitivities else 40
            if cr is None:
                cr = self.carb_ratios[subject_idx] if self.carb_ratios else 10

            # Create base input dict
            input_dict = self.get_input_dict(isf, cr, basal)

            for _, row in df_subset.iterrows():
                # Get predictions using pyloopkit with proper format
                output_dict = self.get_prediction_output(row, input_dict)
                
                if output_dict and "predicted_glucose_values" in output_dict:
                    predictions = output_dict["predicted_glucose_values"]
                    # Clip to physiological range
                    predictions = [max(1, min(600, val)) for val in predictions]
                    # Skip first value (reference) and take only what we need
                    if len(predictions) > n_predictions:
                        y_pred += [predictions[1:n_predictions + 1]]
                    else:
                        # Pad with last value if needed
                        result = predictions[1:] if len(predictions) > 1 else [row['CGM']]
                        while len(result) < n_predictions:
                            result.append(result[-1] if result else row['CGM'])
                        y_pred += [result[:n_predictions]]
                else:
                    # Fallback to constant prediction
                    y_pred += [[row['CGM']] * n_predictions]

        return y_pred

    def get_input_dict(self, insulin_sensitivity, carb_ratio, basal):
        """Create the base input dictionary for pyloopkit, matching loop.py format."""
        return {
            'carb_value_units': 'g',
            'settings_dictionary': {
                'model': [self.DIA, 75],  # DIA and peak time
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
                'max_basal_rate': basal * 4,  # Allow up to 4x basal
                'max_bolus': 12.0,
                'retrospective_correction_enabled': True
            },
            'sensitivity_ratio_start_times': [datetime.time(0, 0)],
            'sensitivity_ratio_end_times': [datetime.time(0, 0)],
            'sensitivity_ratio_values': [insulin_sensitivity],
            'sensitivity_ratio_value_units': 'mg/dL/U',
            'carb_ratio_start_times': [datetime.time(0, 0)],
            'carb_ratio_values': [carb_ratio],
            'carb_ratio_value_units': 'g/U',
            'basal_rate_start_times': [datetime.time(0, 0)],
            'basal_rate_minutes': [1440],
            'basal_rate_values': [basal],
            'basal_rate_value_units': 'U/hour',
            'target_range_start_times': [datetime.time(0, 0)],
            'target_range_end_times': [datetime.time(0, 0)],
            'target_range_minimum_values': [100],
            'target_range_maximum_values': [110],
            'target_range_value_units': 'mg/dL'
        }

    def get_prediction_output(self, df_row, input_dict):
        """Get prediction output from pyloopkit, matching loop.py implementation."""
        time_to_calculate = df_row.name
        if isinstance(time_to_calculate, np.int64):
            time_to_calculate = datetime.datetime.now()

        input_dict["time_to_calculate_at"] = time_to_calculate

        # Helper function to extract dates and values from dataframe row
        def get_dates_and_values(column, data):
            relevant_columns = [val for val in data.index if val.startswith(column)]
            dates = []
            values = []

            date = data.name
            if isinstance(date, np.int64):
                date = datetime.datetime.now()

            for col in relevant_columns:
                if col == column:
                    values.append(data[col])
                    dates.append(date)
                elif "what_if" in col:
                    values.append(data[col])
                    new_date = date + datetime.timedelta(minutes=int(col.split("_")[-1]))
                    dates.append(new_date)
                else:
                    values.append(data[col])
                    new_date = date - datetime.timedelta(minutes=int(col.split("_")[-1]))
                    dates.append(new_date)

            if dates and values:
                # Sort by dates
                combined = list(zip(dates, values))
                combined.sort(key=lambda x: x[0])
                dates, values = zip(*combined)

            return dates, values

        # Get glucose data
        glucose_dates, glucose_values = get_dates_and_values("CGM", df_row)
        input_dict["glucose_dates"] = glucose_dates
        input_dict["glucose_values"] = glucose_values

        # Get insulin data
        bolus_dates, bolus_values = get_dates_and_values("bolus", df_row)
        basal_dates, basal_values = get_dates_and_values("basal", df_row)
        
        dose_types = []
        dose_start_times = []
        dose_end_times = []
        dose_values = []
        dose_delivered_units = []

        # Add bolus doses
        for date, value in zip(bolus_dates, bolus_values):
            if value > 0:
                dose_types.append(DoseType.bolus)
                dose_start_times.append(date)
                dose_end_times.append(date)
                dose_values.append(value)
                dose_delivered_units.append(None)

        # Add basal doses
        for date, value in zip(basal_dates, basal_values):
            if value > 0:
                dose_types.append(DoseType.tempbasal)
                dose_start_times.append(date)
                dose_end_times.append(date + pd.Timedelta(minutes=5))
                dose_values.append(value)  # Already in U/hr
                dose_delivered_units.append(None)

        input_dict["dose_types"] = dose_types
        input_dict["dose_start_times"] = dose_start_times
        input_dict["dose_end_times"] = dose_end_times
        input_dict["dose_values"] = dose_values
        input_dict["dose_delivered_units"] = dose_delivered_units

        # Get carb data
        carb_data = df_row[df_row.index.str.startswith('carbs') & (df_row != 0)]
        carb_dates, carb_values = get_dates_and_values("carbs", carb_data)
        input_dict["carb_dates"] = carb_dates
        input_dict["carb_values"] = carb_values
        input_dict["carb_absorption_times"] = [180 for _ in carb_values]

        try:
            return update(input_dict)
        except Exception as e:
            print(f"Error in pyloopkit update: {e}")
            return None

    def best_params(self):
        best_params = [{
            "basal rates": self.basal_rates,
            "insulin sensitivities": self.insulin_sensitivities,
            "carbohydrate ratios": self.carb_ratios,
        }]
        
        return best_params

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)