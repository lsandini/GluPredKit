#!/usr/bin/env python3
import click
import dill
import requests
import warnings
import pickle
import os
import ast
import importlib
import pandas as pd
from pathlib import Path
from datetime import timedelta, datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import numpy as np

# Modules from this repository
from .parsers.base_parser import BaseParser
from .plots.base_plot import BasePlot
from glupredkit.helpers.unit_config_manager import unit_config_manager
from glupredkit.helpers.model_config_manager import ModelConfigurationManager, generate_model_configuration
import glupredkit.helpers.cli as helpers
import glupredkit.helpers.generate_report as generate_report
from datetime import timedelta, datetime, timezone


# TODO: Fix so that all default values are defined upstream (=here in the CLI), and removed from downstream


@click.command()
def setup_directories():
    """Set up necessary directories for GluPredKit."""
    cwd = os.getcwd()
    print("Creating directories...")

    folder_path = 'data'
    folder_names = ['raw', 'configurations', 'trained_models', 'tested_models', 'figures', 'reports']

    for folder_name in folder_names:
        path = os.path.join(cwd, folder_path, folder_name)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory {path}.")

    print("Directories created for usage of GluPredKit.")


@click.command()
@click.option('--parser', type=click.Choice(['tidepool', 'nightscout', 'apple_health', 'ohio_t1dm']),
              help='Choose a parser', required=True)
@click.option('--username', type=str, required=False)
@click.option('--password', type=str, required=False)
@click.option('--start-date', type=str,
              help='Start date for data retrieval. Default is two weeks ago. Format "dd-mm-yyyy"')
@click.option('--file-path', type=str, required=False)
@click.option('--end-date', type=str, help='End date for data retrieval. Default is now. Format "dd-mm-yyyy"')
@click.option('--output-file-name', type=str, help='The file name for the output.')
@click.option('--test-size', type=float, default=0.25)
def parse(parser, username, password, start_date, file_path, end_date, output_file_name, test_size):
    """Parse data and store it as CSV in data/raw using a selected parser"""

    # Load the chosen parser dynamically based on user input
    parser_module = importlib.import_module(f'glupredkit.parsers.{parser}')

    # Ensure the chosen parser inherits from BaseParser
    if not issubclass(parser_module.Parser, BaseParser):
        raise click.ClickException(f"The selected parser '{parser}' must inherit from BaseParser.")

    # Create an instance of the chosen parser
    chosen_parser = parser_module.Parser()

    click.echo(f"Parsing data using {parser}...")

    date_format = "%d-%m-%Y"

    if not end_date:
        end_date = datetime.now()
    else:
        end_date = datetime.strptime(end_date, date_format)

    if not start_date:
        start_date = end_date - timedelta(days=14)
    else:
        start_date = datetime.strptime(start_date, date_format)

    def save_data(output_file_name, data):
        output_path = 'data/raw/'
        date_format = "%d-%m-%Y"

        if output_file_name:
            file_name = output_file_name + '.csv'
        else:
            file_name = (parser + '_' + start_date.strftime(date_format) + '_to_' + end_date.strftime(
                date_format) + '.csv')

        click.echo("Storing data as CSV...")
        helpers.store_data_as_csv(data, output_path, file_name)
        click.echo(f"Data stored as CSV at '{output_path}' as '{file_name}'")
        click.echo(f"Data has the shape: {data.shape}")

    # Ensure that the optional params match the parser
    if parser in ['tidepool', 'nightscout']:
        if username is None or password is None:
            raise ValueError(f"{parser} parser requires that you provide --username and --password")
        else:
            parsed_data = chosen_parser(start_date=start_date, end_date=end_date, username=username, password=password)
    elif parser in ['apple_health']:
        if file_path is None:
            raise ValueError(f"{parser} parser requires that you provide --file-path")
        else:
            parsed_data = chosen_parser(start_date=start_date, end_date=end_date,
                                        file_path=file_path)
    elif parser in ['ohio_t1dm']:
        if file_path is None:
            raise ValueError(f"{parser} parser requires that you provide --file-path")
        else:
            ids_2018 = ['559', '563', '570', '575', '588', '591']
            ids_2020 = ['540', '544', '552', '567', '584', '596']

            merged_df = pd.DataFrame()

            for subject_id in ids_2018:
                parsed_data = chosen_parser(file_path=file_path, subject_id=subject_id, year='2018')
                parsed_data['id'] = subject_id
                merged_df = pd.concat([parsed_data, merged_df], ignore_index=False)

            for subject_id in ids_2020:
                parsed_data = chosen_parser(file_path=file_path, subject_id=subject_id, year='2020')
                parsed_data['id'] = subject_id
                merged_df = pd.concat([parsed_data, merged_df], ignore_index=False)
            save_data(output_file_name="OhioT1DM", data=merged_df)

            return
    else:
        raise ValueError(f"unrecognized parser: '{parser}'")

    # Train and test split
    # Adding a margin of 24 hours to the train and the test data to avoid memory leak
    margin = int((12 * 24) / 2)
    split_index = int((len(parsed_data)) * (1 - test_size))

    parsed_data['is_test'] = False
    parsed_data['is_test'].iloc[split_index:] = True
    parsed_data = parsed_data.drop(parsed_data.index[split_index - margin:split_index + margin])

    save_data(output_file_name=output_file_name, data=parsed_data)


@click.command()
@click.option('--file-name', help='Give a file name to the configuration file (e.g., "my_config").',
              callback=helpers.validate_config_file_name, required=True)
@click.option('--data', help='Name of the input data file (e.g., "data.csv"). '
                             'This file should be located in data/raw/.',
              callback=helpers.check_if_data_file_exists, required=True)
@click.option('--subject-ids', help='The ids you want to include in the model training and testing. '
                                    ' will include all of the subjects.', callback=helpers.validate_subject_ids,
              required=False, default='')
@click.option('--preprocessor', required=False, default='basic', type=click.Choice([
    'basic',
    'standardscaler'
]))
@click.option('--prediction-horizon', help='Integer for prediction horizon in minutes.',
              callback=helpers.validate_prediction_horizon, required=True)
@click.option('--num-lagged-features', help='Number of lagged features.',
              callback=helpers.validate_num_lagged_features, required=True)
@click.option('--num-features', help='Comma-separated list of numerical features.',
              callback=helpers.validate_feature_list, required=True, default='CGM')
@click.option('--cat-features', help='Comma-separated list of categorical features.',
              callback=helpers.validate_feature_list, required=False, default='')
@click.option('--what-if-features', help='Comma-separated list of categorical features.',
              callback=helpers.validate_feature_list, required=False, default='')
def generate_config(file_name, data, subject_ids, preprocessor, prediction_horizon, num_lagged_features, num_features,
                    cat_features, what_if_features):

    if data == 'synthetic_data':
        cwd = os.getcwd()
        url = 'https://raw.githubusercontent.com/miriamkw/GluPredKit/main/example_data/synthetic_data.csv'
        save_folder = 'data/raw/'
        save_path = os.path.join(cwd, save_folder, 'synthetic_data.csv')

        if not os.path.exists(save_path):
            response = requests.get(url)
            with open(save_path, 'wb') as file:
                file.write(response.content)

            click.echo(f"Synthetic data saved to {save_path}")

    generate_model_configuration(file_name, data, subject_ids, preprocessor, int(prediction_horizon),
                                 int(num_lagged_features), num_features, cat_features, what_if_features)
    click.echo(f"Storing configuration file to data/configurations/{file_name}...")
    click.echo(f"Note that it might take a minute before the file appears in the folder.")


@click.command()
@click.argument('model', type=click.Choice([
    'double_lstm',
    'loop',
    'lstm',
    'mtl',
    'naive_linear_regressor',
    'random_forest',
    'ridge',
    'stacked_plsr',
    'stl',
    'svr',
    'tcn',
    'uva_padova',
    'zero_order'
]))
@click.argument('config-file-name', type=str)
@click.option('--epochs', type=int, required=False)
@click.option('--n-cross-val-samples', type=int, required=False)
@click.option('--n-steps', type=int, required=False)
@click.option('--training-samples-per-subject', type=int, required=False)
def train_model(model, config_file_name, epochs, n_cross_val_samples, n_steps, training_samples_per_subject):
    """
    This method does the following:
    1) Process data using the given configurations
    2) Train the given models for the given prediction horizons in the configuration
    """
    # Filtering out UserWarning because we are using an old Keras file format on purpose
    warnings.filterwarnings(
        "ignore",
        message=".*You are saving your model as an HDF5 file.*"
    )

    click.echo(f"Starting pipeline to train model {model} with configurations in {config_file_name}...")
    model_config_manager = ModelConfigurationManager(config_file_name)

    prediction_horizon = model_config_manager.get_prediction_horizon()
    model_module = helpers.get_model_module(model)

    # PREPROCESSING
    # Perform data preprocessing using your preprocessor
    train_data, _ = helpers.get_preprocessed_data(prediction_horizon, model_config_manager)
    click.echo(f"Training data finished preprocessing...")

    # MODEL TRAINING
    # Create an instance of the chosen model
    chosen_model = model_module.Model(prediction_horizon)

    processed_data = chosen_model.process_data(train_data, model_config_manager, real_time=False)
    target_columns = [column for column in processed_data.columns if column.startswith('target')]
    x_train = processed_data.drop(target_columns, axis=1)
    y_train = processed_data[target_columns]

    click.echo(f"Training model...")

    # Initialize and train the model
    # Ensure that the optional params match the parser
    if model in ['double_lstm', 'lstm', 'mtl', 'stl', 'tcn'] and epochs:
        model_instance = chosen_model.fit(x_train, y_train, epochs)
    elif model in ['loop'] and n_cross_val_samples:
        model_instance = chosen_model.fit(x_train, y_train, n_cross_val_samples)
    elif model in ['uva_padova'] and n_steps or training_samples_per_subject:
        model_instance = chosen_model.fit(x_train, y_train, n_steps, training_samples_per_subject)
    else:
        model_instance = chosen_model.fit(x_train, y_train)

    click.echo(f"Model {model} with prediction horizon {prediction_horizon} minutes trained successfully!")

    # Assuming model_instance is your class instance
    output_dir = Path("data") / "trained_models"
    output_file_name = f'{model}__{config_file_name}__{prediction_horizon}.pkl'
    output_path = output_dir / output_file_name

    try:
        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the model to a file
        with open(output_path, 'wb') as f:
            click.echo(f"Saving model {model} to {output_path}...")
            dill.dump(model_instance, f)
    except Exception as e:
        click.echo(f"Error saving model {model}: {e}")

    # Optionally print model hyperparameters if they exist
    if hasattr(model_instance, 'best_params'):
        click.echo(f"Model hyperparameters: {model_instance.best_params()}")


@click.command()
@click.argument('model_file', type=str)
@click.option('--max-samples', type=int, required=False)
def evaluate_model(model_file, max_samples):
    tested_models_path = "data/tested_models"

    model_name, config_file_name, prediction_horizon = (model_file.split('__')[0], model_file.split('__')[1],
                                                        int(model_file.split('__')[2].split('.')[0]))

    model_config_manager = ModelConfigurationManager(config_file_name)
    model_instance = helpers.get_trained_model(model_file)
    training_data, test_data = helpers.get_preprocessed_data(prediction_horizon, model_config_manager)

    processed_data = model_instance.process_data(test_data, model_config_manager, real_time=False)
    target_cols = [col for col in processed_data if col.startswith('target')]

    if max_samples:
        x_test = processed_data.drop(target_cols, axis=1)[-max_samples:]
        y_test = processed_data[target_cols][-max_samples:]
    else:
        x_test = processed_data.drop(target_cols, axis=1)
        y_test = processed_data[target_cols]
    y_pred = model_instance.predict(x_test)

    hypo_threshold = 70
    hyper_threshold = 180

    # Create a dataframe to store the model name, configuration, predictions, and other results
    results_df = pd.DataFrame({
        'Model Name': [model_name],
        'training_samples': [training_data.shape[0]],
        'test_samples': [test_data.shape[0]],
        'hypo_training_samples': [training_data[training_data['CGM'] < hypo_threshold].shape[0]],
        'hypo_test_samples': [test_data[test_data['CGM'] < hypo_threshold].shape[0]],
        'hyper_training_samples': [training_data[training_data['CGM'] > hyper_threshold].shape[0]],
        'hyper_test_samples': [test_data[test_data['CGM'] > hyper_threshold].shape[0]],
        'unit': [unit_config_manager.get_unit()]
    })

    configs = model_config_manager.load_config()
    for config in configs:
        results_df[config] = [configs[config]]

    # Add daily average insulin if relevant
    num_features = model_config_manager.get_num_features()
    if 'bolus' in num_features and 'basal' in num_features:
        test_data['insulin'] = test_data['bolus'] + (test_data['basal'] / 12)
        results_df['daily_avg_insulin'] = np.mean(test_data.groupby(pd.Grouper(freq='D')).agg({'insulin': 'sum'}))
    elif ['insulin'] in num_features:
        results_df['daily_avg_insulin'] = np.mean(test_data.groupby(pd.Grouper(freq='D')).agg({'insulin': 'sum'}))

    metrics = helpers.list_files_in_package('metrics')
    metrics = [os.path.splitext(file)[0] for file in metrics if file not in ('__init__.py', 'base_metric.py')]

    if target_cols == ['target']:
        # In this case, targets are stored into sequences for Neural Networks
        targets = [np.array(ast.literal_eval(target_str)) for target_str in y_test['target']]
        targets = np.array(targets)
        _, n_predictions = targets.shape

        for i, minutes in enumerate(range(5, n_predictions * 5 + 1, 5)):
            curr_y_test = targets[:, i].tolist()
            curr_y_pred = [float(val[i]) for val in y_pred]
            results_df = results_df.copy()  # To silent PerformanceWarning
            results_df[f'target_{minutes}'] = [curr_y_test]
            results_df[f'y_pred_{minutes}'] = [curr_y_pred]

            for metric in metrics:
                metric_module = helpers.get_metric_module(metric)
                chosen_metric = metric_module.Metric()
                score = chosen_metric(curr_y_test, curr_y_pred)
                results_df[f'{metric}_{minutes}'] = [score]

    else:
        for i, minutes in enumerate(range(5, len(target_cols) * 5 + 1, 5)):
            curr_y_test = y_test[target_cols[i]].tolist()
            curr_y_pred = [float(val[i]) for val in y_pred]
            results_df = results_df.copy()  # To silent PerformanceWarning
            results_df[target_cols[i]] = [curr_y_test]
            results_df[f'y_pred_{minutes}'] = [curr_y_pred]

            for metric in metrics:
                metric_module = helpers.get_metric_module(metric)
                chosen_metric = metric_module.Metric()
                score = chosen_metric(y_test[target_cols[i]], curr_y_pred)
                results_df[f'{metric}_{minutes}'] = [score]

    subset_size = x_test.shape[0]

    # For physiological models the insulin and meal curves are deterministic, and we can reduce the samples
    if (model_name == 'loop') | (model_name == 'uva_padova'):
        subset_size = 1000
    subset_df_x = x_test[-subset_size:]

    insulin_doses = [1, 5, 10]
    carb_intakes = [10, 50, 100]

    insulin_col = None
    if 'insulin' in num_features:
        insulin_col = 'insulin'
    elif 'bolus' in num_features:
        insulin_col = 'bolus'

    if insulin_col:
        if 'sequence' in x_test.columns:
            _, test_data = helpers.get_preprocessed_data(prediction_horizon, model_config_manager)
            x_test_copy = test_data.copy()
        else:
            x_test_copy = subset_df_x.copy()

        x_test_copy[insulin_col] = 0

        if 'sequence' in x_test.columns:
            processed_data = model_instance.process_data(x_test_copy, model_config_manager, real_time=False)
            x_test_copy = processed_data.drop(target_cols, axis=1)[-subset_size:]

        y_pred_bolus_0 = model_instance.predict(x_test_copy)
        y_pred_bolus_0 = [np.nanmean(x) for x in zip(*y_pred_bolus_0)]

        for insulin_dose in insulin_doses:
            if 'sequence' in x_test.columns:
                x_test_copy = test_data.copy()
            else:
                x_test_copy = subset_df_x.copy()

            x_test_copy[insulin_col] = insulin_dose

            if 'sequence' in x_test.columns:
                processed_data = model_instance.process_data(x_test_copy, model_config_manager, real_time=False)
                x_test_copy = processed_data.drop(target_cols, axis=1)[-subset_size:]

            y_pred = model_instance.predict(x_test_copy)

            # Calculate the average of elements at each index position
            averages = [np.nanmean(x) for x in zip(*y_pred)]
            averages = [float(x - y) for x, y in zip(averages, y_pred_bolus_0)]

            results_df[f'partial_dependency_bolus_{insulin_dose}'] = [averages]

    if 'carbs' in num_features:
        if 'sequence' in x_test.columns:
            x_test_copy = test_data.copy()
        else:
            x_test_copy = subset_df_x.copy()

        x_test_copy['carbs'] = 0

        if 'sequence' in x_test.columns:
            processed_data = model_instance.process_data(x_test_copy, model_config_manager, real_time=False)
            x_test_copy = processed_data.drop(target_cols, axis=1)[-subset_size:]

        y_pred_carbs_0 = model_instance.predict(x_test_copy)
        y_pred_carbs_0 = [np.nanmean(x) for x in zip(*y_pred_carbs_0)]

        for carb_intake in carb_intakes:
            if 'sequence' in x_test.columns:
                x_test_copy = test_data.copy()
            else:
                x_test_copy = subset_df_x.copy()

            x_test_copy['carbs'] = carb_intake

            if 'sequence' in x_test.columns:
                processed_data = model_instance.process_data(x_test_copy, model_config_manager, real_time=False)
                x_test_copy = processed_data.drop(target_cols, axis=1)[-subset_size:]

            y_pred = model_instance.predict(x_test_copy)
            # Calculate the average of elements at each index position
            averages = [np.nanmean(x) for x in zip(*y_pred)]
            averages = [float(x - y) for x, y in zip(averages, y_pred_carbs_0)]
            results_df[f'partial_dependency_carbs_{carb_intake}'] = [averages]

    # Define the path to store the dataframe
    output_file = f"{tested_models_path}/{model_name}__{config_file_name}__{prediction_horizon}.csv"

    # Store the dataframe in a file
    results_df.to_csv(output_file, index=False)
    click.echo(f"Model {model_name} is finished testing. Results are stored in {tested_models_path}")


@click.command()
@click.option('--results-files', help='The name of the tested model results to evaluate, with ".csv". If '
                                      'None, all models will be tested.')
@click.option('--plots', help='List of plots to be computed, separated by comma. '
                              'By default a scatter plot will be drawn. ', default='scatter_plot')
@click.option('--start-date', type=str,
              help='Start date for the predictions. Default is the first sample in the test data. '
                   'Format "dd-mm-yyyy/hh:mm"', default=None)
@click.option('--end-date', type=str,
              help='End date, or prediction date for one prediction plots. Default is the last sample in the test data.'
                   'Format "dd-mm-yyyy/hh:mm"', default=None)
@click.option('--prediction-horizons', help='Integer for prediction horizons in minutes. Comma-separated'
                                            'without space. Required for scatter plot. ', default=None)
def draw_plots(results_files, plots, start_date, end_date, prediction_horizons):
    """
    This command draws the given plots and store them in data/figures/.
    """
    # Prepare a list of plots
    plots = helpers.split_string(plots)

    if results_files is None:
        results_files = helpers.list_files_in_directory('data/tested_models/')
    else:
        results_files = helpers.split_string(results_files)

    dfs = []
    for results_file in results_files:
        df = generate_report.get_df_from_results_file(results_file)
        dfs += [df]

    plot_results_path = 'data/figures/'

    # Draw plots
    click.echo(f"Drawing plots...")
    os.makedirs(plot_results_path, exist_ok=True)
    for plot in plots:
        plot_module = importlib.import_module(f'glupredkit.plots.{plot}')
        if not issubclass(plot_module.Plot, BasePlot):
            raise click.ClickException(f"The selected plot '{plot}' must inherit from BasePlot.")
        chosen_plot = plot_module.Plot()

        if plot == 'scatter_plot':
            if prediction_horizons is None:
                raise ValueError(f"{plot} requires that you provide --prediction-horizons")
            prediction_horizons = helpers.split_string(prediction_horizons)
            for prediction_horizon in prediction_horizons:
                chosen_plot(dfs, prediction_horizon)

        elif plot == 'trajectories':
            chosen_plot(dfs)

        else:
            click.echo(f"Plot {plot} does not exist. Please look in the documentation for the existing plots.")

    click.echo(f"{plots} for trained_models {results_files} are stored in '{plot_results_path}'")


@click.command()
@click.option('--results-file', help='The name of the tested model results to evaluate, with ".csv".',
              required=True)
def generate_evaluation_pdf(results_file):
    """
    This command stores a standardized pdf report of the given model in data/reports/.
    """
    click.echo(f"Generating evaluation report...")
    df = generate_report.get_df_from_results_file(results_file)

    # Convert results to DataFrame and save as CSV
    results_file_path = 'data/reports/'
    os.makedirs(results_file_path, exist_ok=True)
    results_file_name = f'{results_file.split(".")[0]}.pdf'

    # Create a PDF canvas
    c = canvas.Canvas(f"data/reports/{results_file_name}", pagesize=letter)

    # FRONT PAGE
    c = generate_report.set_title(c, f'Model Evaluation for {df["Model Name"][0]}')
    c = generate_report.generate_single_model_front_page(c, df)
    c = generate_report.set_bottom_text(c)
    c.showPage()

    # MODEL ACCURACY
    c = generate_report.set_title(c, f'Model Accuracy')
    c = generate_report.draw_model_accuracy_table(c, df)
    plot_height = 1.3  # TODO: Dynamically scale when table is larger
    c = generate_report.plot_across_prediction_horizons(c, df, f'RMSE [{unit_config_manager.get_unit()}]',
                                                        ['rmse'], y_placement=420, height=plot_height)
    c = generate_report.plot_across_prediction_horizons(c, df, f'ME [{unit_config_manager.get_unit()}]',
                                                        ['me'], y_placement=260, height=plot_height)
    c = generate_report.plot_across_prediction_horizons(c, df, f'MARE [%]',
                                                        ['mare'], y_placement=100, height=plot_height)
    c = generate_report.set_bottom_text(c)
    c.showPage()

    def round_to_nearest_5(number, divisor):
        if number <= 5:
            return 5
        result = number / divisor
        rounded_result = 5 * round(result / 5)
        return rounded_result

    prediction_horizon = generate_report.get_ph(df)
    if prediction_horizon < 20:
        ph_subparts_list = list(range(5, prediction_horizon + 5, 5))
    else:
        ph_quarter = round_to_nearest_5(prediction_horizon, 4)
        ph_subparts_list = [ph_quarter, ph_quarter * 2, ph_quarter * 3, prediction_horizon]

    c = generate_report.set_title(c, f'Error Grid Analysis')
    c = generate_report.draw_error_grid_table(c, df)

    c = generate_report.draw_scatter_plot(c, df, ph_subparts_list[0], 100, 360)
    c = generate_report.draw_scatter_plot(c, df, ph_subparts_list[1], 380, 360)

    if len(ph_subparts_list) > 2:
        c = generate_report.draw_scatter_plot(c, df, ph_subparts_list[2], 100, 120)
    if len(ph_subparts_list) > 3:
        c = generate_report.draw_scatter_plot(c, df, prediction_horizon, 380, 120)

    c = generate_report.set_bottom_text(c)
    c.showPage()

    # GLYCEMIA DETECTION
    c = generate_report.set_title(c, f'Glycemia Detection - Matthews Correlation Coefficient (MCC)')
    c = generate_report.draw_mcc_table(c, df)
    c = generate_report.plot_across_prediction_horizons(c, df, f'Matthews Correlation Coefficient (MCC)',
                                                        ['mcc_hypo', 'mcc_hyper'], y_placement=150,
                                                        height=4, y_label='MCC',
                                                        y_labels=['MCC Hypoglycemia', 'MCC Hyperglycemia'])
    c = generate_report.set_bottom_text(c)
    c.showPage()

    # GLYCEMIA DETECTION PAGE 2
    c = generate_report.set_title(c, f'Glycemia Detection - Confusion Matrices')

    # Plot confusion matrixes
    classes = ['Hypo', 'Target', 'Hyper']

    c = generate_report.plot_confusion_matrix(c, df, classes, ph_subparts_list[0], 50, 450)
    c = generate_report.plot_confusion_matrix(c, df, classes, ph_subparts_list[1], 320, 450)

    if len(ph_subparts_list) > 2:
        c = generate_report.plot_confusion_matrix(c, df, classes, ph_subparts_list[2], 50, 150)
    if len(ph_subparts_list) > 3:
        c = generate_report.plot_confusion_matrix(c, df, classes, prediction_horizon, 320, 150)

    c = generate_report.set_bottom_text(c)
    c.showPage()

    # PHYSIOLOGICAL ALIGNMENT PAGE
    c = generate_report.set_title(c, f'Physiological Alignment')
    c = generate_report.set_subtitle(c, f'Physiological Alignment for insulin', 720)
    c = generate_report.draw_physiological_alignment_single_dimension_table(c, df, 'bolus', 670)
    c = generate_report.plot_partial_dependencies_across_prediction_horizons(c, df, 'bolus', y_placement=440)

    c = generate_report.set_subtitle(c, f'Physiological Alignment for carbohydrates', 360)
    c = generate_report.draw_physiological_alignment_single_dimension_table(c, df, 'carbs', 310)
    c = generate_report.plot_partial_dependencies_across_prediction_horizons(c, df, 'carbs', y_placement=100)

    c = generate_report.set_bottom_text(c)
    c.showPage()

    # PREDICTION DISTRIBUTION PAGE
    c = generate_report.set_title(c, f'Distribution of Predictions')
    c = generate_report.plot_predicted_distribution(c, df, 100, 400)

    # Show the page
    c.showPage()

    # Save the PDF
    c.save()

    click.echo(f"An evaluation report for {results_file} is stored in '{results_file_path}' as '{results_file_name}'")


@click.command()
@click.option('--results-files', help='The name of the tested model results to evaluate, with ".csv". If '
                                      'None, all models will be tested.')
def generate_comparison_pdf(results_files):
    """
    This command stores a standardized pdf comparison report of the given models in data/reports/.
    """
    click.echo(f"Generating comparison report...")

    if results_files is None:
        results_files = helpers.list_files_in_directory('data/tested_models/')
    else:
        results_files = helpers.split_string(results_files)

    results_file_path = "data/reports/"

    timestamp = datetime.now().isoformat()
    safe_timestamp = timestamp.replace(':', '_')  # Windows does not allow ":" in file names
    safe_timestamp = safe_timestamp.replace('.', '_')

    results_file_name = f'comparison_report_{safe_timestamp}.pdf'

    dfs = []
    for results_file in results_files:
        df = generate_report.get_df_from_results_file(results_file)
        dfs += [df]

    # Create a PDF canvas
    c = canvas.Canvas(f"data/reports/{results_file_name}", pagesize=letter)

    # FRONT PAGE
    c = generate_report.set_title(c, f'Model Comparison Report')
    # TODO: Add front page summary of all parts and all the configurations
    c = generate_report.draw_overall_ranking_table(c, dfs, y_placement=700 - 20 * len(dfs))
    c = generate_report.set_bottom_text(c)
    c.showPage()

    # MODEL ACCURACY
    c = generate_report.set_title(c, f'Model Accuracy')
    c = generate_report.draw_model_comparison_accuracy_table(c, dfs, 'rmse', 700 - 20 * len(dfs))
    #c = generate_report.draw_model_comparison_accuracy_table(c, dfs, 'me', 550 - 20 * len(dfs))
    c = generate_report.plot_rmse_across_prediction_horizons(c, dfs, y_placement=200)
    c = generate_report.set_bottom_text(c)
    c.showPage()

    c = generate_report.set_title(c, f'Error Grid Analysis')
    c = generate_report.draw_model_comparison_error_grid_table(c, dfs, 700 - 20 * len(dfs))
    c = generate_report.plot_error_grid_across_prediction_horizons(c, dfs, y_placement=400 - 20 * len(dfs))
    c = generate_report.set_bottom_text(c)
    c.showPage()

    # GLYCEMIA DETECTION
    c = generate_report.set_title(c, f'Glycemia Detection')
    # TODO: Draw both in the table, and add a total score. Plot the total score for each model.
    c = generate_report.draw_model_comparison_glycemia_detection_table(c, dfs, 700 - 20 * len(dfs))
    c = generate_report.plot_mcc_across_prediction_horizons(c, dfs, y_placement=400 - 20 * len(dfs))
    c = generate_report.set_bottom_text(c)
    c.showPage()

    # PREDICTED DISTRIBUTION
    c = generate_report.set_title(c, f'Predicted Distribution')
    # TODO: Draw both in the table, and add a total score. Plot the total score for each model.
    c = generate_report.draw_model_comparison_predicted_distribution_table(c, dfs, 700 - 20 * len(dfs))
    c = generate_report.plot_predicted_dristribution_across_prediction_horizons(c, dfs, y_placement=400 - 20 * len(dfs))
    c = generate_report.set_bottom_text(c)
    c.showPage()

    # Save the PDF
    c.save()

    click.echo(f"An evaluation report for {results_files} is stored in '{results_file_path}' as '{results_file_name}'")


@click.command()
@click.option('--use-mgdl', type=bool, help='Set whether to use mg/dL or mmol/L', default=None)
def set_unit(use_mgdl):
    # Update the config
    unit_config_manager.use_mgdl = use_mgdl
    click.echo(f'Set unit to {"mg/dL" if use_mgdl else "mmol/L"}.')


# COMMAND FOR LOOP PREDICTION

@click.command()
@click.option('--parser', type=click.Choice(['nightscout']), default='nightscout',
              help='Data source parser (currently only nightscout supported)')
@click.option('--username', type=str, required=True,
              help='Nightscout URL (e.g., https://yoursite.herokuapp.com)')
@click.option('--password', type=str, required=True,
              help='Nightscout API key')
@click.option('--model', type=str, required=True,
              help='Model file name in ./data/trained_models/ (e.g., loop__my_config_3__60.pkl)')
@click.option('--prediction-time', type=str, required=False,
              help='Timestamp for prediction (format: dd-mm-yyyy/HH:MM). If not specified, uses current time')
def predict_loop2(parser, username, password, model, prediction_time):
    """Generate glucose predictions using downloaded historical data."""
    try:
        # 1. Setup and time handling
        local_tz = datetime.now().astimezone().tzinfo
        
        # Load model
        model_path = os.path.join('./data/trained_models/', model)
        if not os.path.exists(model_path):
            click.echo(f"Error: Model file not found at {model_path}")
            return

        with open(model_path, 'rb') as f:
            model_instance = pickle.load(f)

        # Parse prediction time
        if prediction_time:
            try:
                local_pred_time = datetime.strptime(prediction_time, "%d-%m-%Y/%H:%M")
                local_pred_time = local_pred_time.replace(tzinfo=local_tz)
            except ValueError:
                click.echo("Error: prediction-time must be in format 'dd-mm-yyyy/HH:MM'")
                return
        else:
            local_pred_time = datetime.now().astimezone(local_tz)

        utc_pred_time = local_pred_time.astimezone(timezone.utc)
        
        # 2. Get required data window (24h + DIA)
        required_history = timedelta(hours=24) + timedelta(minutes=model_instance.DIA)
        utc_start = utc_pred_time - required_history
        utc_end = utc_pred_time + timedelta(hours=1)  # Include next hour for future events

        # 3. Get data using parser
        parser_module = importlib.import_module(f'glupredkit.parsers.{parser}')
        parser_instance = parser_module.Parser()
        df = parser_instance(
            start_date=utc_start,
            end_date=utc_end,
            username=username,
            password=password
        )

        if df.empty:
            click.echo("\nError: No data retrieved")
            return

        # 4. Calculate daily statistics from parser output
        last_24h_end = utc_pred_time
        last_24h_start = last_24h_end - timedelta(hours=24)
        
        # Use the known basal rate from profile (0.9-1.0 U/hr)
        basal_rate = 0.9  # U/hr (using conservative value)
        
        # Calculate totals
        total_basal = basal_rate * 24  # U/hr * 24 hours
        
        mask = (df.index >= last_24h_start) & (df.index <= last_24h_end)
        total_bolus = df.loc[mask, 'bolus'].sum()
        total_insulin = total_basal + total_bolus
        current_basal_rate = basal_rate

        # 5. Prepare historical data for prediction
        window_size = model_instance.DIA // 5  # Convert DIA minutes to number of 5-min intervals
        historical_window = df[df.index <= utc_pred_time].tail(window_size).copy()
        latest_time = historical_window.index[-1]

        # Create prediction row with historical data
        prediction_data = {
            'CGM': [historical_window['CGM'].iloc[-1]],
            'basal': [historical_window['basal'].iloc[-1]],
            'bolus': [historical_window['bolus'].iloc[-1]],
            'carbs': [historical_window['carbs'].iloc[-1]]
        }

        # Add historical CGM values
        for i in range(1, len(historical_window)):
            prediction_data[f'CGM_minus_{i*5}'] = [historical_window['CGM'].iloc[-(i+1)]]

        # Add historical bolus and carb values
        for i in range(1, len(historical_window)):
            prediction_data[f'bolus_minus_{i*5}'] = [historical_window['bolus'].iloc[-(i+1)]]
            prediction_data[f'carbs_minus_{i*5}'] = [historical_window['carbs'].iloc[-(i+1)]]

        # Get future events (next hour)
        future_df = df[df.index > utc_pred_time].copy()
        
        # Add what-if projections for next hour
        future_times = pd.date_range(
            start=latest_time + pd.Timedelta(minutes=5),
            periods=12,
            freq='5min'
        )

        # Add projections including any known future events
        for i, future_time in enumerate(future_times, 1):
            time_diff = i * 5
            
            # If we have data for this future time, use it
            if future_time in future_df.index:
                prediction_data[f'basal_what_if_{time_diff}'] = [future_df.loc[future_time, 'basal']]
                prediction_data[f'bolus_what_if_{time_diff}'] = [future_df.loc[future_time, 'bolus']]
                prediction_data[f'carbs_what_if_{time_diff}'] = [future_df.loc[future_time, 'carbs']]
            else:
                # Otherwise use current basal rate and zero for bolus/carbs
                prediction_data[f'basal_what_if_{time_diff}'] = [df['basal'].iloc[-1]]
                prediction_data[f'bolus_what_if_{time_diff}'] = [0]
                prediction_data[f'carbs_what_if_{time_diff}'] = [0]

        # Create prediction DataFrame
        prediction_row = pd.DataFrame(prediction_data, index=[latest_time])

        # Debug future events
        click.echo("\nFuture events in prediction window:")
        future_events = future_df[(future_df['carbs'] > 0) | (future_df['bolus'] > 0)]
        for idx, row in future_events.iterrows():
            local_time = idx.astimezone(local_tz)
            if row['carbs'] > 0:
                click.echo(f"Carbs at {local_time.strftime('%H:%M')}: {row['carbs']}g")
            if row['bolus'] > 0:
                click.echo(f"Bolus at {local_time.strftime('%H:%M')}: {row['bolus']}U")

        # 6. Make prediction using fixed ISF/CR values
        trained_isf = model_instance.insulin_sensitivity_factor[0] if model_instance.insulin_sensitivity_factor else None
        trained_cr = model_instance.carb_ratio[0] if model_instance.carb_ratio else None
        
        click.echo("\nModel's trained optimal parameters:")
        click.echo(f"Trained ISF: {trained_isf:.1f} mg/dL/U" if trained_isf is not None else "Trained ISF: Not found in model")
        click.echo(f"Trained CR: {trained_cr:.1f} g/U" if trained_cr is not None else "Trained CR: Not found in model")
        
        # Currently using hardcoded values
        isf = 36  # Fixed ISF for testing
        cr = 10   # Fixed CR for testing
        
        click.echo("\nCurrently using hardcoded parameters:")
        click.echo(f"Current ISF: {isf:.1f} mg/dL/U")
        click.echo(f"Current CR: {cr:.1f} g/U")

        output_dict = model_instance.get_prediction_output(
            df_row=prediction_row.iloc[0],
            input_dict=model_instance.get_input_dict(
                insulin_sensitivity=trained_isf,
                carb_ratio=trained_cr,
                basal=current_basal_rate
            ),
            time_to_calculate=latest_time
        )

        if not output_dict or "predicted_glucose_values" not in output_dict:
            click.echo("\nError: Failed to generate predictions")
            return

        predictions = output_dict["predicted_glucose_values"][:12]

        # 7. Output results
        click.echo("\nDaily Statistics (Last 24 Hours):")
        click.echo(f"Total Daily Insulin: {total_insulin:.2f} U")
        click.echo(f"Total Basal: {total_basal:.2f} U")
        click.echo(f"Total Bolus: {total_bolus:.2f} U")
        click.echo(f"Current Basal Rate: {current_basal_rate:.3f} U/hr")

        click.echo(f"\nPrediction Parameters:")
        click.echo(f"ISF: {isf:.1f} mg/dL/U")
        click.echo(f"CR: {cr:.1f} g/U")

        click.echo("\nHistorical Data Summary:")
        click.echo(f"Latest CGM: {prediction_row['CGM'].iloc[0]}")
        click.echo(f"Historical CGM values: {len([col for col in prediction_row.columns if 'CGM_minus_' in col])}")
        click.echo(f"Historical bolus values: {len([col for col in prediction_row.columns if 'bolus_minus_' in col])}")
        click.echo(f"Future intervals: {len([col for col in prediction_row.columns if 'what_if' in col])}")

        click.echo("\nPredictions for the next hour:")
        for i, pred in enumerate(predictions):
            pred_time = utc_pred_time + timedelta(minutes=5*i)
            local_time = pred_time.astimezone(local_tz)
            click.echo(f"{local_time.strftime('%Y-%m-%d %H:%M')}: {pred:.1f} mg/dL")

        return predictions

    except Exception as e:
        click.echo(f"\nError during prediction: {str(e)}")
        raise


# Create the CLI group OUTSIDE of any function
cli = click.Group(commands={
    'setup_directories': setup_directories,
    'parse': parse,
    'generate_config': generate_config,
    'train_model': train_model,
    'evaluate_model': evaluate_model,
    'draw_plots': draw_plots,
    'generate_evaluation_pdf': generate_evaluation_pdf,
    'generate_comparison_pdf': generate_comparison_pdf,
    'set_unit': set_unit,
    'predict_loop2': predict_loop2
})

if __name__ == "__main__":
    cli()
