import pandas as pd
import numpy as np
from glupredkit.helpers.model_config_manager import ModelConfigurationManager

# ORIGINAL FUNCTION
# def prepare_sequences(df_X, df_y, window_size, what_if_columns, prediction_horizon, real_time, step_size=1):
#     X, y, dates = [], [], []
#     target_columns = df_y.columns
#     exclude_list = list(target_columns) + ["imputed", "iob", "cob", "carbs"]
#     sequence_columns = [item for item in df_X.columns if item not in exclude_list]
#     n_what_if = prediction_horizon // 5

#     print("Preparing sequences...")

#     for i in range(0, len(df_X) - window_size - n_what_if, step_size):
#         label = df_y.iloc[i + window_size - 1]

#         if df_X.iloc[i:i + window_size + n_what_if].isnull().any().any():
#             continue  # Skip this sequence if there are NaN values in the input data

#         if 'imputed' in df_X.columns:
#             imputed = df_X['imputed'].iloc[i + window_size - 1]
#             if imputed:
#                 continue  # Skip this sequence

#         if not real_time:
#             if pd.isna(label).any():
#                 continue  # Skip this sequence

#         # sequence = df_X[sequence_columns][i:i + window_size]
#         date = df_y.index[i + window_size - 1]

#         # Initialize an empty DataFrame for mixed_sequence with the same columns as sequence
#         mixed_sequence = pd.DataFrame(index=df_X[i:i + window_size + n_what_if].index, columns=sequence_columns).iloc[0:window_size + n_what_if]

#         for col in sequence_columns:
#             if col in what_if_columns:
#                 # For "what if" columns, extend the sequence
#                 mixed_sequence[col] = df_X[col][i:i + window_size + n_what_if]
#             else:
#                 # Slice the original window size for other columns
#                 original_sequence = df_X[col][i:i + window_size]
#                 # Create a padding series with -1 values and the appropriate date index
#                 padding_index = df_X.index[i + window_size:i + window_size + n_what_if]
#                 padding_series = pd.Series([-1] * n_what_if, index=padding_index)
#                 # Concatenate the original sequence with the padding series
#                 full_sequence = pd.concat([original_sequence, padding_series])
#                 mixed_sequence[col] = full_sequence

#         X.append(mixed_sequence)
#         y.append(label.values.tolist())
#         dates.append(date)

#     return np.array(X), np.array(y), dates

# WITH ADDED LOGGING
def prepare_sequences(df_X, df_y, window_size, what_if_columns, prediction_horizon, real_time, step_size=1):
    X, y, dates = [], [], []
    target_columns = df_y.columns
    exclude_list = list(target_columns) + ["imputed", "iob", "cob"]
    sequence_columns = [item for item in df_X.columns if item not in exclude_list]
    n_what_if = prediction_horizon // 5

    print(f"\nDetailed Debug Information:")
    print(f"Window size: {window_size}")
    print(f"Prediction horizon: {prediction_horizon}")
    print(f"Data points available: {len(df_X)}")
    print(f"Required points for one sequence: {window_size + n_what_if}")
    print(f"Sequence columns: {sequence_columns}")
    print(f"Target columns: {target_columns}")
    print(f"Real time mode: {real_time}")
    print(f"\nFirst few rows of input data:")
    print(df_X.head())
    print(f"\nLast few rows of input data:")
    print(df_X.tail())
    
    if len(df_X) < window_size + n_what_if:
        print(f"ERROR: Not enough data points. Have {len(df_X)}, need {window_size + n_what_if}")
        return np.array([]), np.array([]), []

    print("\nAttempting sequence creation...")
    possible_sequences = len(df_X) - window_size - n_what_if
    print(f"Theoretically possible sequences: {possible_sequences}")
    
    for i in range(0, len(df_X) - window_size - n_what_if, step_size):
        print(f"\nTrying sequence at position {i}:")
        print(f"Window range: {i} to {i + window_size}")
        print(f"What-if range: {i + window_size} to {i + window_size + n_what_if}")
        
        # Get label and print its value
        label = df_y.iloc[i + window_size - 1]
        print(f"Label at position {i + window_size - 1}: {label}")
        
        # Check for NaN values in the sequence range
        sequence_data = df_X.iloc[i:i + window_size + n_what_if]
        if sequence_data.isnull().any().any():
            print(f"Skipped: NaN values found in sequence data")
            continue

        # Check imputed flag if it exists
        if 'imputed' in df_X.columns:
            imputed = df_X['imputed'].iloc[i + window_size - 1]
            if imputed:
                print(f"Skipped: Found imputed value")
                continue

        # Check label validity for non-real-time mode
        if not real_time:
            if pd.isna(label).any():
                print(f"Skipped: NaN in label (non-real-time mode)")
                continue

        date = df_y.index[i + window_size - 1]
        print(f"Using date: {date}")
        
        try:
            # Create the sequence
            mixed_sequence = pd.DataFrame(index=df_X[i:i + window_size + n_what_if].index, 
                                        columns=sequence_columns).iloc[0:window_size + n_what_if]
            
            for col in sequence_columns:
                if col in what_if_columns:
                    mixed_sequence[col] = df_X[col][i:i + window_size + n_what_if]
                else:
                    original_sequence = df_X[col][i:i + window_size]
                    padding_index = df_X.index[i + window_size:i + window_size + n_what_if]
                    padding_series = pd.Series([-1] * n_what_if, index=padding_index)
                    full_sequence = pd.concat([original_sequence, padding_series])
                    mixed_sequence[col] = full_sequence
            
            X.append(mixed_sequence)
            y.append(label.values.tolist())
            dates.append(date)
            print(f"Successfully created sequence")
            
        except Exception as e:
            print(f"Error creating sequence: {str(e)}")
            continue

    total_sequences = len(X)
    print(f"\nFinal Summary:")
    print(f"Total sequences created: {total_sequences}")
    if total_sequences > 0:
        print(f"First sequence shape: {X[0].shape}")
        print(f"First target shape: {len(y[0])}")
    
    return np.array(X), np.array(y), dates

def create_dataframe(sequences, targets, dates):
    # Convert sequences to lists
    sequences_as_strings = [str(list(map(list, seq))) for seq in sequences]
    targets_as_strings = [','.join(map(str, target)) for target in targets]

    dataset_df = pd.DataFrame({
        'date': dates,
        'sequence': sequences_as_strings,  # sequences_as_lists
        'target': targets_as_strings
    })
    return dataset_df.set_index('date')


def process_data(df, model_config_manager: ModelConfigurationManager, real_time=False):
    target_columns = [col for col in df.columns if col.startswith('target')]
    df_X, df_y = df.drop(target_columns, axis=1), df[target_columns]

    # Add sliding windows of features
    sequences, targets, dates = prepare_sequences(df_X, df_y, window_size=model_config_manager.get_num_lagged_features(),
                                                  what_if_columns=model_config_manager.get_what_if_features(),
                                                  prediction_horizon=model_config_manager.get_prediction_horizon(),
                                                  real_time=real_time)

    # Store as a dataframe with two columns: targets and sequences
    df = create_dataframe(sequences, targets, dates)

    return df

