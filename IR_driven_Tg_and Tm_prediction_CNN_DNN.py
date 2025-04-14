# -*- coding: utf-8 -*-
"""
Created on April 14 2025
Employed neural network models: Convolutional Neural Network, Deep Neural Network 

Input: IR Spectroscopy Data
Output: Tg or Tm, the output can be selected changing the target dataframe

@author: Gorkem Anil Al
"""

import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, BatchNormalization, ReLU
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import os
import re


# Initialize dictionaries
Tg = {}
Tm = {}
data_Tg = {}
classification = {}
data_Tm = {}

meta = np.loadtxt('meta.csv', dtype=float, usecols = (0,1,2), delimiter=',')
for lines in meta:
    Tg.update({int(lines[0]): (lines[1])})
    Tm.update({int(lines[0]): (lines[2])})
    
graphs = np.loadtxt('graphs.csv', dtype=float, usecols = (0,1), delimiter=',')
for lines in graphs:
    temp = Tg.get(lines[1])
    Tm_temp = Tm.get(lines[1])
    data_Tg.update({int(lines[0]): temp})
    data_Tm.update({int(lines[0]): Tm_temp})
    classification.update({int(lines[0]): str(int(lines[1]))})
  
#%%

def extract_number(filename):
    # This regular expression extracts numbers from a filename
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return float('inf')  # Return a large number if no digits found


# Assuming all CSV files are in the directory 'data_directory'
data_directory = 'D:/university of bath postdoc files/files from x drive/codes_to_submit/data'
file_names = os.listdir(data_directory)

# Sort the file names based on numerical values extracted
file_names.sort(key=extract_number)

# Initialize an empty dictionary to store the data
csv_data = {}
        
# Loop through all file names and load them into the 
for file_name in file_names:
    if file_name.endswith('.csv'):  # Make sure to check the file extension
        file_path = os.path.join(data_directory, file_name)
        # Read the CSV file with specified column names
        # If your CSV files have a header, add the parameter header=0 to replace it
        df = pd.read_csv(file_path, names=['wavelength', 'absorbance'], header=None)
        # The file name without the extension can be the key
        key = file_name.rsplit('.', 1)[0]
        csv_data[key] = df

#%%  find which dataframes have the minimum and maximum data points

min_data_points = float('inf')  # Initialize with infinity for minimum
max_data_points = float('-inf')  # Initialize with negative infinity for maximum
min_key = None  # Variable to store the key with the minimum data points
max_key = None  # Variable to store the key with the maximum data points

# Iterate over all DataFrames in the dictionary to find the minimum and maximum number of data points
for key, df in csv_data.items():
    num_points = len(df)
    if num_points < min_data_points:
        min_data_points = num_points
        min_key = key
    if num_points > max_data_points:
        max_data_points = num_points
        max_key = key

# Outputting the results
if min_key is not None and max_key is not None:
    print(f"The DataFrame with the minimum number of data points is from key '{min_key}' with {min_data_points} data points.")
    print(f"The DataFrame with the maximum number of data points is from key '{max_key}' with {max_data_points} data points.")
else:
    print("No DataFrames found in the dictionary.")

#%% Interpolate all the data in selected_dataframes according to the
# maximum number of data points
#Use this method for the all data 

from scipy.interpolate import interp1d

# Initialize min_wavelength to a high value
min_wavelength = float('inf')

# Iterate over each DataFrame in selected_dataframes to find the minimum wavelength
for data in csv_data.values():
    current_min = data.iloc[:, 0].min()
    if current_min < min_wavelength:
        min_wavelength = current_min

# Initialize max_wavelength to a low value
max_wavelength = float('-inf')

# Iterate over each DataFrame in selected_dataframes to find the maximum wavelength
for data in csv_data.values():
    current_max = data.iloc[:, 0].max()
    if current_max > max_wavelength:
        max_wavelength = current_max
        
        
# Determine the minimum and maximum number of data points
min_num_points = min(len(data) for data in csv_data.values())
max_num_points = max(len(data) for data in csv_data.values())

# Create the common range of wavelengths based on the minimum and maximum number of data points
common_wavelengths = np.linspace(min_wavelength, max_wavelength, max_num_points)
#=================================================================================================
# Create a dictionary to store interpolation functions for each dataset
interp_func_dict = {}

# Interpolate each dataset using the common range of wavelengths
for key, data in csv_data.items():
    # Interpolate only if the number of data points is less than or equal to max_num_points
    if len(data) <= max_num_points:
         interp_func_dict[key] = interp1d(data.iloc[:, 0], data.iloc[:, 1], kind='nearest', fill_value="extrapolate")
        #interp_func_dict[key] = CubicSpline(data.iloc[:, 0], data.iloc[:, 1], bc_type='natural')    
     
#%% After interpolation, organize machine learning dataset
machineLearningData = {}
for key, data in csv_data.items():
 #   Obtain the interpolated intensity values for the common wavelengths
    interpolated_intensities = interp_func_dict[key](common_wavelengths)
# Create a DataFrame with common_wavelengths as the second column and interpolated intensity values as the first column
    df = pd.DataFrame({'Wavelength': common_wavelengths, 'Intensity': interpolated_intensities})
    
    # Add the DataFrame to the machineLearningData dictionary
    machineLearningData[key] = df

# Prepare a Target dataset from Tg (glass transition temperature)
ML_Target = pd.DataFrame()
ML_Target = data_Tg

ML_Tm_target = pd.DataFrame()
ML_Tm_target = data_Tm
    
intensity_list = []
for key in machineLearningData.keys():
    intensity_series = machineLearningData[key]['Intensity']
    intensity_list.append(intensity_series.values)

#Input dataset; X composed of FTIR signals
combined_df = pd.DataFrame(intensity_list)
X = combined_df

y = ML_Target

#Target Values for Glass Transition Temperature
y_Tg = pd.DataFrame(list(ML_Target.values()), columns=['Value']) #Target dataset for Tg
#Target Values for Melting Transition Temperature
y_Tm = pd.DataFrame(list(ML_Tm_target.values()), columns=['Value']) # Target dataset for Tm

#%% Prepare the Dataset, 
# y_Tg for the glass transition temperature target dataset
# t_Tm for the melting temperature target dataset

X_train, X_test, y_train, y_test = train_test_split(X, y_Tg, test_size=0.2, random_state=40)
#%% Grid Search for DNN
def evaluate_dnn_model(inputData, outputData, num_folds=5, epochs=150, batch_size=3, random_state=42):
    """
    Evaluate a DNN model using K-Fold cross-validation and Grid Search hyperparameter tuning.
    """
    # Define K-Fold Cross Validator
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    fold_no = 1
    loss_per_fold = []
    r2_per_fold = []
    loss_mse_per_fold = []
    best_model_paths = []
    all_fold_results = []

    # Define the model creation function with hyperparameters
    def create_model(neurons_layer1=128, neurons_layer2=64, neurons_layer3=32, neurons_layer4=16, neurons_layer5=8, learning_rate=0.001):
        input_layer = Input(shape=(inputData.shape[1],))

        x = Dense(neurons_layer1, activation='relu')(input_layer)
        x = Dropout(0.3)(x)
        x = Dense(neurons_layer2, activation='relu')(x)
        x = Dense(neurons_layer3, activation='relu')(x)
        x = Dense(neurons_layer4, activation='relu')(x)
        x = Dense(neurons_layer5, activation='relu')(x)

        output_layer = Dense(1, activation='linear')(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    # Grid of hyperparameters to search
    param_grid = {
        'neurons_layer1': [64, 128, 256],
        'neurons_layer2': [64, 128, 256],
        'neurons_layer3': [32, 64, 128],
        'neurons_layer4': [16, 32, 64],
        'neurons_layer5': [8, 16, 32],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'epochs': [epochs],
        'batch_size': [batch_size],
        'verbose': [0]
    }

    # Wrap model into scikit-learn compatible model
    model_wrapper = KerasRegressor(build_fn=create_model)

    # Grid search with cross-validation (3-fold)
    grid = GridSearchCV(estimator=model_wrapper, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_result = grid.fit(inputData, outputData)

    # Get the best hyperparameters
    best_params = grid_result.best_params_
    print(f"Best hyperparameters found: {best_params}")

    # Loop through each fold
    for train_index, test_index in kf.split(inputData, outputData):
        print(f'Training fold {fold_no}...')

        X_train, X_test = inputData.iloc[train_index], inputData.iloc[test_index]
        y_train, y_test = outputData.iloc[train_index], outputData.iloc[test_index]

        # Create model with best hyperparameters
        model = create_model(neurons_layer1=best_params['neurons_layer1'],
                             neurons_layer2=best_params['neurons_layer2'],
                             neurons_layer3=best_params['neurons_layer3'],
                             neurons_layer4=best_params['neurons_layer4'],
                             neurons_layer5=best_params['neurons_layer5'],
                             learning_rate=best_params['learning_rate'])

        best_model_path = f'best_model_fold_{fold_no}.h5'
        best_model_paths.append(best_model_path)

        model.fit(X_train, y_train,
                  epochs=best_params['epochs'],
                  batch_size=best_params['batch_size'],
                  verbose=0)

        y_pred = model.predict(X_test)
        y_pred = y_pred.squeeze()
        y_test = y_test.values.squeeze()

        loss = mean_absolute_error(y_test, y_pred)
        loss_mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f'Fold {fold_no} - MAE: {loss}, R²: {r2}, MSE: {loss_mse}')

        fold_results = {
            'Fold': fold_no,
            'MAE': loss,
            'MSE': loss_mse,
            'R²': r2,
        }
        all_fold_results.append(fold_results)

        loss_per_fold.append(loss)
        loss_mse_per_fold.append(loss_mse)
        r2_per_fold.append(r2)

        # Save fold predictions to CSV
        results_df = pd.DataFrame({
            'Index': test_index,
            'Actual': y_test,
            'Predicted': y_pred,
            'Difference': y_pred - y_test
        })
        results_df.to_csv(f'fold_{fold_no}_results.csv', index=False)

        # Plot and save
        plt.figure(figsize=(10, 6))
        sns.regplot(x=y_test, y=y_pred, ci=95, line_kws={"color": "black"}, scatter_kws={"s": 100})
        plt.xlabel('Actual Tm (K)', fontsize=16)
        plt.ylabel('Predicted Tm (K)', fontsize=16)
        plt.title(f'ANN - Fold {fold_no}')
        plt.text(0.05, 0.95, f'R² = {r2:.2f}', transform=plt.gca().transAxes,
                 fontsize=14, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(f'R2_plot_fold_{fold_no}.png')
        plt.close()

        fold_no += 1

    # Save aggregated results
    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv('dnn_fold_results.csv', index=False)
    print("Saved fold results to dnn_fold_results.csv")

    # Save models
    for i, model_path in enumerate(best_model_paths):
        model.save(model_path)
    print("Saved the best models for each fold.")

    print(f'Mean MAE: {np.mean(loss_per_fold)}, Std MAE: {np.std(loss_per_fold)}')
    print(f'Mean R²: {np.mean(r2_per_fold)}, Std R²: {np.std(r2_per_fold)}')

    # Save metrics separately
    pd.DataFrame({'Fold': range(1, num_folds + 1), 'MAE': loss_per_fold}).to_csv('dnn_mae_per_fold.csv', index=False)
    pd.DataFrame({'Fold': range(1, num_folds + 1), 'R²': r2_per_fold}).to_csv('dnn_r2_per_fold.csv', index=False)
    pd.DataFrame({'Fold': range(1, num_folds + 1), 'MSE': loss_mse_per_fold}).to_csv('dnn_mse_per_fold.csv', index=False)

    return loss_per_fold, r2_per_fold, loss_mse_per_fold, best_model_paths

#%% Run DNN

inputData = X # Your feature data (e.g., a pandas DataFrame)
outputData = y_Tg  # Your target data (e.g., a pandas Series or DataFrame)

# Call the function to evaluate the model
loss_per_fold, r2_per_fold, loss_mse_per_fold, best_model_paths = evaluate_dnn_model(
    inputData=inputData,
    outputData=outputData,
    num_folds=5,  # Number of folds for cross-validation
    epochs=150,   # Number of epochs for training
    batch_size=3, # Batch size for training
    random_state=42  # For reproducibility
)

#%%

import numpy as np
import pandas as pd
import seaborn as sns


def create_cnn_model(filters_1=256, filters_2=64, filters_3=48, kernel_size=3, dense_units=200, learning_rate=5e-4, input_shape=(714,1)):
    input_layer = Input(shape=input_shape)

    x = Conv1D(filters=filters_1, kernel_size=kernel_size, use_bias=False)(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=filters_2, kernel_size=kernel_size)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=filters_3, kernel_size=kernel_size)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Flatten()(x)
    x = Dense(dense_units, activation='relu', kernel_regularizer='l2')(x)

    output_layer = Dense(1, activation='linear')(x)

    lr_schedule = ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=50000,
        decay_rate=0.90
    )

    optimizer = Adam(learning_rate=lr_schedule)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def evaluate_cnn_model(inputData, outputData, num_folds=5, epochs=50, batch_size=3, random_state=42):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    fold_no = 1
    loss_per_fold = []
    r2_per_fold = []
    loss_mse_per_fold = []
    best_model_paths = []
    all_fold_results = []

    number_of_attributes = inputData.shape[1]

    if not os.path.exists('results'):
        os.makedirs('results')

    # Wrap model for sklearn compatibility
    model_wrapper = KerasRegressor(build_fn=create_cnn_model, input_shape=(number_of_attributes,1))

    # Define hyperparameter grid
    param_grid = {
        'filters_1': [128, 256],
        'filters_2': [32, 64],
        'filters_3': [32, 48],
        'kernel_size': [3, 5],
        'dense_units': [128, 200, 256],
        'learning_rate': [5e-4],
        'epochs': [epochs],
        'batch_size': [batch_size],
        'verbose': [0]
    }

    grid = GridSearchCV(estimator=model_wrapper, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_result = grid.fit(np.expand_dims(inputData, axis=2), outputData)

    best_params = grid_result.best_params_
    print(f"Best hyperparameters found: {best_params}")

    for train_index, test_index in kf.split(inputData):
        print(f'Training fold {fold_no}...')

        X_train, X_test = inputData.iloc[train_index], inputData.iloc[test_index]
        y_train, y_test = outputData.iloc[train_index], outputData.iloc[test_index]

        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)

        model = create_cnn_model(
            filters_1=best_params['filters_1'],
            filters_2=best_params['filters_2'],
            filters_3=best_params['filters_3'],
            kernel_size=best_params['kernel_size'],
            dense_units=best_params['dense_units'],
            learning_rate=best_params['learning_rate'],
            input_shape=(number_of_attributes, 1)
        )

        best_model_path = f'results/best_model_fold_{fold_no}.h5'
        best_model_paths.append(best_model_path)

        model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], validation_split=0.2, verbose=0)

        y_pred = model.predict(X_test).squeeze()
        y_test = y_test.values.squeeze()

        loss = mean_absolute_error(y_test, y_pred)
        loss_mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f'Fold {fold_no} - MAE: {loss}, R²: {r2}, MSE: {loss_mse}')

        fold_results = {
            'Fold': fold_no,
            'MAE': loss,
            'MSE': loss_mse,
            'R²': r2,
        }
        all_fold_results.append(fold_results)

        loss_per_fold.append(loss)
        loss_mse_per_fold.append(loss_mse)
        r2_per_fold.append(r2)

        results_df = pd.DataFrame({
            'Index': test_index,
            'Actual': y_test,
            'Predicted': y_pred,
            'Difference': y_pred - y_test
        })
        results_df.to_csv(f'fold_{fold_no}_results.csv', index=False)

        plt.figure(figsize=(10, 6))
        sns.regplot(x=y_test, y=y_pred, ci=95, line_kws={"color": "black"}, scatter_kws={"s": 100})
        plt.xlabel('Actual Tm (K)', fontsize=16)
        plt.ylabel('Predicted Tm (K)', fontsize=16)
        plt.title(f'CNN - Fold {fold_no}')
        plt.text(0.05, 0.95, f'R² = {r2:.2f}', transform=plt.gca().transAxes,
                 fontsize=14, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(f'R2_plot_fold_{fold_no}.png')
        plt.close()

        fold_no += 1

    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv('cnn_fold_results.csv', index=False)
    print("Saved fold results to cnn_fold_results.csv")

    for i, model_path in enumerate(best_model_paths):
        model.save(model_path)
    print("Saved the best models for each fold.")

    print(f'Mean MAE: {np.mean(loss_per_fold)}, Std MAE: {np.std(loss_per_fold)}')
    print(f'Mean R²: {np.mean(r2_per_fold)}, Std R²: {np.std(r2_per_fold)}')

    pd.DataFrame({'Fold': range(1, num_folds + 1), 'MAE': loss_per_fold}).to_csv('cnn_mae_per_fold.csv', index=False)
    pd.DataFrame({'Fold': range(1, num_folds + 1), 'R²': r2_per_fold}).to_csv('cnn_r2_per_fold.csv', index=False)
    pd.DataFrame({'Fold': range(1, num_folds + 1), 'MSE': loss_mse_per_fold}).to_csv('cnn_mse_per_fold.csv', index=False)

    return loss_per_fold, r2_per_fold, loss_mse_per_fold, best_model_paths


#%% Run CNN

inputData = X  # Your feature data (e.g., a pandas DataFrame)
outputData = y_Tg  # Your target data (e.g., a pandas Series or DataFrame)

# Call the evaluate_cnn_model function
loss_per_fold, r2_per_fold, loss_mse_per_fold, best_model_paths = evaluate_cnn_model(
    inputData=inputData, 
    outputData=outputData, 
    num_folds=5,     # Number of K-Fold splits (default is 5)
    epochs=50,       # Number of epochs for training (default is 50)
    batch_size=3,    # Batch size for model training (default is 3)
    random_state=42  # Random state for reproducibility
)

