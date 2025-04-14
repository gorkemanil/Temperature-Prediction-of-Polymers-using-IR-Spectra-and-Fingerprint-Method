# -*- coding: utf-8 -*-
"""
Created on April 14 2025
Employed machine learning models:

Input: IR Spectroscopy Data
Output: Tg or Tm, the output can be selected changing the target dataframe

@author: Gorkem Anil Al
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
data_directory = '' #write here the directory path

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

#%% Plot the histogram and density plot of Tg
import seaborn as sns

min_temperature = min(data_Tg.values())
max_temperature = max(data_Tg.values())

temperature_data = list(data_Tg.values())
# Plotting the histogram and density plot
plt.figure(figsize=(10, 6))
sns.histplot(temperature_data, kde=False, bins=30, color='green')
# Set the x-axis limit to start from 100
#plt.xlim(100, max(temperature_data))  # Replace 100 with your desired starting value

#plt.title('Histogram of Temperature Data')
plt.xlabel('Tg (K)', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.grid(True)

# Show the plot
plt.show()


#Plot the histogram and density plot of Tm

Tm_temperature_data = list(data_Tm.values())
# Plotting the histogram and density plot
plt.figure(figsize=(10, 6))
sns.histplot(Tm_temperature_data, kde=False, bins=30, color='green')
# Set the x-axis limit to start from 100
#plt.xlim(100, max(temperature_data))  # Replace 100 with your desired starting value

#plt.title('Histogram of Temperature Data')
plt.xlabel('Tm (K)', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.grid(True)

# Show the plot
plt.show()


#Plot Tg of all polymers in the Dataset
Tg_all_polymers = list(Tg.values())
# Plotting the histogram and density plot
plt.figure(figsize=(10, 6))
sns.histplot(Tg_all_polymers, kde=False, bins=30, color='green')
# Set the x-axis limit to start from 100
#plt.xlim(100, max(temperature_data))  # Replace 100 with your desired starting value

#plt.title('Histogram of Temperature Data')
plt.xlabel('Tg (K)', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Set integer y-axis ticks
max_freq = max(np.histogram(Tg_all_polymers, bins=30)[0])  # Get max frequency from histogram
plt.yticks(range(0, max_freq + 1, 5), fontsize=14)  # Adjust step size (5) as needed

plt.grid(True)

# Show the plot
plt.show()

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
        
    
#%% Plot the minimum and maximum
# (csv_data)[780-3] gives (csv_data)[780]
data1= csv_data[list(csv_data)[780-3]] 
data2= csv_data[list(csv_data)[155-3]]


plt.figure(figsize=(10, 6))  # Width=10 inches, Height=6 inches
plt.plot(data1['wavelength'], data1['absorbance'])
plt.plot(data2['wavelength'], data2['absorbance'], linewidth=3)
plt.xlabel('Wavelength', fontsize=16)
plt.ylabel('Absorbance', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Turn off the grid
plt.grid(False)

plt.show()

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


#%% Plot interpolated and original data

plt.figure(1, figsize=(10, 6)) 

data = csv_data[list(csv_data)[1]]  # Adjust the key as needed
plt.plot(data['wavelength'], data['absorbance'], label=f'Measured Data', linewidth=3)

interpolated_data =  machineLearningData[list(machineLearningData)[1]]
plt.plot(interpolated_data['Wavelength'], interpolated_data['Intensity'], label=f'Processed Data', linewidth=3, linestyle='--') 

plt.xlabel('Wavelength (cm$^{-1}$)', fontsize=14)
plt.ylabel('Absorbance', fontsize=14)
plt.title('Measured and Processed Data', fontsize=16)
plt.legend()

# Change the size of the tick labels on x and y axes
plt.tick_params(axis='both', which='major', labelsize=12)

# Save the plot
#plt.savefig('absorbance_vs_wavelength.png')

plt.show()


#%% PCA method

from sklearn.decomposition import PCA

# PCA 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled)# convert to dataframe

pca = PCA()
X_pca = pca.fit(X_scaled)

# Calculate cumulative explained variance
cumulative_explained_variance = np.cumsum(X_pca.explained_variance_ratio_)

# Determine the number of components to retain 95% variance
threshold = 0.95
num_components = np.argmax(cumulative_explained_variance >= threshold) + 1

# Fit PCA with the selected number of components
pca_final = PCA(n_components=num_components)
X_reduced = pca_final.fit_transform(X_scaled)
#Convert X_reduced from array of float to dataframe to use for ML models
X_reduced = pd.DataFrame(X_reduced)

print(f"Number of components to retain 95% variance: {num_components}")


# Plot cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--', color='b')
#plt.title('Cumulative Explained Variance by Number of Principal Components',fontsize=14)
plt.xlabel('Number of Principal Components', fontsize=16)
plt.ylabel('Explained Variance', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.96, '95% Threshold', color='red', fontsize=12)
plt.show()


#%% Prepare the Dataset, 
# y_Tg for the glass transition temperature target dataset
# t_Tm for the melting temperature target dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_Tg, test_size=0.2, random_state=40)


#%%
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer

def evaluate_model(X, y, model, param_grid, cv=5, scoring=None, refit_metric='r2', plot=True, plot_title=None):
    """
    Evaluate a machine learning model using grid search and cross-validation with KFold.

    Parameters:
    - X: Features (entire dataset, not just training set)
    - y: Target (entire dataset, not just training set)
    - model: Machine learning model instance
    - param_grid: Dictionary of parameters to search
    - cv: Number of cross-validation folds (default: 5)
    - scoring: Scoring strategy for cross-validation (default: None)
    - refit_metric: Metric used to refit the best model (default: 'r2')
    - plot: Whether to plot predicted vs actual values (default: True)
    - plot_title: Custom title for the plot (default: None, uses model class name)

    Returns:
    - best_params_list: List of best parameters found by grid search for each fold
    - best_metrics_list: List of dictionaries containing best R², RMSE, MAE, MPE on the test set for each fold
    - mean_metrics_list: List of dictionaries containing mean R², RMSE, MAE, MPE across cross-validation folds for each fold
    - diff_list: List of differences between actual and predicted values for each fold
    - all_grid_search_results_df: A DataFrame containing the detailed results from the grid search process, including best metrics.
    """

    def mean_percentage_error(y_true, y_pred):
        """
        Calculate the Mean Percentage Error (MPE).

        Parameters:
        y_true (array-like): Array of actual values.
        y_pred (array-like): Array of predicted values.

        Returns:
        float: The mean percentage error.
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mpe = np.mean((y_true - y_pred) / y_true) * 100
        return mpe

    kf = KFold(n_splits=cv, shuffle=True, random_state=40)

    best_params_list = []
    best_metrics_list = []
    mean_metrics_list = []
    diff_list = []
    all_grid_search_results = []  # To store grid search results for each fold

    for fold_index, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, refit=refit_metric, return_train_score=True)
        grid_search.fit(X_train_fold, y_train_fold)
        best_params = grid_search.best_params_

        print(f"Best parameters for fold {fold_index}:", best_params)
        best_params_list.append(best_params)

        # Store grid search results
        grid_search_results = pd.DataFrame(grid_search.cv_results_)
        grid_search_results['fold'] = fold_index

        # Predict on the test set with the best model
        y_pred = grid_search.predict(X_test_fold)
        mse = mean_squared_error(y_test_fold, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_fold, y_pred)
        r2 = r2_score(y_test_fold, y_pred)
        mpe = mean_percentage_error(y_test_fold, y_pred)

        best_metrics = {
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MPE': mpe
        }
        best_metrics_list.append(best_metrics)

        # Add best metrics to the grid search results
        grid_search_results['best_test_r2'] = r2
        grid_search_results['best_test_rmse'] = rmse
        grid_search_results['best_test_mae'] = mae
        grid_search_results['best_test_mpe'] = mpe

        all_grid_search_results.append(grid_search_results)

        print(f"Best R-squared Score for fold {fold_index}:", r2)
        print(f"Best RMSE for fold {fold_index}:", rmse)
        print(f"Best MAE for fold {fold_index}:", mae)
        print(f'Best MPE for fold {fold_index}: {mpe:.2f}%')

        y_pred = pd.DataFrame(y_pred, columns=['Predicted'])
        y_test_fold = y_test_fold.reset_index(drop=True)
        diff = y_test_fold['Value'] - y_pred.squeeze()

        diff_list.append(diff)

        # Save the results for each fold
        fold_results = pd.DataFrame({
            'Actual': y_test_fold['Value'],
            'Predicted': y_pred['Predicted'],
            'Difference': diff
        })
        fold_results.to_csv(f'fold_{fold_index}_results.csv', index=False)
        print(f"Saved results for fold {fold_index} to fold_{fold_index}_results.csv")

        if scoring is None:
            scoring = {
                'r2': 'r2',
                'rmse': make_scorer(mean_squared_error, squared=False),
                'mae': 'neg_mean_absolute_error',
                'mpe': make_scorer(mean_percentage_error)
            }

        cv_results = cross_validate(grid_search.best_estimator_, X_train_fold, y_train_fold, cv=cv, scoring=scoring)

        mean_r2 = np.mean(cv_results['test_r2'])
        mean_rmse = np.mean(cv_results['test_rmse'])
        mean_mae = -np.mean(cv_results['test_mae'])  # Negate because 'neg_mean_absolute_error' returns negative values
        mean_mpe = np.mean(cv_results['test_mpe'])

        mean_metrics = {
            'R²': mean_r2,
            'RMSE': mean_rmse,
            'MAE': mean_mae,
            'MPE': mean_mpe
        }
        mean_metrics_list.append(mean_metrics)

        print(f"Mean R² across folds for fold {fold_index}:", mean_r2)
        print(f"Mean RMSE across folds for fold {fold_index}:", mean_rmse)
        print(f"Mean MAE across folds for fold {fold_index}:", mean_mae)
        print(f'Mean MPE across folds for fold {fold_index}: {mean_mpe:.2f}%')

        if plot:
            y_test_array = y_test_fold.to_numpy()
            plt.figure(figsize=(10, 6))
            sns.regplot(x=y_test_array, y=y_pred, ci=95, line_kws={"color": "black"}, scatter_kws={"s": 100})
            plt.xlabel('Actual Tg (K)', fontsize=16)
            plt.ylabel('Predicted Tg (K)', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            # Use the custom plot title if provided, otherwise use the model's class name
            title = plot_title if plot_title else model.__class__.__name__
            plt.title(f"{title} - Fold {fold_index}", fontsize=18)

            plt.text(0.05, 0.95, f'R² = {r2:.2f}', transform=plt.gca().transAxes,
                     fontsize=14, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))        

            plt.savefig(f'predicted_vs_actual_plot_fold_{fold_index}.png')
            print(f"Saved plot for fold {fold_index} to predicted_vs_actual_plot_fold_{fold_index}.png")
            plt.show()

    # Save the overall best model
    joblib.dump(grid_search.best_estimator_, 'best_model.joblib')
    print("Saved best model to best_model.joblib")

    # Combine all grid search results into a single DataFrame
    all_grid_search_results_df = pd.concat(all_grid_search_results, ignore_index=True)
    all_grid_search_results_df.to_csv('grid_search_results_all_folds.csv', index=False)
    print("Saved all grid search results to grid_search_results_all_folds.csv")

    # Save all fold results as CSV files
    pd.DataFrame(best_params_list).to_csv('best_params_all_folds.csv', index=False)
    pd.DataFrame(best_metrics_list).to_csv('best_metrics_all_folds.csv', index=False)
    pd.DataFrame(mean_metrics_list).to_csv('mean_metrics_all_folds.csv', index=False)
    pd.concat(diff_list).to_csv('diff_all_folds.csv', index=False)
    print("Saved all fold results to CSV files")

    return best_params_list, best_metrics_list, mean_metrics_list, diff_list, all_grid_search_results_df


#%%
# Example usage with SVR
from sklearn.svm import SVR

param_grid_svr = {
    'C': [300, 400, 700, 900], #400, 600, 700, 900
    'gamma': [0.005, 0.1, 1, 10], #0.005, 0.1, 1, 10
    'epsilon': [0.0001, 0.01, 0.5, 0.75],
    'kernel': ['rbf']
}


#model = SVR()
best_params_list, best_metrics_list, mean_metrics_list, diff_list, all_grid_search_results_df = evaluate_model(X, y_Tg, SVR(), param_grid_svr, cv=5, refit_metric='r2', plot_title='SVR')



#%% KRR (Kernel Ridge Regression)
from sklearn.kernel_ridge import KernelRidge

param_grid_krr = {
    'alpha': [0.1, 1, 10, 40, 50, 60, 90],#0.1, 1, 10,
    'kernel': ['polynomial', 'rbf'],
    'degree': [2, 3, 4,5,6],  # only relevant for polynomial
    'coef0': [2,3,4,5,6, 7, 8, 9]       # only relevant for polynomial and sigmoid
}


# Use the evaluate_model function
best_params_list, best_metrics_list, mean_metrics_list, diff_list, all_grid_search_results_df = evaluate_model(X_reduced, y_Tg, KernelRidge(), param_grid_krr, cv=5, refit_metric='r2', plot_title='KRR')


#%% Random Forest
from sklearn.ensemble import RandomForestRegressor

# Define the parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [200,300,400],
    'max_depth': [20,40],
    'min_samples_split': [2,3],
    'min_samples_leaf': [1,2]
}


best_params_list, best_metrics_list, mean_metrics_list, diff_list, all_grid_search_results_df = evaluate_model(X, y_Tg, RandomForestRegressor(), param_grid_rf, cv=5, refit_metric='r2', plot_title='RF')


#%%Gradient Boosting Machines
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid_gbm = {
    'n_estimators': [200,300,400],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [5,6,7],
    'subsample': [0.9],
    'min_samples_split': [8,10],
    'min_samples_leaf': [3,4,5],
    'max_features': [None],
    'loss': ['squared_error']
}



best_params_list, best_metrics_list, mean_metrics_list, diff_list, all_grid_search_results_df = evaluate_model(X_reduced, y_Tg, GradientBoostingRegressor(), param_grid_gbm, cv=5, refit_metric='r2', plot_title='GBM')


#%% xgboost
import xgboost as xgb

# Define the parameter grid
param_grid_xgb = {
    'n_estimators': [200,300,400],
    'learning_rate': [0.05,0.1,0.2],
    'max_depth': [5,6,7],
    'subsample': [0.5,0.6,0.9],
    'colsample_bytree': [0.9],
    'gamma': [0.05,0.1,0.2],
    'reg_alpha': [0.5],
    'reg_lambda': [1.5]
}


best_params_list, best_metrics_list, mean_metrics_list, diff_list, all_grid_search_results_df = evaluate_model(X, y_Tg, xgb.XGBRegressor(), param_grid_xgb, cv=5, refit_metric='r2', plot_title='XGBoost')

#%% CatBoost
from catboost import CatBoostRegressor

# Define the parameter grid
param_grid_catboost = {
    'iterations': [100,200,300],  # Equivalent to n_estimators in other models
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4,5,6,7],  # Equivalent to max_depth
    'subsample': [0.7,0.8,0.9],
}


best_params_list, best_metrics_list, mean_metrics_list, diff_list, all_grid_search_results_df = evaluate_model(X, y_Tg, CatBoostRegressor(silent=True), param_grid_catboost, cv=5, refit_metric='r2', plot_title='CatBoost')

#%% kNN regressor
from sklearn.neighbors import KNeighborsRegressor

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}


# Create an instance of KNeighborsRegressor
best_params_list, best_metrics_list, mean_metrics_list, diff_list, all_grid_search_results_df = evaluate_model(X, y_Tg, KNeighborsRegressor(),param_grid_knn, cv=5, refit_metric='r2', plot_title='k-NN')



