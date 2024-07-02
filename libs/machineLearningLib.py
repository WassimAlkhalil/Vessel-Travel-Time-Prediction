import pandas as pd
import numpy as np
from cleaningLib import geopy_haversine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils import resample
import joblib
import os
import shutil
import time
import math

# Timing decorator
def time_it(func):
    """
    Decorator to measure the execution time of a function.

    Args:
        func (function): The function to be timed.

    Returns:
        function: The wrapped function with timing.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            result = None
        end_time = time.time()
        print(f"Time taken for {func.__name__}: {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# Data preprocessing function
def preprocess_data(data_frame, target, features):
    """
    Preprocess the data for model training.

    Args:
        data_frame (pd.DataFrame): The input data frame.
        target (str): The target column name.
        features (list of str): List of feature column names.

    Returns:
        tuple: A tuple containing features (X) and target (Y) arrays.
    """
    try:
        X = data_frame[features].values
        Y = data_frame[target].values
        return X, Y
    except KeyError as e:
        print(f"Error in preprocess_data: {e}")
        return None, None

# Model training with GridSearchCV for multiple models
@time_it
def train_model(model, X_train, y_train, param_grid=None, skip_hyperparameter=True): # skip_hyperparameter=True, because It can be very time-consuming for large hyperparameter spaces
    """
    Train a machine learning model with optional hyperparameter tuning.

    Args:
        model: The machine learning model to train.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        param_grid (dict, optional): Hyperparameter grid for GridSearchCV.
        skip_hyperparameter (bool, optional): Whether to skip hyperparameter tuning.

    Returns:
        model: The trained model.
    """
    try:
        if (not skip_hyperparameter) and param_grid:  # Check if param_grid is not empty
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=2)
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_
        else:
            model.fit(X_train, y_train)
            return model
    except Exception as e:
        print(f"Error in train_model: {e}")
        return None

# Evaluate model function
def evaluate_model(model, X_test, y_test, model_name, dataset):
    """
    Evaluate a trained model on test data.

    Args:
        model: The trained machine learning model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test targets.
        model_name (str): The name of the model.
        dataset (str): The name of the dataset.

    Returns:
        tuple: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and predictions.
    """
    try:
        print(f"Evaluating {model_name} on {dataset} data...")
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return mae, rmse, y_pred
    except Exception as e:
        print(f"Error in evaluate_model: {e}")
        return None, None, None

# Save results function
def save_results(results, predictions, save_path, dataset):
    """
    Save the results and predictions to CSV files.

    Args:
        results (list of dict): Evaluation results.
        predictions (pd.DataFrame): Predictions data frame.
        save_path (str): Directory path to save the results.
        dataset (str): The name of the dataset.
    """
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(save_path, f'model_evaluation_results_{dataset.lower().replace(" ", "_")}.csv'), index=False)
        predictions.to_csv(os.path.join(save_path, f'predictions_{dataset.lower().replace(" ", "_")}.csv'), index=False)
    except Exception as e:
        print(f"Error in save_results: {e}")

# Clear results folder function
def clear_results_folder(results_dir, clean_path):
    """
    Clear the results folder by deleting and recreating it.

    Args:
        results_dir (str): The directory path of the results folder.
    """
    try:
        if results_dir is not None:
            if os.path.exists(results_dir):
                if clean_path:
                    shutil.rmtree(results_dir)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
    except Exception as e:
        print(f"Error in clear_results_folder: {e}")

# Central training function
def train_models(data_frame, dataset=None, clean_path=False, models_names=['ExtraTrees', 'RandomForest'], test_size=0.2, random_state=42, skip_hyperparameter=True, total_time=False, save_model=False, save_path = './data/results'):
    """
    Train and evaluate specified machine learning models on the provided dataset.

    Args:
        data_frame (pd.DataFrame): The input data frame containing the dataset.
        dataset (str, optional): The name of the dataset. Defaults to 'Single Dataset'.
        clean_path (bool, optional): Whether to clear the results folder before saving new results. Defaults to False.
        models_names (list, optional): List of model names to be trained. Defaults to ['ExtraTrees'].
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): The random seed for train-test split. Defaults to 42.
        skip_hyperparameter (bool, optional): Whether to skip hyperparameter tuning. Defaults to True.
        total_time (bool, optional): If True, target is 'TotalTripTime', otherwise 'TimeUntilEnd'. Defaults to False.
        save_model (bool, optional): Whether to save the trained models. Defaults to False.
        save_path (str, optional): Directory path to save the results. Defaults to './data/results

    Returns:
        tuple: A tuple containing (results, best_model, best_model_instance, stacking_model, stacking_metrics, stacking_path):
            - results (list of dict): A list of dictionaries with evaluation metrics for each model.
            - best_model (dict): A dictionary with the name and MAE of the best model.
            - best_model_instance (object): The instance of the best trained model.
            - stacking_model (str): The instance of the trained stacking model.
            - stacking_metrics (dict): The evaluation metrics (MAE, RMSE) of the stacking model.
            - stacking_path (str): The path to the saved stacking model.
    """
    try:
        dataset = dataset if dataset is not None else 'Single Dataset'
        target = 'TotalTripTime' if total_time else 'TimeUntilEnd'
        features = ['Latitude', 'Longitude', 'SOG', 'COG', 'DirectionChange', 'Mean_SOG', 'DistanceFromStart', 'DistanceFromEnd', 'TimeFromStart']

        # Define default models
        models = {
            'LinearRegression': LinearRegression(),
            'ExtraTrees': ExtraTreesRegressor(),
            'AdaBoost': AdaBoostRegressor(),
            'GradientBoosting': GradientBoostingRegressor(),
            'RandomForest': RandomForestRegressor()
        }

        # Clear the results folder before saving new results
        clear_results_folder(save_path, clean_path)
        
        # Preprocess the data
        print("Preprocess the data")
        X, y = preprocess_data(data_frame, target, features)
        if X is None or y is None or models_names is None:
            return None, None, None

        # Train-test split
        print("Train-test split")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        param_distributions = {
            'LinearRegression': {},
            'ExtraTrees': {'n_estimators': range(10, 101, 10), 'random_state': range(6, 19), 'max_features': [1.0]},
            'AdaBoost': {'n_estimators': range(10, 101, 10), 'learning_rate': np.linspace(0.0001, 1.0001, 10)},
            'GradientBoosting': {'n_estimators': range(10, 101, 10), 'learning_rate': np.linspace(0.01, 1, 10)},
            'RandomForest': {'n_estimators': range(10, 101, 10), 'max_depth': range(1, 21)}
        }
        
        results = []
        best_model = {'name': None, 'mae': float('inf')}
        best_model_instance = None
        trained_models = []

        for model_name, model in models.items():
            if model_name not in models_names:
                continue
            param_dist = param_distributions.get(model_name, {})
            # Train the model with hyperparameter tuning
            print(f"Train the model with hyperparameter tuning: {model_name}")
            trained_model_instance = train_model(model, X_train, y_train, param_grid=param_dist, skip_hyperparameter=skip_hyperparameter)

            if trained_model_instance is None:
                continue

            # Evaluate the trained model
            print(f" Evaluate the trained model: {model_name}")
            mae, rmse, y_pred = evaluate_model(trained_model_instance, X_test, y_test, model_name, dataset)
            if mae is None or rmse is None or y_pred is None:
                continue

            print(f"Results for {model_name} on {dataset} data:")
            print(f"Mean Absolute Error: {mae}")
            print(f"Root Mean Squared Error: {rmse}")
            results.append({
                'Dataset': dataset,
                'Model': model_name,
                'Mean Absolute Error': mae,
                'Root Mean Squared Error': rmse
            })
            trained_models.append((model_name, trained_model_instance))

            # Save predictions
            predictions = pd.DataFrame({
                'Actual ETA': y_test,
                'Predicted ETA': y_pred
            })
            predictions.to_csv(os.path.join(save_path, f'predictions_{dataset.lower().replace(" ", "_")}_{model_name}.csv'), index=False)

            # Save the trained model if save_model is True
            if save_model:
                joblib.dump(trained_model_instance, os.path.join(save_path, f'{model_name}_model_{dataset.lower().replace(" ", "_")}.joblib'))

            # Update the best model if current model has lower MAE
            if mae < best_model['mae'] or best_model['name'] == None:
                best_model['name'] = model_name
                best_model['mae'] = mae
                best_model_instance = trained_model_instance

        # Ensure there are at least two models for stacking
        if len(trained_models) < 2:
            print("Not enough models to perform stacking. At least two models are required.")
            return results, best_model, best_model_instance, None, None, None

        # Use a smaller subset of data for stacking  # ------------------------------------------------------------------------------------------------------------>><< Fo avoid the Long Time, we can use small Set for training
        # X_train_small, y_train_small = resample(X_train, y_train, n_samples=1000, random_state=random_state)
        # print(f"Training stacking model with a subset of {len(X_train_small)} samples")

        # Train the stacking model
        print("Train the stacking model with the best models")
        stacking_path = os.path.join(save_path, f'stacking_model_{dataset.lower().replace(" ", "_")}.joblib')
        stacking_model, stacking_metrics = train_stacking_model(trained_models, X_train, y_train, X_test, y_test, final_estimator=LinearRegression(), save_path=stacking_path)

        # Save overall results
        save_results(results, pd.DataFrame(results), save_path, dataset)

        # Print the best model
        print(f"Best model: {best_model['name']} with Mean Absolute Error: {best_model['mae']}")

        return results, best_model, best_model_instance, stacking_model, stacking_metrics, stacking_path

    except Exception as e:
        print(f"Error in train_models: {e}")
        return None, None, None, None, None

# Stacking model training function
@time_it
def train_stacking_model(models, X_train, y_train, X_test, y_test, final_estimator=ExtraTreesRegressor(n_estimators=range(10, 101, 10), random_state=range(6, 19), max_features=[1.0]), save_path='./data/results/stacking_model.joblib'):
    """
    Train a stacking regressor with multiple models and save the trained model.

    Args:
        models (list of tuples): List of tuples where each tuple contains the name and instance of the model.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test targets.
        final_estimator: The final estimator to use in the stacking regressor.
        save_path (str): Path to save the trained stacking model.

    Returns:
        stacking_model: The trained stacking regressor.
        metrics: The evaluation metrics (MAE, RMSE) on the test set.
    """
    try:
        stacking_model = StackingRegressor(estimators=models, final_estimator=final_estimator, n_jobs=2) # -------------------------------------------------------->><< final_estimator
        stacking_model.fit(X_train, y_train)

        # Save the stacking model
        joblib.dump(stacking_model, save_path)

        # Evaluate the stacking model
        y_pred = stacking_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"Stacking Model Evaluation:")
        print(f"Mean Absolute Error: {mae}")
        print(f"Root Mean Squared Error: {rmse}")

        metrics = {'mae': mae, 'rmse': rmse}

        return stacking_model, metrics
    except Exception as e:
        print(f"Error in train_stacking_model: {e}")
        return None, None


def calculate_distances(latitude, longitude, start_coords, end_coords):
    """
    Calculate the distances from the current location to the start and end coordinates.

    Args:
        latitude (float): Current latitude.
        longitude (float): Current longitude.
        start_coords (tuple): Starting coordinates (latitude, longitude).
        end_coords (tuple): Ending coordinates (latitude, longitude).

    Returns:
        tuple: Distance from start and distance from end.
    """
    try:
        distance_from_start = geopy_haversine(start_coords[0], start_coords[1], latitude, longitude)
        distance_from_end = geopy_haversine(latitude, longitude, end_coords[0], end_coords[1])
        return distance_from_start, distance_from_end
    except Exception as e:
        print(f"Error in calculate_distances: {e}")
        return None, None

def calculate_initial_compass_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the initial compass bearing between two points.

    Args:
        lat1 (float): Latitude of the start point.
        lon1 (float): Longitude of the start point.
        lat2 (float): Latitude of the end point.
        lon2 (float): Longitude of the end point.

    Returns:
        float: The initial compass bearing in degrees.
    """
    try:
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        diffLong = lon2 - lon1
        x = math.sin(diffLong) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))
        initial_bearing = math.atan2(x, y)
        initial_bearing = math.degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360
        return compass_bearing
    except Exception as e:
        print(f"Error in calculate_initial_compass_bearing: {e}")
        return None

def calculate_new_position(lat1, lon1, bearing, distance):
    """
    Calculate the new position given a start point, bearing, and distance.

    Args:
        lat1 (float): Latitude of the start point.
        lon1 (float): Longitude of the start point.
        bearing (float): Bearing in degrees.
        distance (float): Distance to travel in kilometers.

    Returns:
        tuple: Latitude and longitude of the new position.
    """
    try:
        R = 6371  # Radius of the Earth in kilometers
        lat1, lon1, bearing = map(math.radians, [lat1, lon1, bearing])
        lat2 = math.asin(math.sin(lat1) * math.cos(distance/R) + math.cos(lat1) * math.sin(distance/R) * math.cos(bearing))
        lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(distance/R) * math.cos(lat1), math.cos(distance/R) - math.sin(lat1) * math.sin(lat2))
        lat2, lon2 = map(math.degrees, [lat2, lon2])
        return lat2, lon2
    except Exception as e:
        print(f"Error in calculate_new_position: {e}")
        return None, None

def calculate_cog_and_direction_change(current_lat, current_lon, start_lat, start_lon, end_lat, end_lon, distance_traveled):
    """
    Calculate Course Over Ground (COG) and direction change based on coordinates.

    Args:
        current_lat (float): Current latitude.
        current_lon (float): Current longitude.
        start_lat (float): Starting latitude.
        start_lon (float): Starting longitude.
        end_lat (float): Ending latitude.
        end_lon (float): Ending longitude.
        distance_traveled (float): Distance traveled from the start point.

    Returns:
        tuple: COG and direction change.
    """
    try:
        # Calculate initial bearing from start to end coordinates
        initial_bearing = calculate_initial_compass_bearing(start_lat, start_lon, end_lat, end_lon) 
        if initial_bearing is None:
            return None, None
        # Calculate intermediate position after the given time
        new_lat, new_lon = calculate_new_position(start_lat, start_lon, initial_bearing, distance_traveled)
        if new_lat is None or new_lon is None:
            return None, None
        # Calculate COG (bearing from current position to new position)
        cog = calculate_initial_compass_bearing(current_lat, current_lon, new_lat, new_lon)
        if cog is None:
            return None, None
        # Calculate the change in direction
        direction_change = (cog - initial_bearing + 360) % 360
        return cog, direction_change
    except Exception as e:
        print(f"Error in calculate_cog_and_direction_change: {e}")
        return None, None
    
# Central prediction function
def predict_eta(latitude, longitude, time, start_lat, start_lon, start_time, end_lat, end_lon,
                sog=None, cog=None, direction_change=None, mean_sog=None, time_from_start=None, model_path=None, model_instance=None):
    """
    Predict the Estimated Time of Arrival (ETA) using a trained model.

    Args:
        latitude (float): Current latitude.
        longitude (float): Current longitude.
        time (datetime): Current time.
        start_lat (float): Starting latitude.
        start_lon (float): Starting longitude.
        start_time (datetime): Starting time.
        end_lat (float): Ending latitude.
        end_lon (float): Ending longitude.
        sog (float, optional): Speed Over Ground.
        cog (float, optional): Course Over Ground.
        direction_change (float, optional): Change in direction.
        mean_sog (float, optional): Mean Speed Over Ground.
        time_from_start (float, optional): Time from start in seconds.
        model_path (str, optional): Path to the trained model.

    Returns:
        float: Predicted ETA.
    """
    try:
        # Load the model
        if model_path is not None:
            if os.path.exists(model_path):
                model = joblib.load(model_path)
        elif model_instance is not None:
            model = model_instance
        else:
            raise ValueError("Either model_path or model_instance must be provided")
        
        if not model:
            raise ValueError("Provided model not found")
        
        # Default values
        distance_from_start, distance_from_end = calculate_distances(start_lat, start_lon, (latitude, longitude), (end_lat, end_lon))
        if distance_from_start is None or distance_from_end is None:
            return None
        compute_T_from_start = (time - start_time).total_seconds()
        compute_sog = round((distance_from_start/(compute_T_from_start/3600)),2) if compute_T_from_start > 0 else 0
        compute_cog, compute_direction_change = calculate_cog_and_direction_change(latitude, longitude, start_lat, start_lon, end_lat, end_lon, distance_from_start)
        if compute_cog is None or compute_direction_change is None:
            return None
        
        sog = sog if sog is not None else compute_sog
        cog = cog if cog is not None else compute_cog
        direction_change = direction_change if direction_change is not None else compute_direction_change
        mean_sog = mean_sog if mean_sog is not None else 14.0
        time_from_start = time_from_start if time_from_start is not None else compute_T_from_start

        input_data = pd.DataFrame({
            'Latitude': [latitude],
            'Longitude': [longitude],
            'SOG': [sog],
            'COG': [cog],
            'DirectionChange': [direction_change],
            'Mean_SOG': [mean_sog],
            'DistanceFromStart': [distance_from_start],
            'DistanceFromEnd': [distance_from_end],
            'TimeFromStart': [time_from_start]
        })
        
        # Convert DataFrame to NumPy array
        input_array = input_data.to_numpy()
        
        eta = model.predict(input_array)[0]

        return eta
    except Exception as e:
        print(f"Error in predict_eta: {e}")
        return None

