import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../libs')))

import machineLearningLib

class TestMachineLearningLib(unittest.TestCase):

    def test_preprocess_data(self):
        data = pd.DataFrame({
            'Latitude': [0, 1],
            'Longitude': [0, 1],
            'TotalTripTime': [100, 200],
            'SOG': [10, 20],
            'COG': [30, 40],
            'DirectionChange': [5, 10],
            'Mean_SOG': [15, 25],
            'DistanceFromStart': [50, 100],
            'DistanceFromEnd': [150, 200],
            'TimeFromStart': [1000, 2000]
        })
        target = 'TotalTripTime'
        features = ['Latitude', 'Longitude', 'SOG', 'COG', 'DirectionChange', 'Mean_SOG', 'DistanceFromStart', 'DistanceFromEnd', 'TimeFromStart']
        X, y = machineLearningLib.preprocess_data(data, target, features)
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)

    @patch('machineLearningLib.mean_absolute_error')
    @patch('machineLearningLib.mean_squared_error')
    def test_evaluate_model(self, mock_mse, mock_mae):
        from sklearn.linear_model import LinearRegression

        mock_mae.return_value = 1
        mock_mse.return_value = 1

        model = LinearRegression()
        X_train = pd.DataFrame({'Latitude': [0, 1], 'Longitude': [0, 1]})
        y_train = pd.Series([2, 3])
        model.fit(X_train, y_train)
        X_test = pd.DataFrame({'Latitude': [0, 1], 'Longitude': [0, 1]})
        y_test = pd.Series([2, 3])

        mae, rmse, y_pred = machineLearningLib.evaluate_model(model, X_test, y_test, 'LinearRegression', 'Test')
        self.assertEqual(mae, 1)
        self.assertEqual(rmse, 1)

    @patch('machineLearningLib.pd.DataFrame.to_csv')
    def test_save_results(self, mock_to_csv):
        results = [{'Dataset': 'Test', 'Model': 'LinearRegression', 'Mean Absolute Error': 1, 'Root Mean Squared Error': 1}]
        predictions = pd.DataFrame({'Actual ETA': [2, 3], 'Predicted ETA': [2, 3]})
        save_path = './daten/results'
        dataset = 'Test'
        machineLearningLib.save_results(results, predictions, save_path, dataset)
        mock_to_csv.assert_called()

    @patch('machineLearningLib.pd.DataFrame.to_csv')
    def test_train_models(self, mock_to_csv):
        data_frame = pd.DataFrame({
            'Latitude': [0, 1],
            'Longitude': [0, 1],
            'TotalTripTime': [100, 200],
            'SOG': [10, 20],
            'COG': [30, 40],
            'DirectionChange': [5, 10],
            'Mean_SOG': [15, 25],
            'DistanceFromStart': [50, 100],
            'DistanceFromEnd': [150, 200],
            'TimeFromStart': [1000, 2000]
        })
        dataset = 'Test'
        results, best_model, best_model_instance, _, _, _ = machineLearningLib.train_models(data_frame, dataset=dataset, clean_path=False, test_size=0.2, random_state=42, skip_hyperparameter=True, total_time=True)
        self.assertIsInstance(results, list)
        self.assertIsInstance(best_model, dict)
        self.assertIsNotNone(best_model_instance)

    @patch('machineLearningLib.pd.DataFrame.to_csv')
    def test_train_models_with_hyperparameter(self, mock_to_csv):
        data_frame = pd.DataFrame({
            'Latitude': [0, 1],
            'Longitude': [0, 1],
            'TotalTripTime': [100, 200],
            'SOG': [10, 20],
            'COG': [30, 40],
            'DirectionChange': [5, 10],
            'Mean_SOG': [15, 25],
            'DistanceFromStart': [50, 100],
            'DistanceFromEnd': [150, 200],
            'TimeFromStart': [1000, 2000]
        })
        dataset = 'Test'
        results, best_model, best_model_instance, _, _, _ = machineLearningLib.train_models(data_frame, dataset=dataset, clean_path=False, test_size=0.2, random_state=42, skip_hyperparameter=False, total_time=True)
        self.assertIsInstance(results, list)
        self.assertIsInstance(best_model, dict)

    def test_time_it_decorator(self):
        @machineLearningLib.time_it
        def sample_function(x):
            return x * x

        result = sample_function(5)
        self.assertEqual(result, 25)

    def test_calculate_distances(self):
        latitude = 0
        longitude = 0
        start_coords = (0, 0)
        end_coords = (1, 1)
        with patch('machineLearningLib.geopy_haversine', return_value=1):
            distance_from_start, distance_from_end = machineLearningLib.calculate_distances(latitude, longitude, start_coords, end_coords)
            self.assertEqual(distance_from_start, 1)
            self.assertEqual(distance_from_end, 1)

    def test_calculate_initial_compass_bearing(self):
        lat1 = 0
        lon1 = 0
        lat2 = 1
        lon2 = 1
        bearing = machineLearningLib.calculate_initial_compass_bearing(lat1, lon1, lat2, lon2)
        self.assertIsInstance(bearing, float)

    def test_calculate_new_position(self):
        lat1 = 0
        lon1 = 0
        bearing = 45
        distance = 100
        new_lat, new_lon = machineLearningLib.calculate_new_position(lat1, lon1, bearing, distance)
        self.assertIsInstance(new_lat, float)
        self.assertIsInstance(new_lon, float)

if __name__ == '__main__':
    unittest.main()
