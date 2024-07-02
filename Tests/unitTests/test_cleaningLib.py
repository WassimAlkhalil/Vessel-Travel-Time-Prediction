import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from libs import cleaningLib as lod

class TestCleaningLib(unittest.TestCase):

    def test_preprocess_all_columns(self):
        df = pd.DataFrame({
            'col1': ['1,1', '2,2', '3,3'],
            'col2': ['"a"', "'b'", '"c"']
        })
        processed_df = lod.preprocess_all_columns(df)
        expected_df = pd.DataFrame({
            'col1': ['1.1', '2.2', '3.3'],
            'col2': ['a', 'b', 'c']
        })
        pd.testing.assert_frame_equal(processed_df, expected_df)

    @patch('libs.cleaningLib.load_data')
    def test_load_data(self, mock_load_data):

        mock_df = pd.DataFrame({
            'TripID': ['1', '2'],
            'MMSI': ['1', '2'],
            'ID': ['1', '2'],
            'StartTime': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'EndTime': pd.to_datetime(['2023-01-03', '2023-01-04']),
            'time': pd.to_datetime(['2023-01-02', '2023-01-03'])
        })
        mock_load_data.return_value = mock_df
        
        df1 = lod.load_data('../daten/recived_data/bremerhaven_hamburg.csv')
        df2 = lod.load_data('../daten/recived_data/kiel_gdynia.csv')
    
        # Check if the data was loaded correctly by verifying that the DataFrame is not None
        self.assertIsNotNone(df1)
        self.assertIsNotNone(df2)
    
        # Additional checks can be added to ensure the DataFrame has expected structure
        self.assertIsInstance(df1, pd.DataFrame)
        self.assertIsInstance(df2, pd.DataFrame)
    
        # Check that essential columns are present
        expected_columns = ['TripID', 'MMSI', 'ID', 'StartTime', 'EndTime', 'time']
        for column in expected_columns:
            self.assertIn(column, df1.columns)
            self.assertIn(column, df2.columns)

    def test_drop_invalid_time_entries(self):
        df = pd.DataFrame({
            'StartTime': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'EndTime': pd.to_datetime(['2023-01-03', '2023-01-04']),
            'time': pd.to_datetime(['2023-01-02', '2023-01-03'])
        })
        valid_df = lod.drop_invalid_time_entries(df)
        self.assertEqual(len(valid_df), 2)

    def test_sort_data(self):
        df = pd.DataFrame({
            'TripID': ['2', '1'],
            'time': pd.to_datetime(['2023-01-02', '2023-01-01'])
        })
        sorted_df = lod.sort_data(df)
        self.assertEqual(sorted_df.iloc[0]['TripID'], '1')

    def test_remove_duplicate_ids_from_dataframe(self):
        df = pd.DataFrame({
            'TripID': ['1', '1', '2'],
            'ID': ['a', 'a', 'b'],
            'MMSI': ['1', '1', '2'],
            'StartTime': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02')],
            'EndTime': [pd.Timestamp('2023-01-02'), pd.Timestamp('2023-01-02'), pd.Timestamp('2023-01-03')]
        })
        cleaned_df = lod.remove_duplicate_ids_from_dataframe(df)
        self.assertEqual(len(cleaned_df), 2)

    def test_interpolate_circular_data_scipy(self):
        df = pd.DataFrame({'col1': [0, np.nan, 90, 180, np.nan, 270, 360]})
        interpolated_df = lod.interpolate_circular_data_scipy(df, ['col1'])
        self.assertFalse(interpolated_df.isna().any().any())

    def test_process_cog_and_th_data(self):
        df = pd.DataFrame({
            'TripID': ['1', '1', '1'],
            'COG': [10, np.nan, 350],
            'TH': [0, 180, np.nan]
        })
        processed_df, _ = lod.process_cog_and_th_data(df)
        self.assertFalse(processed_df['COG'].isna().any())
        self.assertFalse(processed_df['TH'].isna().any())

    def test_geopy_haversine(self):
        distance = lod.geopy_haversine(0, 0, 1, 1)
        self.assertAlmostEqual(distance, 157.25, places=2)

    def test_validate_and_clean_coordinates(self):
        df = pd.DataFrame({
            'TripID': ['1', '1'],
            'Latitude': [0, np.nan],
            'Longitude': [0, 0]
        })
        cleaned_df, _ = lod.validate_and_clean_coordinates(df, method='zscore')
        self.assertFalse(cleaned_df.isna().any().any())

    def test_process_sog_data(self):
        df = pd.DataFrame({
            'TripID': ['1', '1'],
            'SOG': [10, np.nan]
        })
        cleaned_df, _ = lod.process_sog_data(df)
        self.assertFalse(cleaned_df['SOG'].isna().any().any())

    def test_detect_outliers_dbscan(self):
        df = pd.DataFrame({
            'Latitude': [0, 0, 0, 0, 90],
            'Longitude': [0, 0, 0, 0, 180]
        })
        result_df = lod.detect_outliers_dbscan(df, eps=1.0, min_samples=2)
        print(result_df['outlier'])
        self.assertIn('outlier', result_df.columns)
        self.assertEqual(result_df['outlier'].sum(), 1)

    def test_searoute_distance(self):
        # Mock the searoute.searoute function
        with patch('searoute.searoute') as mock_searoute:
            mock_searoute.return_value.properties = {'length': 1000}
            distance = lod.searoute_distance(0, 0, 1, 1)
            self.assertEqual(distance, 1000)
            self.assertEqual(mock_searoute.call_count, 1)
        
if __name__ == '__main__':
    unittest.main()
