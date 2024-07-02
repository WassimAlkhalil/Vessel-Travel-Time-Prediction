import unittest
import sys
import os
from pathlib import Path

# Add the root of the project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from libs import visualizationLib as vis
import pandas as pd
from dash import Input, Output

class TestVisualizationLib(unittest.TestCase):

    def setUp(self):
        # Setup a sample cleaned_data DataFrame
        self.cleaned_data = pd.DataFrame({
            'TripID': [1, 1, 2, 2],
            'MMSI': [123456, 123456, 789012, 789012],
            'Name': ['ShipA', 'ShipA', 'ShipB', 'ShipB'],
            'StartPort': ['Port1', 'Port1', 'Port2', 'Port2'],
            'EndPort': ['Port3', 'Port3', 'Port4', 'Port4'],
            'StartTime': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
            'EndTime': ['2023-01-03', '2023-01-03', '2023-01-04', '2023-01-04'],
            'Latitude': [53.0, 53.1, 54.0, 54.1],
            'Longitude': [8.0, 8.1, 9.0, 9.1],
            'COG': [10, 20, 30, 40],
            'SOG': [5, 6, 7, 8],
            'time': ['2023-01-01 10:00:00', '2023-01-01 11:00:00', '2023-01-02 10:00:00', '2023-01-02 11:00:00'],
            'ID': [1, 2, 3, 4]
        })

    def test_create_app(self):
        app, server = vis.create_app(self.cleaned_data)
        self.assertIsNotNone(app)
        self.assertIsNotNone(server)

    def test_run_app(self):
        app, server = vis.create_app(self.cleaned_data)
        # Check that the app title is the default "Dash"
        self.assertEqual(app.title, "Dash")

        # Check that the server is running
        self.assertIsNotNone(server)

    def test_layout_structure(self):
        app, server = vis.create_app(self.cleaned_data)
        layout = app.layout

        # Check for layout components
        self.assertIsNotNone(layout.children)
        # Check for the correct number of layout components
        self.assertEqual(len(layout.children), 8)  # Update this to match the actual number of children

    def test_callbacks(self):
        app, server = vis.create_app(self.cleaned_data)

        # Check that callback functions are registered
        callbacks = list(app.callback_map.keys())
        expected_callbacks = [
            '..trip-details.children...trip-paths.figure...direction-changes.figure...direction-over-time.figure..',
            'shutdown-flag.data'
        ]
        for callback in expected_callbacks:
            self.assertIn(callback, callbacks)

    def test_data_integrity(self):
        app, server = vis.create_app(self.cleaned_data)
        layout = app.layout

        # Check if data passed into the app remains unchanged
        pd.testing.assert_frame_equal(self.cleaned_data, pd.DataFrame({
            'TripID': [1, 1, 2, 2],
            'MMSI': [123456, 123456, 789012, 789012],
            'Name': ['ShipA', 'ShipA', 'ShipB', 'ShipB'],
            'StartPort': ['Port1', 'Port1', 'Port2', 'Port2'],
            'EndPort': ['Port3', 'Port3', 'Port4', 'Port4'],
            'StartTime': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
            'EndTime': ['2023-01-03', '2023-01-03', '2023-01-04', '2023-01-04'],
            'Latitude': [53.0, 53.1, 54.0, 54.1],
            'Longitude': [8.0, 8.1, 9.0, 9.1],
            'COG': [10, 20, 30, 40],
            'SOG': [5, 6, 7, 8],
            'time': ['2023-01-01 10:00:00', '2023-01-01 11:00:00', '2023-01-02 10:00:00', '2023-01-02 11:00:00'],
            'ID': [1, 2, 3, 4]
        }))

if __name__ == '__main__':
    unittest.main()
