# visualization.py
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import webbrowser
import threading
import requests
from flask import Flask, request

def create_app(cleaned_data):
    """
    Create the Dash app and layout.
    
    Args:
        cleaned_data (pd.DataFrame): DataFrame containing the cleaned maritime data.

    Returns:
        app (Dash): Dash app instance.
        server (Flask): Flask server instance used by the Dash app.
    """
    # Initialize the Flask server
    server = Flask(__name__)

    # Initialize the Dash app
    app = dash.Dash(__name__, server=server, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

    # Setup layout of the app
    app.layout = html.Div([
        html.H1("Maritime Data Visualization"),
        dcc.Dropdown(
            id='trip-dropdown',
            options=[{'label': f'Trip {trip_id}', 'value': trip_id} for trip_id in cleaned_data['TripID'].unique()],
            value=cleaned_data['TripID'].unique()[0]
        ),
        html.Div(id='trip-details', style={'margin-bottom': '20px'}),
        dcc.Graph(id='trip-paths'),
        dcc.Graph(id='direction-changes'),
        dcc.Graph(id='direction-over-time'),
        dcc.Interval(id='interval-component', interval=1*1000, n_intervals=0),
        dcc.Store(id='shutdown-flag', data=False)  # Store to keep track of shutdown flag
    ])

    @app.callback(
        [Output('trip-details', 'children'),
         Output('trip-paths', 'figure'),
         Output('direction-changes', 'figure'),
         Output('direction-over-time', 'figure')],
        [Input('trip-dropdown', 'value')]
    )
    def update_graphs(trip_id):
        """
        Update the trip details and graphs based on the selected trip ID.

        Args:
            trip_id (str): The selected TripID from the dropdown.

        Returns:
            trip_details (html.Div): HTML div containing trip details.
            fig_paths (plotly.graph_objs.Figure): Plotly figure showing trip paths.
            fig_direction (plotly.graph_objs.Figure): Plotly figure showing direction changes.
            fig_time (plotly.graph_objs.Figure): Plotly figure showing direction changes over time.
        """
        try:
            trip_data = cleaned_data[cleaned_data['TripID'] == trip_id]
            
            # Trip Details
            trip_details = html.Div([
                html.H4("Trip Details"),
                html.P(f"TripID: {trip_id}"),
                html.P(f"Ship Number: {trip_data['MMSI'].iloc[0]}"),
                html.P(f"Ship Name: {trip_data['Name'].iloc[0]}"),
                html.P(f"Start Port: {trip_data['StartPort'].iloc[0]}"),
                html.P(f"End Port: {trip_data['EndPort'].iloc[0]}"),
                html.P(f"Start Time: {trip_data['StartTime'].iloc[0]}"),
                html.P(f"End Time: {trip_data['EndTime'].iloc[0]}")
            ])
            
            # Trip Paths with additional details
            fig_paths = px.scatter_mapbox(
                trip_data, 
                lat="Latitude", 
                lon="Longitude", 
                color="TripID", 
                hover_data={
                    "SOG": True,
                    "time": True,
                    "Latitude": True,
                    "Longitude": True,
                    "ID": True
                },
                title="Ship Trip Paths",
                zoom=4,
                height=600
            )
            fig_paths.update_layout(mapbox_style="open-street-map")
            
            # Direction Changes (Circular Plot)
            bins = np.linspace(0, 360, 37)  # 36 bins (10 degrees each)
            hist, edges = np.histogram(trip_data['COG'], bins=bins)
            bin_centers = 0.5 * (edges[:-1] + edges[1:])
            
            fig_direction = go.Figure()
            fig_direction.add_trace(go.Barpolar(
                r=hist,
                theta=bin_centers,
                width=10,
                marker_color=hist,
                marker_colorscale='Viridis'
            ))
            fig_direction.update_layout(
                title="Direction Changes During Trip",
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(hist) + 1]
                    ),
                    angularaxis=dict(
                        tickmode='array',
                        tickvals=[0, 90, 180, 270],
                        ticktext=['N', 'E', 'S', 'W']
                    )
                )
            )
            
            # Direction Over Time
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=trip_data['time'],
                y=trip_data['COG'],
                mode='lines+markers',
                name='COG'
            ))
            fig_time.add_trace(go.Scatter(
                x=trip_data['time'],
                y=trip_data['TH'],
                mode='lines+markers',
                name='TH'
            ))
            fig_time.update_layout(
                title='Direction Changes Over Time',
                xaxis_title='Time',
                yaxis_title='Degrees',
                yaxis=dict(range=[0, 360])
            )
            
            return trip_details, fig_paths, fig_direction, fig_time
        except Exception as e:
            print(f"Error updating graphs: {e}")
            return html.Div(["Error loading trip details."]), go.Figure(), go.Figure(), go.Figure()

    @app.callback(
        Output('shutdown-flag', 'data'),
        [Input('interval-component', 'n_intervals')]
    )
    def check_shutdown(n_intervals):
        """
        Check for shutdown condition periodically.

        Args:
            n_intervals (int): Number of intervals passed.

        Returns:
            bool: True if shutdown condition is met, otherwise dash.no_update.
        """
        if not request.environ.get('werkzeug.server.shutdown'):
            return dash.no_update
        if request.remote_addr == '127.0.0.1' and 'shutdown' in request.cookies:
            return True
        return dash.no_update

    @app.server.route('/shutdown', methods=['POST'])
    def shutdown():
        """
        Shutdown the server.

        Returns:
            str: Server shutdown message.
        """
        shutdown_func = request.environ.get('werkzeug.server.shutdown')
        if shutdown_func:
            shutdown_func()
        return 'Server shutting down...'

    return app, server

def open_browser():
    """
    Open the web browser to access the Dash app.
    """
    webbrowser.open_new("http://127.0.0.1:8050/")

def run_app(app, server):
    """
    Run the Dash app with browser opening and OS-independent shutdown.

    Args:
        app (Dash): Dash app instance.
        server (Flask): Flask server instance used by the Dash app.
    """
    threading.Timer(1, open_browser).start()
    app.run_server(debug=True, use_reloader=False)

