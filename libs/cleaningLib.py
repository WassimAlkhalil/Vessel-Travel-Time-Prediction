import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from geopy.distance import geodesic
import searoute

#---------------------------------------------------------------------------------------
# Load and sort .csv file
def preprocess_all_columns(df):
    """
    Preprocess all columns in the DataFrame to replace commas with underscores
    and remove quotes.
    
    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
        
    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    try:
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].apply(lambda x: x.replace(',', '.').replace('"', '').replace("'", '') if isinstance(x, str) else x)
        return df
    except Exception as e:
        print(f"Error in preprocess_all_columns: {e}")
        return df

def load_data(filepath):
    """
    Loads the CSV file into a pandas DataFrame.
    
    Args:
        filepath (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(filepath, delimiter=',', quotechar="\'", engine='python',
                         dtype={'TripID': str, 'MMSI': str, 'ID': str},
                         parse_dates=['StartTime', 'EndTime', 'time'])
        print("Data loaded successfully.")

        df = preprocess_all_columns(df)
        
        if 'StartTime' in df.columns:
            df['StartTime'] = pd.to_datetime(df['StartTime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        if 'EndTime' in df.columns:
            df['EndTime'] = pd.to_datetime(df['EndTime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

#---------------------------------------------------------------------------------------
def save_processed_data(df, filename, base_directory='data/cleaned_data'):
    """
    Saves the processed DataFrame to a new CSV file in the specified directory.
    
    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The filename for the CSV file.
        base_directory (str): The base directory to save the file in.
        
    Returns:
        str: The path to the saved file.
    """
    try:
        # Determines the path of the folder where the current script is running
        script_directory = os.path.dirname(os.path.abspath(__file__)) # "__file__" specifies the path of the current script
        # Creates the full path to the destination folder
        output_directory = os.path.join(script_directory, '..', base_directory)
        output_directory = os.path.normpath(output_directory)
        output_directory = os.path.join(os.getcwd(), base_directory)

        if not os.path.exists(output_directory):
            os.makedirs(output_directory, exist_ok=True)
            print(f"Directory created: {output_directory}")

        # Creates the full path to the file
        if not filename.endswith(".csv"):
            filename += ".csv"
        output_filepath = os.path.join(output_directory, filename)
        df.to_csv(output_filepath, index=False)
        print(f"Processed data successfully saved at:{output_filepath}")
        return output_filepath
    except Exception as e:
        print(f"Error saving data: {e}")
        return None

#---------------------------------------------------------------------------------------
def drop_invalid_time_entries(dataframe, datetime_cols=['StartTime', 'EndTime', 'time']):
    """
    Validates that for each entry, the 'StartTime' is less than the 'EndTime',
    and that the 'time' point falls within these two times. Separates rows based on valid and invalid time conditions.

    Args:
        dataframe (pd.DataFrame): DataFrame containing the trip data with 'StartTime', 'EndTime', and 'time' columns.

    Returns:
        pd.DataFrame: The DataFrame with valid time entries.
    """
    try:
        # Convert columns to datetime and handle errors
        for col in datetime_cols:
            if pd.api.types.is_object_dtype(dataframe[col]):
                dataframe[col] = pd.to_datetime(dataframe[col], errors='coerce')

        # Separate data based on NaN values in datetime conversion
        valid_datetime_df = dataframe.dropna(subset=datetime_cols)

        # Create a mask that checks if each time condition is met
        valid_time_mask = (valid_datetime_df['EndTime'] > valid_datetime_df['StartTime']) & \
                          (valid_datetime_df['time'] >= valid_datetime_df['StartTime']) & \
                          (valid_datetime_df['time'] <= valid_datetime_df['EndTime'])

        # Apply mask to separate valid and invalid time entries
        valid_df = valid_datetime_df[valid_time_mask]

        return valid_df
    except Exception as e:
        print(f"Error in drop_invalid_time_entries: {e}")
        return dataframe

#---------------------------------------------------------------------------------------
def sort_data(dataframe, primary_sort_column='TripID', time_columns=['time']):
    """
    Sorts the DataFrame first by TripID and then by specified time columns in chronological order.
    
    Args:
        dataframe (pd.DataFrame): The DataFrame to be sorted.
        primary_sort_column (str): Primary column name to sort by (typically 'TripID').
        time_columns (list): List of time column names to sort by chronologically after the primary sort.
        
    Returns:
        pd.DataFrame: A sorted DataFrame if successful, original DataFrame otherwise.
    """
    try:
        # Check if all columns are present in the DataFrame
        all_columns = [primary_sort_column] + time_columns
        missing_cols = [col for col in all_columns if col not in dataframe.columns]
        if missing_cols:
            print(f"Error: Missing columns in DataFrame - {', '.join(missing_cols)}")
            return dataframe
        # Checking and converting time data if necessary
        for time_col in time_columns:
            if pd.api.types.is_object_dtype(dataframe[time_col]):
                try:
                    dataframe[time_col] = pd.to_datetime(dataframe[time_col])
                except Exception as e:
                    print(f"Error converting {time_col}: {e}")
                    return dataframe
        # Sort by TripID and then sort by time
        df_sorted = dataframe.sort_values(by=[primary_sort_column] + time_columns)
        print("Data sorted successfully.")
        return df_sorted
    except Exception as e:
        print(f"Error sorting data: {e}")
        return dataframe

#---------------------------------------------------------------------------------------
def remove_duplicate_ids_from_dataframe(df, based_on=['TripID', 'ID', 'MMSI', 'StartTime', 'EndTime']):
    """
    Removes duplicate IDs from a DataFrame grouped by 'TripID', keeping only the first occurrence.
    Returns a DataFrame with duplicates removed.

    Args:
        df (pd.DataFrame): The DataFrame containing all trips with 'TripID' and 'ID'.

    Returns:
        pd.DataFrame: A DataFrame with duplicates removed.
    """
    try:
        # Detect duplicates across the whole DataFrame considering 'TripID', 'ID', 'MMSI', 'StartTime' and 'EndTime' Or specified by the user
        duplicate_mask = df.duplicated(subset=based_on, keep='first')

        # Create DataFrames for non-duplicates and duplicates
        cleaned_df = df[~duplicate_mask]
        return cleaned_df
    except Exception as e:
        print(f"Error in remove_duplicate_ids_from_dataframe: {e}")
        return df

#---------------------------------------------------------------------------------------
def interpolate_circular_data_scipy(df, columns):
    """
    Interpolates missing values in circular data (e.g., angles in degrees) using SciPy.
    
    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the circular data.
        columns (list of str): The columns to interpolate.
        
    Returns:
        pandas.DataFrame: The DataFrame with interpolated circular data.
    """
    try:
        for col in columns:
            # Convert degrees to radians
            radians = np.deg2rad(df[col])

            # Convert from circular to Cartesian coordinates
            x = np.cos(radians)
            y = np.sin(radians)
            
            # Find non-NaN indices
            valid_idx = ~np.isnan(x)
            idx = np.arange(len(x))
            
            # Interpolate the Cartesian coordinates using scipy's interp1d
            x_interp = interp1d(idx[valid_idx], x[valid_idx], kind='linear', fill_value='extrapolate')(idx)
            y_interp = interp1d(idx[valid_idx], y[valid_idx], kind='linear', fill_value='extrapolate')(idx)
            
            # Convert back to radians and then to degrees
            interpolated_radians = np.arctan2(y_interp, x_interp)
            df[col] = np.rad2deg(interpolated_radians).round(2) % 360

        return df
    except Exception as e:
        print(f"Error in interpolate_circular_data_scipy: {e}")
        return df

#---------------------------------------------------------------------------------------
def process_cog_and_th_data(df, threshold=5, drop=False):
    """
    Processes 'COG' and 'TH' for each trip separately by setting invalid values (greater than 359) to NaN,
    interpolates missing values using circular data interpolation,
    calculates the direction changes accounting for circular continuity,
    and flags significant changes. Optionally drops rows where values remain NaN after interpolation.

    Args:
        df (pd.DataFrame): The DataFrame containing 'COG', 'TH', and 'TripID' data.
        threshold (float): The threshold for significant direction change.
        drop (bool): Whether to drop rows with NaN values in 'COG' or 'TH' after interpolation.

    Returns:
        pd.DataFrame: DataFrame with processed 'COG' and 'TH' data including direction changes.
        pd.DataFrame: DataFrame with removed data if drop is True, else None.
    """
    try:
        # Make a copy of the DataFrame to avoid SettingWithCopyWarning
        df = df.copy()

        # Set invalid values to NaN and convert to numeric is done in 'make_trip_consistent'
        # Normalize values to the range 0-360
        df['COG'] = df['COG'].apply(lambda x: (x % 360))
        df['TH'] = df['TH'].apply(lambda x: (x % 360))

        # Initialize a DataFrame to capture removed data if needed
        removed_data = pd.DataFrame()

        # Process each trip individually
        df['DirectionChange'] = np.nan
        df['SignificantDirectionChange'] = np.nan

        for trip_id, group in df.groupby('TripID'):
            original_index = group.index  # Store original index to reference later
            group = interpolate_circular_data_scipy(group, ['COG', 'TH'])

            # Calculate direction change taking circular nature into account
            group['DirectionChange'] = group['COG'].diff().fillna(0).abs().mod(360)
            group['DirectionChange'] = group['DirectionChange'].apply(lambda x: x if x <= 180 else 360 - x)

            # Identify significant direction changes
            group['SignificantDirectionChange'] = group['DirectionChange'].apply(lambda x: 1 if x > threshold else 0)

            # Round the numerical values to two decimal places
            group['COG'] = group['COG'].round(2)
            group['TH'] = group['TH'].round(2)
            group['DirectionChange'] = group['DirectionChange'].round(2)
            
            df.loc[group.index, ['COG', 'TH', 'DirectionChange', 'SignificantDirectionChange']] = group[['COG', 'TH', 'DirectionChange', 'SignificantDirectionChange']]

            if drop:
                # Identify rows where NaNs still exist and remove them
                still_na = group['COG'].isna() | group['TH'].isna()
                removed_data = pd.concat([removed_data, group[still_na]])
                group = group.dropna(subset=['COG', 'TH'])

            # Ensure only updated parts of the group are re-assigned to the main DataFrame
            df.loc[original_index] = group

        if drop:
            return df, removed_data
        else:
            return df, None  # Return None for removed_data if no rows are dropped
    except Exception as e:
        print(f"Error in process_cog_and_th_data: {e}")
        return df, None

#---------------------------------------------------------------------------------------
def replace_with_most_frequent(group, labels, inconsistent_count):
    """
    Replace values in the specified columns of the group with the most frequent value.
    
    Args:
        group (pd.DataFrame): The group of data to process.
        labels (list of str): The columns to check for inconsistencies.
        inconsistent_count (dict): Dictionary to count inconsistencies.
        
    Returns:
        pd.DataFrame: The processed group with most frequent values replaced.
    """
    try:
        for label in labels:
            most_frequent = group[label].mode()
            if len(most_frequent) > 0:
                most_frequent_value = most_frequent[0]
                if group[label].nunique() > 1:
                    inconsistent_count[label] += 1
                group[label] = most_frequent_value
        return group
    except Exception as e:
        print(f"Error in replace_with_most_frequent: {e}")
        return group

def make_trip_consistent(df: pd.DataFrame, print_info=False) -> pd.DataFrame:
    """
    Makes each trip consistent by replacing inconsistent values with the most common value in the trip.

    Args:
        df (pd.DataFrame): The DataFrame to make consistent.
        print_info (bool): Whether to print information about the consistency of the DataFrame.

    Returns:
        pd.DataFrame: The consistent DataFrame.
    """
    try:
        # Replace '?' with NaN
        new_df = df.replace("?", np.nan).copy()

        # Convert relevant columns to numeric, setting invalid parsing to NaN
        numeric_cols = ['Draught', 'Length', 'Breadth', 'StartLatitude', 'StartLongitude', 'EndLatitude', 'EndLongitude', 'Latitude', 'Longitude', 'SOG', 'COG', 'TH']
        for col in numeric_cols:
            new_df[col] = pd.to_numeric(new_df[col], errors='coerce')

        # Define columns to check for consistency
        labels = ['shiptype', 'Length', 'Breadth', 'Draught', 'StartTime', 'EndTime',
                'Destination', 'Name', 'StartPort', 'EndPort', 'Callsign', 'MMSI',
                'StartLatitude', 'StartLongitude', 'EndLatitude', 'EndLongitude']

        # Drop rows with missing TripID or ID
        new_df.dropna(subset=['TripID', 'ID'], inplace=True)

        # Initialize count for inconsistent trips
        inconsistent_count = {label: 0 for label in labels}

        # Group by TripID and apply consistency check
        new_df = new_df.groupby('TripID').apply(replace_with_most_frequent, labels, inconsistent_count).reset_index(drop=True)

        # Print inconsistency info if required
        if print_info:
            for label, count in inconsistent_count.items():
                if count > 0:
                    print(f"{count} trips had inconsistent '{label}' values. Replaced with the most frequent value in a trip.")

        return new_df
    except Exception as e:
        print(f"Error in make_trip_consistent: {e}")
        return df

#---------------------------------------------------------------------------------------
# Only in sea
def searoute_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the shortest path distance between two geographical points over the sea.
    
    Args:
        lat1 (float): Latitude of the first point.
        lon1 (float): Longitude of the first point.
        lat2 (float): Latitude of the second point.
        lon2 (float): Longitude of the second point.
        
    Returns:
        float: The shortest path distance in meters.
    """
    # Optionally, define the units for the length calculation included in the properties object.
    # Defaults to km, can be 'm' = meters 'mi = miles 'ft' = feet 'in' = inches 'deg' = degrees
    # 'cen' = centimeters 'rad' = radians 'naut' = nautical 'yd' = yards
    try:
        route = searoute.searoute([lon1, lat1], [lon2, lat2], units="m")
        return round(route.properties['length'], 2)
    except Exception as e:
        print(f"Error in searoute_distance: {e}")
        return None

#---------------------------------------------------------------------------------------
def geopy_haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance using geopy.
    
    Args:
        lat1 (float): Latitude of the first point.
        lon1 (float): Longitude of the first point.
        lat2 (float): Latitude of the second point.
        lon2 (float): Longitude of the second point.
        
    Returns:
        float: The great circle distance in kilometers.
    """
    try:
        point1 = (lat1, lon1)
        point2 = (lat2, lon2)
        return round(great_circle(point1, point2).km, 2)
    except Exception as e:
        print(f"Error in geopy_haversine: {e}")
        return None

#---------------------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees).
    
    Args:
        lat1 (float): Latitude of the first point.
        lon1 (float): Longitude of the first point.
        lat2 (float): Latitude of the second point.
        lon2 (float): Longitude of the second point.
        
    Returns:
        float: The great circle distance in kilometers.
    """
    try:
        # convert decimal degrees to radians 
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        # haversine formula
        dlat = lat2 - lat1 
        dlon = lon2 - lon1 
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)) 
        r = 6371.00  # Radius of earth in kilometers
        distance = c * r
        return round(distance, 2) # Round the result to two decimal places
    except Exception as e:
        print(f"Error in haversine: {e}")
        return None

#---------------------------------------------------------------------------------------
def detect_outliers_dbscan(data, eps=1.0, min_samples=5):
    """
    Detect outliers in geospatial data using DBSCAN.
    
    Args:
        data (pd.DataFrame): A pandas DataFrame with 'Latitude' and 'Longitude' columns.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        
    Returns:
        pd.DataFrame: The input DataFrame with an additional 'outlier' column.
    """
    try:
        # Extract latitude and longitude
        coords = data[['Latitude', 'Longitude']].values
        # Apply DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        data['outlier'] = db.labels_
        # Convert clusters to outliers (label -1 as outliers)
        data['outlier'] = data['outlier'].apply(lambda x: 1 if x == -1 else 0)
        return data
    except Exception as e:
        print(f"Error in detect_outliers_dbscan: {e}")
        return data

#---------------------------------------------------------------------------------------
def detect_outliers_critical_distance_haversine(data, critical_distance):
    """
    Detect outliers based on critical distance between consecutive points.

    Args:
        data (pd.DataFrame): A pandas DataFrame with 'Latitude' and 'Longitude' columns.
        critical_distance (float): The maximum allowed distance between consecutive points in kilometers.

    Returns:
        pd.DataFrame: The input DataFrame with an additional 'outlier' column.
    """
    try:
        data['outlier'] = 0
        for i in range(1, len(data)):
            lat1, lon1 = data.at[i-1, 'Latitude'], data.at[i-1, 'Longitude']
            lat2, lon2 = data.at[i, 'Latitude'], data.at[i, 'Longitude']
            distance = haversine(lat1, lon1, lat2, lon2)
            if distance > critical_distance:
                data.at[i, 'outlier'] = 1
        return data
    except Exception as e:
        print(f"Error in detect_outliers_critical_distance_haversine: {e}")
        return data

#---------------------------------------------------------------------------------------
def detect_outliers_critical_distance(data, critical_distance):
    """
    Detect outliers based on critical distance between consecutive points.

    Args:
        data (pd.DataFrame): A pandas DataFrame with 'Latitude' and 'Longitude' columns.
        critical_distance (float): The maximum allowed distance between consecutive points in kilometers.

    Returns:
        pd.DataFrame: The input DataFrame with an additional 'outlier' column.
    """
    try:
        # Ensure the index is a range index
        data = data.reset_index(drop=True)

        # Add the 'outlier' column
        data['outlier'] = 0

        # Iterate over the rows and calculate distances
        for i in range(1, len(data)):
            point1 = (data.at[i-1, 'Latitude'], data.at[i-1, 'Longitude'])
            point2 = (data.at[i, 'Latitude'], data.at[i, 'Longitude'])
            distance = geodesic(point1, point2).kilometers
            if distance > critical_distance:
                data.at[i, 'outlier'] = 1
        return data
    except Exception as e:
        print(f"Error in detect_outliers_critical_distance: {e}")
        return data

#---------------------------------------------------------------------------------------
def detect_outliers_zscore(data, threshold=3):
    """
    Detect outliers based on z-score.
    
    Args:
        data (pd.DataFrame): A pandas DataFrame with 'Latitude' and 'Longitude' columns.
        threshold (float): The z-score threshold above which a point is considered an outlier.
        
    Returns:
        pd.DataFrame: The input DataFrame with an additional 'outlier' column.
    """
    try:
        z_scores = np.abs(zscore(data[['Latitude', 'Longitude']]))
        data['outlier'] = ((z_scores > threshold).any(axis=1)).astype(int)
        return data
    except Exception as e:
        print(f"Error in detect_outliers_zscore: {e}")
        return data

#---------------------------------------------------------------------------------------
def validate_and_clean_coordinates(df, method='dbscan', threshold=3, critical_distance=5, eps=1.0, min_samples=5):
    """
    Validates, corrects, or removes geographic coordinates within each trip group based on logical,
    completeness, and outlier detection criteria. Interpolates outliers if possible, otherwise removes them.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        method (str): The method to use for outlier detection ('dbscan', 'critical_distance', 'zscore', 'haver').
        threshold (float): The z-score threshold for outlier detection when method is 'zscore'.
        critical_distance (float): The critical distance for outlier detection when method is 'critical_distance' or 'haver'.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other when method is 'dbscan'.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point when method is 'dbscan'.

    Returns:
        tuple: Two DataFrames:
               - The first with entries having valid, logical, and complete coordinate data.
               - The second with entries removed due to being uncorrectable outliers or other invalid data.
    """
    try:
        coords_columns = ['Latitude', 'Longitude']
        cleaned_df = pd.DataFrame()
        invalid_data_df = pd.DataFrame()

        for trip_id, group in df.groupby('TripID'):
            group = group.copy()
            # Convert data types to float, if conversion fails replace with NaN is done in 'make_trip_consistent()'
            # Forward-fill and backward-fill to handle the first and last values
            group.reset_index(drop=True, inplace=True)
            group[coords_columns] = group[coords_columns].interpolate(method='linear', limit_direction='both').round(2)

            # Handle initial missing values
            initial_na_mask = group[coords_columns].isna().any(axis=1)
            initial_invalid_group = group[initial_na_mask]
            group = group[~initial_na_mask]
            
            # Check for logical bounds
            condition = (group['Latitude'].between(-90, 90)) & (group['Longitude'].between(-180, 180))
            initial_invalid_group = pd.concat([initial_invalid_group, group[~condition]], ignore_index=True)
            group = group[condition]

            if method == 'dbscan':
                group = detect_outliers_dbscan(group, eps, min_samples)
            elif method == 'critical_distance':
                group = detect_outliers_critical_distance(group, critical_distance)
            elif method == 'zscore':
                group = detect_outliers_zscore(group, threshold)
            elif method == 'haver':
                group = detect_outliers_critical_distance_haversine(group, critical_distance)
            else:
                raise ValueError("Invalid method specified. Choose 'dbscan', 'critical_distance', or 'zscore'.")

            outliers_mask = group['outlier'] == 1
            initial_invalid_group = pd.concat([initial_invalid_group, group[outliers_mask]], ignore_index=True)
            group.loc[outliers_mask, coords_columns] = None
            
            # Attempt to correct outliers by interpolation
            group.reset_index(drop=True, inplace=True)
            group[coords_columns] = group[coords_columns].interpolate(method='linear', limit_direction='both').round(2)

            # Drop non-correctable outliers if any remain
            group = group.dropna(subset=coords_columns)

            # Remove temporary columns
            group.drop(columns=['outlier'], inplace=True)

            # Append to respective DataFrames
            cleaned_df = pd.concat([cleaned_df, group], ignore_index=True)
            invalid_data_df = pd.concat([invalid_data_df, initial_invalid_group], ignore_index=True)

        return cleaned_df, invalid_data_df
    except Exception as e:
        print(f"Error in validate_and_clean_coordinates: {e}")
        return df, pd.DataFrame()

#---------------------------------------------------------------------------------------
def process_sog_data(df, min_speed=0, threshold=3):
    """
    Processes 'SOG' (Speed Over Ground) data within a DataFrame to interpolate missing values,
    handle outliers, and calculate speed metrics for each trip.

    Args:
        df (pd.DataFrame): The DataFrame containing all trip data.
        min_speed (float): The minimum valid speed value.
        threshold (float): The threshold multiplier for determining outliers based on standard deviations.

    Returns:
        pd.DataFrame: The cleaned DataFrame with missing and outlier 'SOG' values handled.
        pd.DataFrame: DataFrame containing all rows invalid due to missing or outlier 'SOG' values.
    """
    try:
        cleaned_df = pd.DataFrame()
        invalid_df = pd.DataFrame()

        # Process each trip separately
        for trip_id, group in df.groupby('TripID'):
            group = group.copy()
            # Convert data types to float, if conversion fails replace with NaN is done in 'make_trip_consistent()'
            # Handle missing values by interpolation
            group['SOG'] = group['SOG'].interpolate(method='linear', limit_direction='both').round(2)
            
            # Detect and handle outliers based on predefined valid range
            mean_sog = group['SOG'].mean()
            std_sog = group['SOG'].std()
            outliers_mask = ((group['SOG'] < min_speed) | (group['SOG'] > (mean_sog + threshold * std_sog)))
            group.loc[outliers_mask, 'SOG'] = np.nan
            group_invalid = group[outliers_mask & group['SOG'].isna()]
            group['SOG'] = group['SOG'].interpolate(method='linear', limit_direction='both').round(2)

            # Replace remaining NaNs (at start/end) with nearest valid values
            group['SOG'] = group['SOG'].bfill().ffill()

            # Separate outliers after interpolation
            group_cleaned = group.dropna(subset=['SOG'])

            # Append to final DataFrames
            cleaned_df = pd.concat([cleaned_df, group_cleaned], ignore_index=True)
            invalid_df = pd.concat([invalid_df, group_invalid], ignore_index=True)
            
        return cleaned_df, invalid_df
    except Exception as e:
        print(f"Error in process_sog_data: {e}")
        return df, pd.DataFrame()

#---------------------------------------------------------------------------------------
def process_and_construct_new_features(df):
    """
    Processes each trip group to compute distances, speeds, and directional changes and combines them into a single DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing trip data.

    Returns:
        pd.DataFrame: A DataFrame containing all processed trips.
        dict: Dictionary of DataFrames, each processed group keyed by 'TripID'.
    """
    try:
        processed_groups = []
        trip_dict = {}

        for trip_id, group in df.groupby('TripID'):
            # Calculate distances
            group['DistanceFromStart'] = group.apply(lambda row: geopy_haversine(row['StartLatitude'], row['StartLongitude'], row['Latitude'], row['Longitude']), axis=1)
            group['DistanceFromEnd'] = group.apply(lambda row: geopy_haversine(row['EndLatitude'], row['EndLongitude'], row['Latitude'], row['Longitude']), axis=1)

            # Time calculations if needed (convert to total seconds)
            group['TimeFromStart'] = (group['time'] - group['StartTime']).dt.total_seconds().round(2)
            group['TimeUntilEnd'] = (group['EndTime'] - group['time']).dt.total_seconds().round(2)

            # Shift Latitude and Longitude for previous and next calculations
            group['Latitude_prev'] = group['Latitude'].shift()
            group['Longitude_prev'] = group['Longitude'].shift()
            group['Latitude_next'] = group['Latitude'].shift(-1)
            group['Longitude_next'] = group['Longitude'].shift(-1)

            # Calculate distances from/to checkpoints with immediate rounding
            group['DistanceFromLastCheckpoint'] = group.apply(
                lambda row: geopy_haversine(row['Latitude'], row['Longitude'], row['Latitude_prev'], row['Longitude_prev'])
                if pd.notnull(row['Latitude']) and pd.notnull(row['Longitude']) and pd.notnull(row['Latitude_prev']) and pd.notnull(row['Longitude_prev'])
                else 0, axis=1)
            
            group['DistanceToNextCheckpoint'] = group.apply(
                lambda row: geopy_haversine(row['Latitude'], row['Longitude'], row['Latitude_next'], row['Longitude_next'])
                if pd.notnull(row['Latitude']) and pd.notnull(row['Longitude']) and pd.notnull(row['Latitude_next']) and pd.notnull(row['Longitude_next'])
                else 0, axis=1)

            # Calculate time differences from last and to next checkpoints with immediate rounding
            group['TimeFromLastCheckpoint'] = group['time'].diff().dt.total_seconds().fillna(0).round(2)
            group['TimeToNextCheckpoint'] = group['time'].diff(-1).dt.total_seconds().fillna(0).round(2)

            # Define the initial rolling window size
            initial_window = 5

            # Manually set the first row values
            group.loc[group.index[0], 'Max_SOG'] = group.loc[group.index[0], 'SOG']
            group.loc[group.index[0], 'Min_SOG'] = group.loc[group.index[0], 'SOG']
            group.loc[group.index[0], 'Mean_SOG'] = group.loc[group.index[0], 'SOG']
            group.loc[group.index[0], 'Mean_COG'] = group.loc[group.index[0], 'COG']

            # Speed over ground statistics
            group['Max_SOG'] = group['SOG'].rolling(window=initial_window, min_periods=1).max().combine_first(
                group['SOG'].expanding(min_periods=initial_window).max()).round(2)
            group['Min_SOG'] = group['SOG'].rolling(window=initial_window, min_periods=1).min().combine_first(
                group['SOG'].expanding(min_periods=initial_window).min()).round(2)
            group['Mean_SOG'] = group['SOG'].rolling(window=initial_window, min_periods=1).mean().combine_first(
                group['SOG'].expanding(min_periods=initial_window).mean()).round(2)

            # Course statistics
            group['Mean_COG'] = group['COG'].rolling(window=initial_window, min_periods=1).mean().combine_first(
                group['COG'].expanding(min_periods=initial_window).mean()).round(2)

            # Trip duration
            group['TotalTripTime'] = (group['EndTime'] - group['StartTime']).dt.total_seconds().round(2)

            # Total distance
            group['TotalDistance'] = round(geopy_haversine(
                group['StartLatitude'].iloc[0], group['StartLongitude'].iloc[0], 
                group['EndLatitude'].iloc[0], group['EndLongitude'].iloc[0]
            ), 2)

            # Drop the temporary columns
            group.drop(columns=['Latitude_prev', 'Longitude_prev', 'Latitude_next', 'Longitude_next'], inplace=True)

            # Store processed group
            processed_groups.append(group)
            trip_dict[trip_id] = group

        # Concatenate all processed groups back into a single DataFrame
        concatenated_df = pd.concat(processed_groups)

        return concatenated_df, trip_dict
    except Exception as e:
        print(f"Error in process_and_construct_new_features: {e}")
        return df, {}

#---------------------------------------------------------------------------------------
def cleaning_data(loadPath = None, df = None, saveUndername = None, base_directory='data/cleaned_data'):
    """
    Cleans and processes the data from a CSV file, applying various data cleaning and feature engineering steps.

    Args:
        loadPath (str): The path to the CSV file to load.
        base_directory (str): The base directory to save the cleaned data file.

    Returns:
        tuple: The processed DataFrame and the path to the cleaned data file.
    """
    try:
        if not (df or loadPath):
            raise Exception("No data provided to clean")
        
        # Define a base name for saving processed files
        if not saveUndername:
            if loadPath:
                base_name = os.path.basename(loadPath)
                saveUnderName , _ = os.path.splitext(base_name)
            else:
                saveUnderName = 'Single_File'
        else:
            saveUnderName = saveUndername

        if not df and loadPath:
            # Load data from the specified path and notify the user
            print("The data will be loaded now...")
            df = load_data(loadPath)

        # Make data consistent by possibly standardizing or cleaning formats
        print("Make data consistent by possibly standardizing...")
        consist_df = make_trip_consistent(df)

        # Remove duplicate records based on ['TripID', 'ID', 'MMSI', 'StartTime', 'EndTime'] in default from the dataframe
        print("Remove duplicate records based on ['TripID', 'ID', 'MMSI', 'StartTime', 'EndTime']...")
        cl_df = remove_duplicate_ids_from_dataframe(consist_df)
        
        # Sort the data, typically by datetime
        print("Sort the data, typically by datetime...")
        sorted_df = sort_data(cl_df)

        # Drop entries with invalid time data
        print("The time will be validated now...")
        valid_df = drop_invalid_time_entries(sorted_df)
        
        # Validate and clean geographic coordinates, splitting into cleaned and invalid(corrected) dataframes
        print("The coordinates will be validated now...")
        cl_crd_df, rmv_crd_df = validate_and_clean_coordinates(valid_df)
        
        # Process course over ground and temperature/humidity data, return main and auxiliary dataframes
        print("The COG and TH data will be validated now...")
        direction_df, temp = process_cog_and_th_data(cl_crd_df)
        
        # Process and filter data based on speed over ground
        print("The SOG will be validated now...")
        cleaned_df, removed_df = process_sog_data(direction_df)

        # Final processing of the cleaned data and construction of new features
        print("Final processing of the cleaned data and construction of new features will be done now...")
        df_processed, df_dec = process_and_construct_new_features(cleaned_df)

        # Save the fully processed and cleaned data to a new file
        print("Save the fully processed and cleaned data to a new file...")
        cleanedPath = save_processed_data(df_processed, "{}_Fully_Data_No_Rad_Filter.csv".format(saveUnderName), base_directory)

        return df_processed, cleanedPath
    except Exception as e:
        print(f"Error in cleaning_data: {e}")
        return None, None

#---------------------------------------------------------------------------------------
def split_and_save_trips_radius_km_based(dataframe, central_start_lat, central_start_lon, central_end_lat, central_end_lon, radius_km=20):
    """
    Splits the DataFrame based on trip locations and returns each group of trips in a dictionary.
    
    Args:
        dataframe (pd.DataFrame): The DataFrame containing the trips.
        central_start_lat (float): The latitude of the central start point for grouping.
        central_start_lon (float): The longitude of the central start point for grouping.
        central_end_lat (float): The latitude of the central end point for grouping.
        central_end_lon (float): The longitude of the central end point for grouping.
        radius_km (float): The radius in kilometers for grouping trips around the central points.
    
    Returns:
        dict: A dictionary of grouped DataFrames categorized by group.
    """
    try:
        grouped_data = {
            'start_in_end_in': [],
            'start_in_end_out': [],
            'start_out_end_in': [],
            'start_out_end_out': []
        }
        # Iterate through each trip and group them based on the distance to the central points
        for trip_id, trip_data in dataframe.groupby('TripID'):
            start_lat = trip_data['StartLatitude'].iloc[0]
            start_lon = trip_data['StartLongitude'].iloc[0]
            end_lat = trip_data['EndLatitude'].iloc[0]
            end_lon = trip_data['EndLongitude'].iloc[0]

            start_within_radius = geopy_haversine(central_start_lat, central_start_lon, start_lat, start_lon) <= radius_km
            end_within_radius = geopy_haversine(central_end_lat, central_end_lon, end_lat, end_lon) <= radius_km

            if start_within_radius and end_within_radius:
                group = 'start_in_end_in'
            elif start_within_radius and not end_within_radius:
                group = 'start_in_end_out'
            elif not start_within_radius and end_within_radius:
                group = 'start_out_end_in'
            else:
                group = 'start_out_end_out'

            grouped_data[group].append(trip_data)

        # Combine grouped data into DataFrames
        for group in grouped_data:
            if grouped_data[group]:
                grouped_data[group] = pd.concat(grouped_data[group], ignore_index=True)
            else:
                grouped_data[group] = pd.DataFrame()

        return grouped_data

    except Exception as e:
        print(f"Error in split_and_return_trips_radius_km_based: {e}")
        return {}
