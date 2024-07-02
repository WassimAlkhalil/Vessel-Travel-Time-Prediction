import os
import sys
import pandas as pd

# Determine the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the libs directory to the system path
libs_dir = os.path.join(current_dir, '..', 'libs')
sys.path.append(libs_dir)

# Import custom library functions
from cleaningLib import load_data
from machineLearningLib import train_models, predict_eta

def main():
    try:
        # Define file paths for input data sets
        filepath_input1 = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned_data', 'Bremen_Hamburg_Rad_Filtered_Data.csv')
        filepath_input2 = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned_data', 'Kiel_Gdynia_Rad_Filtered_Data.csv')
        save_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'results')
        
        # Load the dataset
        data1 = load_data(filepath_input1)
        data2 = load_data(filepath_input2)

        data = data1
        # Train and save models
            # features = ['Latitude', 'Longitude', 'SOG', 'COG', 'DirectionChange', 'Mean_SOG', 'DistanceFromStart', 'DistanceFromEnd', 'TimeFromStart']
            # target = 'TotalTripTime'
        print("Train and save models...")
        # def train_models(data_frame, dataset=None, clean_path=False, models_names=['ExtraTrees', 'RandomForest'], test_size=0.2, random_state=42, skip_hyperparameter=True, total_time=False, save_model=False, save_path = './data/results'):
        results, best_model, best_model_instance, stacking_model, stacking_metrics, stacking_path = train_models(data, dataset='BREMERHAVEN_to_HAMBURG', save_path=save_path) #---------------------------------------------------->> training the model with data set-1

        data = data2
        # Train and save models
            # used features: ['Latitude', 'Longitude', 'SOG', 'COG', 'DirectionChange', 'Mean_SOG', 'DistanceFromStart', 'DistanceFromEnd', 'TimeFromStart']
            # used target: 'TotalTripTime'
        print("Train and save models...")
        results, best_model, best_model_instance, stacking_model, stacking_metrics, stacking_path = train_models(data, dataset='KIEL_to_GDYNIA', save_path=save_path) #---------------------------------------------------->> training the model with data set-2
        
        print(f"Stacking Metrics: {stacking_metrics['name']}")
        print(f"Best Model: {best_model['name']}")
        print(f"Mean Absolute Error: {best_model['mae']}")

        # Example prediction with the best model based on coordinates
        print(" Example prediction with the best model based on coordinates...")
        latitude = 54.36
        longitude = 10.14
        start_lat = 54.36
        start_lon = 10.14
        end_lat = 54.38
        end_lon = 18.71
        start_time = pd.Timestamp.now()
        time = pd.Timestamp.now() + pd.Timedelta(hours=1)

        # def predict_eta(latitude, longitude, time, start_lat, start_lon, start_time, end_lat, end_lon,
        #                 sog=None, cog=None, direction_change=None, mean_sog=None, time_from_start=None, model_path=None, model_instance=None)
        eta = predict_eta(latitude=latitude, longitude=longitude, time=time, start_lat=start_lat, start_lon=start_lon, start_time=start_time, end_lat=end_lat, end_lon=end_lon,
              model_path=stacking_path) #---------------------------------------------------->> Example prediction with the best model based on coordinates
            
        if eta is not None:
            print(f"\nPredicted ETA given coordinates: {eta/3600} hours")
    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == '__main__':
    main()