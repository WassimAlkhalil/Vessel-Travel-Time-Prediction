import os
import sys
import pandas as pd

# Determine the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the libs directory to the system path
libs_dir = os.path.join(current_dir, '..', 'libs')
sys.path.append(libs_dir)

# Import custom library functions for data cleaning
import cleaningLib as lod

def main():
    # Define file paths for input data sets
    filepath_input1 = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_data', 'bremerhaven_hamburg.csv')
    filepath_input2 = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_data', 'kiel_gdynia.csv')
    filepath_outpu = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned_data')

    # cleaning_data(loadPath = None, df = None, saveUndername = None, base_directory='data/cleaned_data'):
    df1, pth1 = lod.cleaning_data(loadPath=filepath_input1, base_directory=filepath_outpu) # for cleaning bremerhaven_hamburg.csv

    # Splits the DataFrame bremerhaven_hamburg_cleaned 
    # df = lod.load_data("./data/cleaned_data/bremerhaven_hamburg_cleaned.csv") # the cleaned data will be used with central point of start/end-coordinats-port
    sp_data_group1 = lod.split_and_save_trips_radius_km_based(df1, 53.549999, 8.583333, 53.507097, 9.967923) # rad = 20 km
    df1 = sp_data_group1['start_in_end_in']
    # save_processed_data(df, filename, base_directory='data/cleaned_data')
    lod.save_processed_data(df1, 'Bremen_Hamburg_Rad_Filtered_Data', base_directory=filepath_outpu)

    # cleaning_data(loadPath = None, df = None, saveUndername = None, base_directory='data/cleaned_data'):
    df2, pth2 = lod.cleaning_data(loadPath=filepath_input2, base_directory=filepath_outpu) # for cleaning kiel_gdynia.csv

    # Splits the DataFrame kiel_gdynia
    # df = lod.load_data("./data/cleaned_data/kiel_gdynia_cleaned.csv") # the cleaned data will be used with central point of start/end-coordinats-port
    sp_data_group2 = lod.split_and_save_trips_radius_km_based(df2, 54.3179, 10.1389, 54.533831198, 18.570497718) # rad = 20 km
    df2 = sp_data_group2['start_in_end_in']
    # save_processed_data(df, filename, base_directory='data/cleaned_data')
    lod.save_processed_data(df2, 'Kiel_Gdynia_Rad_Filtered_Data', base_directory=filepath_outpu)

if __name__ == "__main__":
    main()
