import os
import sys
import pandas as pd

# Determine the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the libs directory to the system path
libs_dir = os.path.join(current_dir, '..', 'libs')
sys.path.append(libs_dir)

# Import custom library functions
import cleaningLib as lod
import visualizationLib as vis

def main():
    # Define file paths for input data sets
    filepath_input1 = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned_data', 'Bremen_Hamburg_Rad_Filtered_Data.csv')
    filepath_input2 = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned_data', 'Kiel_Gdynia_Rad_Filtered_Data.csv')

    fileLoad = filepath_input1

    # data laden
    print("\nThe data will be loaded now...\n")
    cleaned_data = lod.load_data(fileLoad)

    if cleaned_data is None:
        print(f"\nError loading data: File '{fileLoad}' not found or could not be loaded.")
        return

    print("\nClick on the link to open the visualisation in your browser.\n")
    # Create the app
    app, server = vis.create_app(cleaned_data)

    # Run the app
    vis.run_app(app, server)

if __name__ == '__main__':
    main()
