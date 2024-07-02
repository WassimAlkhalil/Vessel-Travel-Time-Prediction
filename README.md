# Ship Voyages Dataset

The program includes several classes for handling different tasks within a PyQt application: Worker, CleaningWorker, TrainingWorker, and PredictionWorker. The main UI is managed by MainWindow. Key functions handle input validation, file selection, UI updates, and process initiation for data cleaning, model training, and prediction. The application requires Python 3.6+, and libraries like PyQt5, folium, pandas, and joblib. It processes CSV files containing ship trip data, performing data cleaning, model training, and prediction, with results visualized on maps. UI components include input fields, buttons, checkboxes, combo boxes, web views, and a text browser for status logs. The program also includes error handling and performance considerations to ensure responsiveness and extensibility for future modifications.

#### Note

A file to be cleaned must contain exactly these columns: `TripID, MMSI, StartLatitude, StartLongitude, StartTime, EndLatitude, EndLongitude, EndTime, StartPort, EndPort, ID, time, shiptype, Length, Breadth, Draught, Latitude, Longitude, SOG, COG, TH, Destination, Name, Callsign, AisSourcen, DirectionChange, SignificantDirectionChange, DistanceFromStart, DistanceFromEnd, TimeFromStart, TimeUntilEnd, DistanceFromLastCheckpoint, DistanceToNextCheckpoint, TimeFromLastCheckpoint, TimeToNextCheckpoint, Max_SOG, Min_SOG, Mean_SOG, Mean_COG, TotalTripTime, TotalDistance`.

## Overview
The Ship Voyages dataset contains detailed information about ship voyages, captured through Automatic Identification System (AIS) messages. The data is stored in CSV files named `bremerhaven_hamburg.csv` and `hamburg_bremerhaven.csv` in the `data` directory. Each row in these files represents a unique AIS message, offering comprehensive details about the voyages and associated characteristics.

## Data Dictionary
The data dictionary provides a detailed description of each attribute in the dataset.

| Attribute       | Data Type | Description |
|-----------------|-----------|-------------|
| TripID          | Integer   | A unique identifier for each trip. |
| MMSI            | Integer   | Maritime Mobile Service Identity, a unique number assigned to each ship for identification purposes. |
| StartLatitude   | Float     | Latitude coordinate of the trip's planned starting position. |
| StartLongitude  | Float     | Longitude coordinate of the trip's planned starting position. |
| StartTime       | Timestamp | Scheduled start time of the trip. |
| EndLatitude     | Float     | Latitude coordinate of the trip's planned ending position. |
| EndLongitude    | Float     | Longitude coordinate of the trip's planned ending position. |
| EndTime         | Timestamp | Scheduled end time of the trip. |
| StartPort       | String    | The port from which the trip commences. |
| EndPort         | String    | The port at which the trip concludes. |
| ID              | Integer   | Identifier of the AIS message. |
| time            | Timestamp | Timestamp of the AIS message. |
| shiptype        | String    | Classification type of the ship. |
| Length          | Float     | Length of the ship, measured in meters. |
| Breadth         | Float     | Breadth of the ship, measured in meters. |
| Draught         | Float     | The vertical distance between the waterline and the bottom of the hull (keel), measured in meters. |
| Latitude        | Float     | Current latitude of the ship during the AIS message. |
| Longitude       | Float     | Current longitude of the ship during the AIS message. |
| SOG             | Float     | Speed over ground of the ship, in knots. |
| COG             | Float     | Course over ground of the ship, in degrees. |
| TH              | Float     | True heading of the ship, the angle relative to true north. |
| Destination     | String    | Intended destination of the ship at the time of the AIS message. |
| Name            | String    | Name of the ship. |
| Callsign        | String    | Radio call sign of the ship, used for radio communications. |
| AisSource       | String    | Source from which the AIS data was obtained. |

## Derived Attributes
The following derived attributes are calculated to provide additional insights:

| Column Name                 | Data Type | Description |
|-----------------------------|-----------|-------------|
| DirectionChange             | Numeric   | Change in the vessel's heading in degrees since the last recorded position. |
| SignificantDirectionChange  | Boolean   | Indicates significant direction change since the last recorded position. |
| Mean_SOG                    | Numeric   | The average Speed Over Ground (SOG) of the vessel during the voyage. |
| DistanceFromStart           | Numeric   | Distance traveled from the start point of the journey. |
| DistanceFromEnd             | Numeric   | Remaining distance to the end point of the journey. |
| TimeFromStart               | Timestamp | Time elapsed from the start of the journey. |
| TimeUntilEnd                | Timestamp | Estimated time remaining until the end of the journey. |
| DistanceFromLastCheckpoint  | Numeric   | Distance traveled from the last recorded checkpoint. |
| DistanceToNextCheckpoint    | Numeric   | Distance to the next checkpoint from the current position. |
| TimeFromLastCheckpoint      | Timestamp | Time taken from the last checkpoint to the current position. |
| TimeToNextCheckpoint        | Timestamp | Estimated time to reach the next checkpoint from the current position. |

## Data Processing and Cleaning

### Overview
The codebase includes functionalities for managing and processing maritime trip data in CSV formats, ensuring data integrity and enhancing analysis capabilities. Key features include:

- **Data Loading and Parsing**: Handles CSV files with error checks and corrects date-time parsing issues.
- **Data Cleaning and Sorting**: Removes duplicates, validates time entries, and sorts data based on trip identifiers and time.
- **Navigational Data Handling**: Processes and interpolates navigational data like COG and TH, accounting for their circular nature.
- **Geographic and Speed Data Validation**: Validates geographic coordinates, handles outliers, and processes SOG data for consistency.
- **Data Segmentation and Saving**: Splits data by trip identifiers and saves processed segments into separate files.

Duplicates are removed based on `TripID`, `MMSI`, `ID`, and `time`. Data is sorted by `TripID` and `time` to ensure chronological order. The code interpolates `COG` and `TH` to handle circular data, validates geographic coordinates, and processes `SOG` data to identify outliers. The cleaned data is saved into separate CSV files for each trip.

## Installation

### Setting Up the Environment
1. Create and activate a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Application
Run this commant to install PyQt5:
```sh
pip3 install PyQt5
```
This command should run separately from the other requirements to avoid memory allocation issue.

To run the application, use the following command:
```sh
python3 main.py
```

## Running Tests
To run the tests, use the following command:
```sh
pytest tests/
```
## Docker Setup
To build and run the application using Docker, follow these steps:
1. **Build the Docker Image**: This step creates a Docker image containing all the necessary dependencies and the application code.
    ```sh
    docker build -t ship-voyages-app .
    ```
2. **Run the Docker Container**: This step runs the Docker container using the built image, exposing the application on port 8050.
    ```sh
    docker run -d -p 8050:8050 --name ship-voyages-container ship-voyages-app
    ```
After running these commands, the application will be accessible at `http://localhost:8050`. in your web browser.

## Continuous Integration (CI)
The CI pipeline automates the setup, testing, and deployment processes. It performs the following steps:

1. **Setup**: Installs necessary dependencies and tools.
2. **Testing**: Runs automated tests to ensure code quality and functionality.
3. **Deployment**: Deploys the application if all tests pass successfully.

This README provides a comprehensive guide to the Ship Voyages dataset, including data structure, processing, setup instructions, and documentation of the application's functionalities.