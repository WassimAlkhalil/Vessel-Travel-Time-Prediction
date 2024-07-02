import os
import sys

# Determine the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the libs directory to the system path
libs_dir = os.path.join(current_dir, 'libs')
sys.path.append(libs_dir)

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QTextBrowser
from PyQt5.QtCore import QThread, pyqtSignal, QUrl, pyqtSlot
from PyQt5.QtGui import QDoubleValidator
import folium
import pandas as pd

from guiLib import Ui_MainWindow
from cleaningLib import cleaning_data, split_and_save_trips_radius_km_based, load_data, save_processed_data
from machineLearningLib import train_models, predict_eta

class Worker(QThread):
    """
    Base Worker class that provides signals for progress and completion.
    """
    progress = pyqtSignal(str)
    finished = pyqtSignal()

class CleaningWorker(Worker):
    """
    Worker class for cleaning data files in a separate thread.
    
    Attributes:
        file_paths (list): List of file paths to be cleaned.
        start_coor (tuple): Starting coordinates for trip filtering.
        end_coor (tuple): Ending coordinates for trip filtering.
        radius (float): Radius for trip filtering.
        cleaned_data (dict): Dictionary to store cleaned data.
        dataset_name (str): Name of the dataset.
    """
    finished = pyqtSignal(pd.DataFrame, dict, str)

    def __init__(self, file_paths, start_coor, end_coor, radius, cleaned_data, dataset_name, parent=None):
        super(CleaningWorker, self).__init__(parent)
        self.file_paths = file_paths
        self.start_coor = start_coor
        self.end_coor = end_coor
        self.radius = radius
        self.cleaned_data = cleaned_data
        self.dataset_name = dataset_name

    def run(self):
        """
        Runs the data cleaning process.
        """
        try:
            all_data = []
            for file_path in self.file_paths:
                self.progress.emit(f"Cleaning The File...")
                df, pth = cleaning_data(file_path, None, saveUndername = self.dataset_name)
                grouped_data = split_and_save_trips_radius_km_based(df, self.start_coor[0], self.start_coor[1], self.end_coor[0], self.end_coor[1], self.radius)

                start_in_end_in_df = grouped_data.get('start_in_end_in', pd.DataFrame())
                all_data.append(start_in_end_in_df)
                self.cleaned_data[file_path] = start_in_end_in_df
                self.progress.emit(f"Cleaning Process Is Completed. Number Of Processed Samples : {len(start_in_end_in_df)}")

            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data['TripID'] = combined_data['TripID'].astype(str)

            save_name = f"{self.dataset_name}_Rad_Filtered_Data"
            cleaned_data_path = save_processed_data(combined_data, save_name)
            self.progress.emit(f"Processed Data Successfully Saved At: {cleaned_data_path}")
            self.finished.emit(combined_data, self.cleaned_data, cleaned_data_path)
        except Exception as e:
            self.progress.emit(f"Error during cleaning: {e}")

class TrainingWorker(Worker):
    """
    Worker class for training models in a separate thread.
    
    Attributes:
        data (pd.DataFrame): Data to be used for training.
        models (list): List of model names to train.
        dataset_name (str): Name of the dataset.
        target_tte (bool): Whether to use total time as the target.
        save_model (bool): Whether to save the trained model.
    """
    finished = pyqtSignal(object, str, object)

    def __init__(self, data, models, dataset_name, target_tte, save_model, parent=None):
        super(TrainingWorker, self).__init__(parent)
        self.data = data
        self.models = models
        self.dataset_name = dataset_name
        self.target_tte = target_tte
        self.save_model = save_model

    def run(self):
        """
        Runs the model training process.
        """
        try:
            self.progress.emit(f"Training The Models {self.models} Started...")
            results, best_model, best_model_instance, stacking_model, stacking_metrics, stacking_path = train_models(self.data, models_names=self.models,
                                                                                                                     dataset=self.dataset_name,
                                                                                                                     total_time=self.target_tte,
                                                                                                                     save_model=self.save_model)
            self.progress.emit(f"Best Model: {best_model['name']}")
            self.progress.emit(f"Mean Absolute Error: {best_model['mae']}")
            self.progress.emit(f"Stacking Model Metrics: {stacking_metrics}")
            self.finished.emit(stacking_model, stacking_path, best_model_instance)
        except Exception as e:
            self.progress.emit(f"Error during training: {e}")

class PredictionWorker(Worker):
    """
    Worker class for making predictions in a separate thread.
    
    Attributes:
        params (dict): Parameters for prediction.
        model_path (str): Path to the model file.
        model_instance (object): Trained model instance.
    """
    finished = pyqtSignal(float)

    def __init__(self, params, model_path, model_instance, parent=None):
        super(PredictionWorker, self).__init__(parent)
        self.params = params
        self.model_path = model_path
        self.model_instance = model_instance

    def run(self):
        """
        Runs the prediction process.
        """
        try:
            self.progress.emit(f"Prediction Process Started...")
            eta = predict_eta(
                latitude=self.params['latitude'],
                longitude=self.params['longitude'],
                time=self.params['time'],
                start_lat=self.params['start_lat'],
                start_lon=self.params['start_lon'],
                start_time=self.params['start_time'],
                end_lat=self.params['end_lat'],
                end_lon=self.params['end_lon'],
                sog=self.params['sog'],
                cog=self.params['cog'],
                direction_change=self.params['direction_change'],
                mean_sog=self.params['mean_sog'],
                time_from_start=self.params['time_from_start'],
                model_path=self.model_path,
                model_instance=self.model_instance
            )
            if eta is not None:
                self.progress.emit(f"Predicted ETA: {eta/3600:.2f} hours")
                self.finished.emit(eta)
            else:
                self.progress.emit("Error: No ETA could be calculated. Please check the model path or the input parameters.")
                self.finished.emit(-1.0)
        except Exception as e:
            self.progress.emit(f"Error in predict_eta: {e}")
            self.finished.emit(-1.0)

class MainWindow(QMainWindow, Ui_MainWindow):
    """
    Main window class for the application.
    """
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        
        # Initialize variables
        self.file_paths = []
        self.data = None
        self.cleaned_data = {}
        self.cleaned_data_path = None
        self.start_coor = None
        self.end_coor = None
        self.current_coor = None
        self.end_ml_model = None
        self.model_path = None
        self.dataset_name = None
        self.dataset_name_trained = None
        self.textBrowser_Log = self.findChild(QTextBrowser, 'textBrowser_Log')

        # Connect UI elements to their respective methods
        self.lineEdit_TCsv.mousePressEvent = lambda event: self.handle_path_click(
            event, self.lineEdit_TCsv, "Select Files for Testing", "CSV Files (*.csv)")
        self.lineEdit_ModelPath.mousePressEvent = lambda event: self.handle_path_click(
            event, self.lineEdit_ModelPath, "Select Model for Estimation", "joblib Files (*.joblib)")
        self.lineEdit_FilePath.mousePressEvent = lambda event: self.handle_path_click(
            event, self.lineEdit_FilePath, "Select Files for Cleaning", "CSV Files (*.csv)")

        self.pushButton_Cleaning.clicked.connect(self.cleaning)
        self.pushButton_Train.clicked.connect(self.training)
        self.pushButton_Predict.clicked.connect(self.predict_time)

        self.comboBox_TripID.currentIndexChanged.connect(self.show_visualization)

        self.lineEdit_FilePath.textChanged.connect(self.update_ui_state)
        self.lineEdit_DataSet.textChanged.connect(self.update_ui_state)
        self.lineEdit_Rad.textChanged.connect(self.update_ui_state)
        self.lineEdit_Lat.textChanged.connect(self.update_ui_state)
        self.lineEdit_Lon.textChanged.connect(self.update_ui_state)
        self.lineEdit_TCsv.textChanged.connect(self.update_ui_state)
        self.lineEdit_ModelPath.textChanged.connect(self.update_ui_state)
        self.checkBox_CBreHam.stateChanged.connect(self.update_ui_state)
        self.checkBox_CKieGdy.stateChanged.connect(self.update_ui_state)
        self.checkBox_Model1.stateChanged.connect(self.update_ui_state)
        self.checkBox_Model2.stateChanged.connect(self.update_ui_state)
        self.checkBox_Model3.stateChanged.connect(self.update_ui_state)
        self.checkBox_Model4.stateChanged.connect(self.update_ui_state)
        self.checkBox_Model5.stateChanged.connect(self.update_ui_state)
        self.checkBox_TTE.stateChanged.connect(self.update_ui_state)
        self.checkBox_TUE.stateChanged.connect(self.update_ui_state)
        self.checkBox_EBreHam.stateChanged.connect(self.update_ui_state)
        self.checkBox_EKieGdy.stateChanged.connect(self.update_ui_state)

        self.pushButton_Cleaning.setEnabled(False)
        self.pushButton_Train.setEnabled(False)
        self.pushButton_Predict.setEnabled(False)
        self.tabWidget_VizualPage1.setEnabled(False)
        self.tabWidget_VizualPage2.setEnabled(False)
        self.tab_Vis.setEnabled(False)

        # Set validators for input fields
        self.set_validators()

        # Initialize UI state
        self.update_ui_state()

    def set_validators(self):
        """
        Set validators for input fields.
        """
        double_validator = QDoubleValidator()
        self.lineEdit_Rad.setValidator(double_validator)
        self.lineEdit_Lat.setValidator(double_validator)
        self.lineEdit_Lon.setValidator(double_validator)

    def handle_path_click(self, event, line_edit, dialog_title, file_filter):
        """
        Handle the file path selection for line edits.
        
        Args:
            event: The mouse event.
            line_edit: The line edit widget.
            dialog_title: The dialog title.
            file_filter: The file filter for the dialog.
        """
        self.select_files(line_edit, dialog_title, file_filter)

    def select_files(self, line_edit, dialog_title, file_filter):
        """
        Open a file dialog to select files and set the selected files to the line edit.
        
        Args:
            line_edit: The line edit widget.
            dialog_title: The dialog title.
            file_filter: The file filter for the dialog.
        """
        line_edit.clear()
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, dialog_title, "", f"{file_filter};;All Files (*)", options=options)
        if files:
            line_edit.setText("; ".join(files))

    def update_ui_state(self):
        """
        Update the state of the UI elements based on the current inputs.
        """
        # Enable/disable the "Cleaning" button
        has_file_path = bool(self.lineEdit_FilePath.text())
        rad_given = bool(self.lineEdit_Rad.text())
        is_direction_checked = self.checkBox_CBreHam.isChecked() ^ self.checkBox_CKieGdy.isChecked()
        self.pushButton_Cleaning.setEnabled(has_file_path and is_direction_checked and rad_given)

        # Enable/disable the training button
        model_checked = self.checkBox_Model1.isChecked() or self.checkBox_Model2.isChecked() or self.checkBox_Model3.isChecked() or self.checkBox_Model4.isChecked() or self.checkBox_Model5.isChecked()
        target_checked = self.checkBox_TTE.isChecked() ^ self.checkBox_TUE.isChecked()
        data_cleaned = bool(self.data is not None and not self.data.empty)
        self.pushButton_Train.setEnabled(model_checked and target_checked and data_cleaned)

        # Enable/disable the prediction button
        given_point = bool(bool(self.lineEdit_Lat.text()) and bool(self.lineEdit_Lon.text()))
        is_direction_checked = self.checkBox_EBreHam.isChecked() ^ self.checkBox_EKieGdy.isChecked()
        point_file = bool(self.lineEdit_TCsv.text())
        model_trained = bool((self.end_ml_model is not None) or (self.model_path is not None) or bool(self.lineEdit_ModelPath.text()))
        self.pushButton_Predict.setEnabled(((given_point and is_direction_checked) or point_file) and model_trained)

        # Enable/disable the visualization tap
        data_cleaned = bool(self.data is not None and not self.data.empty)
        self.tab_Vis.setEnabled(data_cleaned or bool(self.lineEdit_Out.text()))
        # Enable/disable the all-visualization tap (page 1)
        self.tabWidget_VizualPage1.setEnabled(bool(self.lineEdit_Out.text()))
        # Enable/disable the all-visualization tap (page 2)
        self.tabWidget_VizualPage2.setEnabled(data_cleaned)
    
    def cleaning(self):
        """
        Start the data cleaning process.
        """
        file_path = self.lineEdit_FilePath.text()
        if file_path:
            self.file_paths = file_path.split("; ")

        if not self.file_paths:
            QMessageBox.warning(self, "Input Error", "Please select or input the file paths.")
            return

        if not (self.checkBox_CBreHam.isChecked() ^ self.checkBox_CKieGdy.isChecked()):
            QMessageBox.warning(self, "Input Error", "Please select Trip Direction for uploaded file (but not both or none.)")
            return

        try:
            if self.checkBox_CBreHam.isChecked():
                self.start_coor = (53.549999, 8.583333)
                self.end_coor = (53.507097, 9.967923)
                self.dataset_name = 'Bremen_Hamburg'
            if self.checkBox_CKieGdy.isChecked():
                self.start_coor = (54.3179, 10.1389)
                self.end_coor = (54.533831198, 18.570497718)
                self.dataset_name = 'Kiel_Gdynia'

            rad = float(self.lineEdit_Rad.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please ensure all input fields are correctly filled.")
            return

        
        self.pushButton_Cleaning.setEnabled(False)
        self.pushButton_Train.setEnabled(False)
        self.cleaning_worker = CleaningWorker(self.file_paths, self.start_coor, self.end_coor, rad, self.cleaned_data, self.dataset_name)
        self.cleaning_worker.progress.connect(self.update_status)
        self.cleaning_worker.finished.connect(self.cleaning_finished)
        self.cleaning_worker.start()

    def update_status(self, message):
        """
        Update the status messages in the log and status bar.
        
        Args:
            message (str): The status message to display.
        """
        self.textBrowser_Log.append(message)
        self.statusBar().showMessage(message)

    def cleaning_finished(self, combined_data, cleaned_data, cleaned_data_path):
        """
        Callback for when the cleaning process is finished.
        
        Args:
            combined_data (pd.DataFrame): The combined cleaned data.
            cleaned_data (dict): Dictionary of cleaned data.
            cleaned_data_path (str): Path to the saved cleaned data.
        """
        self.data = combined_data
        self.cleaned_data = cleaned_data
        self.cleaned_data_path = cleaned_data_path
        QMessageBox.information(self, "Cleaning Finished", "Data cleaning is complete. You can now see the Trips paths in the Visuilisation Tap!")
        self.update_status("Cleaning is complete.")
        self.populate_trip_ids()
        self.update_ui_state()

    def populate_trip_ids(self):
        """
        Populate the TripID combo box with unique Trip IDs from the cleaned data.
        """
        self.comboBox_TripID.blockSignals(True)
        self.comboBox_TripID.clear()
        self.comboBox_TripID.addItem("Select TripID...")
        unique_trip_ids = self.data['TripID'].unique()
        for trip_id in unique_trip_ids:
            self.comboBox_TripID.addItem(str(trip_id))
        self.comboBox_TripID.blockSignals(False)
        if self.comboBox_TripID.count() > 0:
            self.comboBox_TripID.setCurrentIndex(0)

    def training(self):
        """
        Start the model training process.
        """
        if not (self.checkBox_TTE.isChecked() ^ self.checkBox_TUE.isChecked()):
            QMessageBox.warning(self, "Input Error", "Please select either 'Target: Total-Time (TTE)' or 'Target: Time_Until_End (TUE)', but not both or none.")
            return

        selected_models = []
        if self.checkBox_Model1.isChecked():
            selected_models.append("AdaBoost")
        if self.checkBox_Model2.isChecked():
            selected_models.append("ExtraTrees")
        if self.checkBox_Model3.isChecked():
            selected_models.append("LinearRegression")
        if self.checkBox_Model4.isChecked():
            selected_models.append("GradientBoosting")
        if self.checkBox_Model5.isChecked():
            selected_models.append("RandomForest")

        if len(selected_models) < 2:
            save_model = True
            QMessageBox.warning(self, "Input Warning", "If fewer than two models are selected, the model will not be optimized and only the provided model will be used.")
        else:
            save_model = self.checkBox_Save.isChecked()

        self.dataset_name_trained = self.lineEdit_DataSet.text()
        target_tte = self.checkBox_TTE.isChecked()

        if not self.data.empty:
            data_training = self.data
        else:
            QMessageBox.warning(self, "Data Error", "No cleaned data available for training.")
            return

        self.pushButton_Cleaning.setEnabled(False)
        self.pushButton_Train.setEnabled(False)
        self.training_worker = TrainingWorker(data_training, selected_models, self.dataset_name_trained, target_tte, save_model)
        self.training_worker.progress.connect(self.update_status)
        self.training_worker.finished.connect(self.training_finished)
        self.training_worker.start()

    def training_finished(self, stacking_model, stacking_path, best_model_instance):
        """
        Callback for when the training process is finished.
        
        Args:
            stacking_model (object): The trained stacking model.
            stacking_path (str): Path to the saved stacking model.
            best_model_instance (object): The best model instance.
        """
        if stacking_model:
            self.end_ml_model = stacking_model
            self.model_path = stacking_path
        else:
            self.end_ml_model = best_model_instance
            self.model_path = None

        QMessageBox.information(self, "Training Finished", "Model training is complete.")
        self.update_status("Training completed.")
        self.update_ui_state()

    def predict_time(self):
        """
        Start the prediction process.
        """
        file_path = self.lineEdit_TCsv.text()
        use_file = os.path.exists(file_path)

        if not use_file:
            if not (self.checkBox_EBreHam.isChecked() ^ self.checkBox_EKieGdy.isChecked()):
                QMessageBox.warning(self, "Input Error", "Please select either 'Kiel-Gdynia' or 'Bremerhaven-Hamburg'.")
                return

            if not self.lineEdit_Lat.text() or not self.lineEdit_Lon.text():
                QMessageBox.warning(self, "Input Error", "Please provide both latitude and longitude.")
                return

        if not self.lineEdit_ModelPath.text():
            QMessageBox.information(self, "Model Path Information", "Model path is empty. Prediction will proceed with the newly trained model.")

        pr_model_path = self.lineEdit_ModelPath.text()
        if not self.model_path and pr_model_path:
            self.model_path = pr_model_path

        if use_file:
            data = pd.read_csv(file_path)
            # Mandatory fields
            self.current_coor = (data['Latitude'].iloc[0], data['Longitude'].iloc[0])
            self.start_coor = (data['StartLatitude'].iloc[0], data['StartLongitude'].iloc[0])
            self.end_coor = (data['EndLatitude'].iloc[0], data['EndLongitude'].iloc[0])
            # Optional fields 1
            start_time = pd.to_datetime(data['StartTime'].iloc[0]) if 'StartTime' in data else pd.Timestamp.now()
            time = pd.to_datetime(data['Time'].iloc[0]) if 'Time' in data else pd.Timestamp.now()
            # Optional fields 2
            sog = data['SOG'].iloc[0] if 'SOG' in data else None
            cog = data['COG'].iloc[0] if 'COG' in data else None
            direction_change = data['DirectionChange'].iloc[0] if 'DirectionChange' in data else None
            mean_sog = data['Mean_SOG'].iloc[0] if 'Mean_SOG' in data else 14.0
            time_from_start = data['TimeFromStart'].iloc[0] if 'TimeFromStart' in data else None

            predict_params = {
                'latitude': self.current_coor[0],
                'longitude': self.current_coor[1],
                'time': time,
                'start_lat': self.start_coor[0],
                'start_lon': self.start_coor[1],
                'start_time': start_time,
                'end_lat': self.end_coor[0],
                'end_lon': self.end_coor[1],
                'sog': sog,
                'cog': cog,
                'direction_change': direction_change,
                'mean_sog': mean_sog,
                'time_from_start': time_from_start
            }
        else:
            # Mandatory fields
            if self.checkBox_EBreHam.isChecked():
                self.start_coor = (53.549999, 8.583333)
                self.end_coor = (53.507097, 9.967923)
                self.dataset_name = 'Bremen_Hamburg'
            elif self.checkBox_EKieGdy.isChecked():
                self.start_coor = (54.3179, 10.1389)
                self.end_coor = (54.533831198, 18.570497718)
                self.dataset_name = 'Kiel_Gdynia'

            self.current_coor = (float(self.lineEdit_Lat.text()),
                                 float(self.lineEdit_Lon.text()))
            # Optional fields 1
            start_time = pd.Timestamp.now()
            time = pd.Timestamp.now()

            predict_params = {
                'latitude': self.current_coor[0],
                'longitude': self.current_coor[1],
                'time': time,
                'start_lat': self.start_coor[0],
                'start_lon': self.start_coor[1],
                'start_time': start_time,
                'end_lat': self.end_coor[0],
                'end_lon': self.end_coor[1],
                'sog': None,
                'cog': None,
                'direction_change': None,
                'mean_sog': None,
                'time_from_start': None
            }
            
        self.pushButton_Predict.setEnabled(False)
        self.prediction_worker = PredictionWorker(predict_params, self.model_path, self.end_ml_model)
        self.prediction_worker.progress.connect(self.update_status)
        self.prediction_worker.finished.connect(self.prediction_finished)
        self.prediction_worker.start()

    def format_eta(self, seconds):
        """
        Format ETA from seconds to a human-readable string.
        
        Args:
            seconds (int): Time in seconds.
        
        Returns:
            str: Formatted time string.
        """
        months = seconds // (30 * 24 * 3600)
        seconds %= (30 * 24 * 3600)
        days = seconds // (24 * 3600)
        seconds %= (24 * 3600)
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60

        parts = []
        if months > 0:
            parts.append(f"{months} months")
        if days > 0:
            parts.append(f"{days} days")
        if hours > 0:
            parts.append(f"{hours} hours")
        if minutes > 0:
            parts.append(f"{minutes} minutes")
        if seconds > 0 or not parts:
            parts.append(f"{seconds:.2f} seconds")

        return ", ".join(parts)
    
    def set_eta(self, eta_seconds):
        """
        Set the formatted ETA in the UI.
        
        Args:
            eta_seconds (int): ETA in seconds.
        """
        formatted_eta = self.format_eta(eta_seconds)
        self.lineEdit_Out.setText(formatted_eta)

    def prediction_finished(self, eta):
        """
        Callback for when the prediction process is finished.
        
        Args:
            eta (float): Predicted ETA in seconds.
        """
        self.set_eta(eta)
        QMessageBox.information(self, "Prediction Finished", self.format_eta(eta))
        self.update_status("Prediction completed.")
        self.update_ui_state()
        self.model_path = None
        self.lineEdit_ModelPath.setText(None)
        self.plot_prediction()

    def plot_prediction(self):
        """
        Plot the current location, start, and end points on a map.
        """
        try:
            map_file = 'libs/point_map.html'
            map_point = folium.Map(location=self.current_coor, zoom_start=10)

            folium.CircleMarker(
                location=self.start_coor,
                radius=6,
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.6,
                tooltip="Start Port"
            ).add_to(map_point)

            folium.CircleMarker(
                location=self.current_coor,
                radius=6,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.6,
                tooltip="Current Location"
            ).add_to(map_point)

            folium.CircleMarker(
                location=self.end_coor,
                radius=6,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.6,
                tooltip="End Port"
            ).add_to(map_point)

            map_point.save(map_file)

            self.webEngineView_Point.setUrl(QUrl.fromLocalFile(os.path.abspath(map_file)))
        except Exception as e:
            self.update_status(f"Error Plotting Point: {e}")

    @pyqtSlot(int)
    def show_visualization(self, index):
        """
        Show visualization for the selected Trip ID.
        
        Args:
            index (int): Index of the selected Trip ID in the combo box.
        """
        trip_id = self.comboBox_TripID.itemText(index)
        if trip_id:
            try:
                trip_data = self.data[self.data['TripID'] == trip_id]
                if trip_data.empty:
                    QMessageBox.warning(self, "Visualization Error", f"No data available for the selected trip ID: {trip_id}.")
                    return
                self.plot_cleaned_data(trip_data)
                self.populate_trip_details(trip_data)
            except Exception as e:
                self.update_status(f"Error filtering data for Trip ID {trip_id}: {e}")
        else:
            QMessageBox.warning(self, "Visualization Error", "No data available for the selected trip ID.")

    def plot_cleaned_data(self, trip_data):
        """
        Plot the cleaned trip data on a map.
        
        Args:
            trip_data (pd.DataFrame): DataFrame containing trip data.
        """
        if trip_data.empty:
            QMessageBox.warning(self, "Plot Error", "No data available to plot.")
            return

        required_columns = ['time', 'SOG', 'COG', 'Latitude', 'Longitude']
        if not all(column in trip_data.columns for column in required_columns):
            QMessageBox.warning(self, "Data Error", "Missing required columns in trip data.")
            return

        map_file = 'libs/trip_map.html'
        map_trip = folium.Map(location=[trip_data['Latitude'].iloc[0], trip_data['Longitude'].iloc[0]], zoom_start=10)

        try:
            folium.CircleMarker(
                location=self.start_coor,
                radius=5,
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.6,
                tooltip="Start"
            ).add_to(map_trip)

            folium.CircleMarker(
                location=self.end_coor,
                radius=5,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.6,
                tooltip="End"
            ).add_to(map_trip)

            for _, row in trip_data.iterrows():
                details = (
                    f"Time: {row['time']}<br>"
                    f"SOG: {row['SOG']}<br>"
                    f"COG: {row['COG']}<br>"
                    f"Lat: {row['Latitude']}<br>"
                    f"Lon: {row['Longitude']}"
                )
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=2,
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    fill_opacity=0.6,
                    tooltip=folium.Tooltip(details, sticky=True)
                ).add_to(map_trip)

            folium.PolyLine(
                locations=list(zip(trip_data['Latitude'], trip_data['Longitude'])),
                color="blue"
            ).add_to(map_trip)

            map_trip.save(map_file)

            self.webEngineView_All.setUrl(QUrl.fromLocalFile(os.path.abspath(map_file)))
        except Exception as e:
            self.update_status(f"Error creating PolyLine: {e}")

    def populate_trip_details(self, trip_data):
        """
        Populate trip details in the UI.
        
        Args:
            trip_data (pd.DataFrame): DataFrame containing trip data.
        """
        if trip_data.empty:
            QMessageBox.warning(self, "Detail Error", "No data available to show details.")
            return
        
        self.lineEdit_TripID.setText(str(trip_data['TripID'].iloc[0]))
        self.lineEdit_StartPort.setText(str(trip_data['StartPort'].iloc[0]))
        self.lineEdit_EndPort.setText(str(trip_data['EndPort'].iloc[0]))
        self.lineEdit_MMSI.setText(str(trip_data['MMSI'].iloc[0]))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
