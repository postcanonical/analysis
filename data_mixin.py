import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QInputDialog, QMessageBox, QFileDialog

class DataMixin:
    def _initialize_core_data_variables(self):
        """Initializes core data structures and state variables."""
        self.data = pd.DataFrame()
        self.copied_data = pd.DataFrame()
        self.paste_count = 0
        self.calibration = dict(self.default_calibration_config) # from config
        self.is_calibrated = False # Or load from a config/setting
        
        # These will be resized/filled when data is loaded or num_pairs changes
        self.amplitude = np.array([]) 
        self.phase = np.array([])
        
        # self.num_pairs will be determined by loaded data or command-line arg
        # self.current_plot_prefixes will be populated based on loaded data columns
        self.current_plot_prefixes = []
        # self.plot_scatter_grouped = np.empty((1, 2), dtype=object) # Example if still needed, ensure size matches num_pairs

        self.current_file_path = None # Store path of loaded file


    def load_data_from_file(self, file_path):
        """Loads data from the specified file path and prepares for plotting."""
        try:
            loaded_df = pd.DataFrame()
            if file_path.lower().endswith('.csv'):
                loaded_df = pd.read_csv(file_path)
            elif file_path.lower().endswith('.dep'):
                n_channels, ok = QInputDialog.getInt(self, "DEP File Loader", "Number of channel pairs:", value=8, min=1)
                if not ok: return # User cancelled
                math_n_channels, ok2 = QInputDialog.getInt(self, "DEP File Loader", "Number of math channels (derived from pair 0):", value=0, min=0)
                if not ok2: return
                loaded_df = self._read_dep_to_dataframe(file_path, n_channels, math_n_channels)
            else: # Try as space/tab separated
                loaded_df = pd.read_csv(file_path, sep=r'\s+', header=None)
                # Check if columns need to be named for X/Y pairs
                if loaded_df.shape[1] % 2 != 0:
                    QMessageBox.warning(self, "Load Error", "Loaded data has an odd number of columns. Cannot form X/Y pairs.")
                    return # Invalid structure
                # Auto-name columns like 0_X, 0_Y, 1_X, 1_Y, ...
                loaded_df.columns = [f"{i//2}_{'X' if i%2==0 else 'Y'}" for i in range(loaded_df.shape[1])]

            if loaded_df.empty:
                # QMessageBox.information(self, "Load Info", "No data loaded from file or format not supported for automatic parsing.")
                self.data = pd.DataFrame() # Ensure self.data is empty
                self.num_pairs = 0
                self.current_plot_prefixes = []
                self.clear_all_plots() # From PlottingMixin
                self.initialize_plot_infrastructure() # Re-init with 0 pairs
                self._resize_phase_amplitude_arrays()
                self.update_status_bar_text()
                return

            self.data = loaded_df
            self.current_file_path = file_path # Store path of successfully loaded file

            # Determine plottable pairs from column names (e.g., "0_X", "0_Y", "math0_X", "math0_Y")
            raw_prefixes = sorted(list(set(c.split('_')[0] for c in self.data.columns if c.split('_')[0].isdigit() and c.endswith(('_X', '_Y')))), key=int)
            math_prefixes = sorted(list(set(c.split('_')[0] for c in self.data.columns if c.startswith("math") and c.endswith(('_X', '_Y')))))
            
            valid_prefixes = []
            for p_set in [raw_prefixes, math_prefixes]:
                for p in p_set:
                    if f"{p}_X" in self.data.columns and f"{p}_Y" in self.data.columns:
                        valid_prefixes.append(p)
            
            self.current_plot_prefixes = valid_prefixes

            if self.predefined_num_pairs is not None:
                self.num_pairs = min(self.predefined_num_pairs, len(self.current_plot_prefixes))
                self.current_plot_prefixes = self.current_plot_prefixes[:self.num_pairs]
            else:
                self.num_pairs = len(self.current_plot_prefixes)

            if self.num_pairs == 0:
                if not self.data.empty : # Data was loaded but no valid pairs found
                     QMessageBox.warning(self, "Load Info", "Data loaded, but no valid X/Y column pairs (e.g., 0_X, 0_Y) found to plot.")
                self.data = pd.DataFrame() # Treat as no usable data
                self.clear_all_plots() # From PlottingMixin
            
            # Initialize plot infrastructure and phase/amplitude arrays based on num_pairs
            self.initialize_plot_infrastructure() # From PlottingMixin
            self._resize_phase_amplitude_arrays()

            # Reset states related to previous data
            self.copied_data = pd.DataFrame()
            self.paste_count = 0
            self.update_status_bar_text() # From UIMixin
            
            QMessageBox.information(self, "Load Success", f"Successfully loaded data from: {os.path.basename(file_path)}\nFound {self.num_pairs} plottable pairs.")

        except Exception as e:
            QMessageBox.critical(self, "Load Data Error", f"Failed to load data file '{os.path.basename(file_path)}':\n{str(e)}")
            self.data = pd.DataFrame()
            self.num_pairs = 0
            self.current_plot_prefixes = []
            self.clear_all_plots() # From PlottingMixin
            self.initialize_plot_infrastructure() # Re-init with 0 pairs
            self._resize_phase_amplitude_arrays()
            self.update_status_bar_text() # From UIMixin

    def _read_dep_to_dataframe(self, file_path, n_channel_pairs, n_math_channels=0):
        """Reads a .dep binary file into a pandas DataFrame."""
        try:
            filesize = os.path.getsize(file_path)
            header_size = 1024  # Standard DEP header size
            bytes_per_value = 2  # int16
            values_per_sample_per_pair = 2  # X and Y
            
            total_signal_channels = n_channel_pairs * values_per_sample_per_pair
            bytes_per_sample_all_pairs = total_signal_channels * bytes_per_value

            if filesize <= header_size:
                raise ValueError("File size is too small (no data beyond header).")
            if (filesize - header_size) % bytes_per_sample_all_pairs != 0:
                raise ValueError(f"Data segment size is not a multiple of expected sample structure for {n_channel_pairs} pairs. "
                                 f"Data bytes: {filesize - header_size}, Bytes per sample block: {bytes_per_sample_all_pairs}")

            num_samples = (filesize - header_size) // bytes_per_sample_all_pairs
            if num_samples == 0:
                raise ValueError("File contains header but no data samples.")

            with open(file_path, 'rb') as f:
                f.seek(header_size)
                # Read all data for all channel pairs at once
                raw_data = np.fromfile(f, dtype='<i2', count=num_samples * total_signal_channels)
            
            # Reshape so each row is a sample, and columns are X0, Y0, X1, Y1, ...
            reshaped_data = raw_data.reshape(num_samples, total_signal_channels)
            
            data_dict = {}
            for i in range(n_channel_pairs):
                data_dict[f"{i}_X"] = reshaped_data[:, 2*i].astype(float)
                data_dict[f"{i}_Y"] = reshaped_data[:, 2*i + 1].astype(float)
            
            df = pd.DataFrame(data_dict)

            # Add math channels if specified, based on channel 0
            if n_math_channels > 0 and "0_X" in df.columns and "0_Y" in df.columns:
                base_x_col = df["0_X"].values
                base_y_col = df["0_Y"].values
                for j in range(n_math_channels):
                    # Example math: simple difference (adjust as per actual DEP spec for math channels)
                    # This is a placeholder for actual math channel calculation logic
                    math_x = np.zeros(num_samples, dtype=float)
                    math_y = np.zeros(num_samples, dtype=float)
                    if num_samples > 11: # Example: ensure enough data for a centered difference
                        diff_x = (base_x_col[11:] - base_x_col[:-11]) / 5.0 # Scaled difference
                        diff_y = (base_y_col[11:] - base_y_col[:-11]) / 5.0
                        # Pad the result to match original length, e.g., by repeating first/last or zero-padding
                        # Simple padding for this example: place result in middle
                        start_offset = 5 
                        end_offset = start_offset + len(diff_x)
                        math_x[start_offset:end_offset] = diff_x
                        math_y[start_offset:end_offset] = diff_y
                    
                    df[f"math{j}_X"] = math_x
                    df[f"math{j}_Y"] = math_y
            return df

        except ValueError as ve:
            QMessageBox.critical(self, "DEP Load Error", f"Error reading DEP file '{os.path.basename(file_path)}':\n{str(ve)}")
            return pd.DataFrame()
        except Exception as e:
            QMessageBox.critical(self, "DEP Load Error", f"An unexpected error occurred while reading DEP file:\n{str(e)}")
            return pd.DataFrame()


    def trigger_load_data_dialog(self):
        """Opens a file dialog to load data and then processes it."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", 
                                                   "Data Files (*.csv *.dep *.dat *.txt);;All Files (*)")
        if file_path:
            # self.clear_all_plots() # From PlottingMixin - ensure clean state before loading
            self.load_data_from_file(file_path) # This handles internal state updates
            
            if not self.data.empty and self.num_pairs > 0:
                self.recreate_all_plots() # From PlottingMixin - creates new plots
            else:
                # If data loading failed or no pairs, load_data_from_file handles messages
                # and ensures plots are cleared.
                 QMessageBox.information(self, "Load Data", "No data loaded or no plottable pairs found. Plots will be cleared.")
                 self.clear_all_plots() 
                 self.initialize_plot_infrastructure() # Reset plot structures for 0 pairs
                 self._resize_phase_amplitude_arrays() # Reset phase/amp arrays
                 self.update_status_bar_text()

        else: # User cancelled dialog
            QMessageBox.information(self, "Load Data", "File loading was cancelled by the user.")
            # Optionally, if no data is currently loaded, load dummy data or clear plots
            if self.data.empty:
                print("No file chosen and no existing data. Clearing plots / loading dummy if configured.")
                # self.load_initial_dummy_data() # If you want dummy data on cancel
                self.clear_all_plots() 
                self.num_pairs = 0
                self.current_plot_prefixes = []
                self.initialize_plot_infrastructure()
                self._resize_phase_amplitude_arrays()
                self.update_status_bar_text()
    
    def load_initial_dummy_data(self):
        """Loads dummy data if no file is provided initially."""
        if not self.data.empty: # Don't load dummy if data already exists
            return

        print("Loading dummy data for initial display.")
        n_samples = 1000
        # Determine num_pairs: use predefined if available, else default to 2 for dummy
        self.num_pairs = self.predefined_num_pairs if self.predefined_num_pairs is not None and self.predefined_num_pairs > 0 else 2
        
        self.initialize_plot_infrastructure() # Setup plot structures based on num_pairs
        self._resize_phase_amplitude_arrays() # Setup phase/amp arrays

        dummy_dict = {}
        self.current_plot_prefixes = [str(i) for i in range(self.num_pairs)]

        for i in range(self.num_pairs):
            prefix = self.current_plot_prefixes[i]
            dummy_dict[f'{prefix}_X'] = np.random.randn(n_samples).cumsum()
            dummy_dict[f'{prefix}_Y'] = np.random.randn(n_samples).cumsum()
            
            # Simulate defect probabilities
            probs = np.random.rand(n_samples) * 0.2 # Mostly low probability
            # Add some regions with higher probability
            if n_samples > 200:
                probs[n_samples//10 : 2*n_samples//10] = np.clip(np.random.rand(n_samples//10) * 0.8, 0, 1) 
                probs[n_samples//2 : n_samples//2 + n_samples//20] = np.clip(np.random.rand(n_samples//20) * 1.0, 0, 1)
            dummy_dict[f'defect_proba_{prefix}'] = probs
            
        self.data = pd.DataFrame(dummy_dict)
        
        # Reset other relevant states
        self.copied_data = pd.DataFrame()
        self.paste_count = 0
        self.update_status_bar_text()
        
        QMessageBox.information(self, "Dummy Data Loaded", f"{self.num_pairs} pairs of dummy data loaded.")


    def trigger_save_data(self, dataframe_to_save):
        if dataframe_to_save is None or dataframe_to_save.empty:
            QMessageBox.warning(self, "Save Data", "There is no data to save.")
            return

        default_filename = "exported_data.csv"
        if self.current_file_path: # If a file was loaded, suggest a modified name
            base, ext = os.path.splitext(os.path.basename(self.current_file_path))
            default_filename = f"{base}_modified.csv"

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Data As", default_filename,
                                                   "CSV Files (*.csv);;All Files (*)")
        if file_path:
            try:
                dataframe_to_save.to_csv(file_path, index=True) # Save with index
                QMessageBox.information(self, "Save Success", f"Data successfully saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Data Error", f"Failed to save data:\n{str(e)}")

    def _resize_phase_amplitude_arrays(self):
        """Resizes phase and amplitude arrays based on self.num_pairs and fills with NaN."""
        if self.num_pairs < 0: self.num_pairs = 0 # Ensure non-negative
        self.amplitude = np.full(self.num_pairs, np.nan)
        self.phase = np.full(self.num_pairs, np.nan)

