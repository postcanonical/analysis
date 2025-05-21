import sys
import os
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import QWidget, QApplication # QApplication is for main.py
from PyQt5.QtCore import Qt

from config import THEMES, DEFAULT_RANDOM_PASTE_OPTIONS, DEFAULT_CALIBRATION
from ui_mixin import UIMixin
from plotting_mixin import PlottingMixin
from data_mixin import DataMixin
from actions_mixin import ActionsMixin

class DataVisualizationTool(QWidget, UIMixin, PlottingMixin, DataMixin, ActionsMixin):
    """
    Main application window for the Data Visualization Tool.
    It orchestrates UI, plotting, data handling, and actions by inheriting from mixins.
    """
    def __init__(self, num_pairs_arg=None, initial_theme_arg="dark", initial_file_path_arg=None, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Advanced Data Visualization Tool")

        # Store initial arguments
        self.predefined_num_pairs = num_pairs_arg  # From command-line or direct instantiation
        self.initial_theme_name = initial_theme_arg
        self.initial_file_to_load = initial_file_path_arg

        # --- Initialize core attributes and configurations ---
        self._load_configurations()
        self._initialize_core_data_variables() # From DataMixin (e.g., self.data, self.copied_data)
        self.current_theme = self.initial_theme_name # Set theme before UI init uses it
        self.is_zooming = False # Flag for preventing xlim_changed recursion


        # --- Initialize UI, Plotting infrastructure, and Actions ---
        # Initialize plot infrastructure first, as UI might depend on num_pairs if dummy data is loaded early
        # However, num_pairs is usually set after data loading. So, init with 0 or a default.
        self.num_pairs = 0 # Default, will be updated
        self.initialize_plot_infrastructure() # From PlottingMixin (sets up empty figure/axes arrays based on self.num_pairs)
        self._resize_phase_amplitude_arrays() # From DataMixin (sets up empty amp/phase arrays)

        self.init_ui_elements()          # From UIMixin (creates all widgets and layouts)
        self.setup_event_shortcuts()     # From ActionsMixin (sets up Ctrl+C, Ctrl+V etc.)
        
        # --- Display the window ---
        self.showMaximized()

    def _load_configurations(self):
        """Loads configurations like themes and default options from config.py."""
        self.themes = dict(THEMES) # Make a copy to prevent modification of original
        self.last_random_paste_options = dict(DEFAULT_RANDOM_PASTE_OPTIONS)
        self.default_calibration_config = dict(DEFAULT_CALIBRATION)


    def handle_initial_startup_tasks(self):
        """
        Called after the window is created and shown (e.g., from main.py).
        Handles initial data loading and plot creation based on arguments.
        """
        if self.initial_file_to_load:
            print(f"Attempting to load initial file: {self.initial_file_to_load}")
            self.load_data_from_file(self.initial_file_to_load) # DataMixin
            
            if not self.data.empty and self.num_pairs > 0:
                self.recreate_all_plots() # PlottingMixin
            else:
                # load_data_from_file or recreate_all_plots would have shown messages
                # If still no data, offer to load or show dummy
                print(f"Failed to load data from '{self.initial_file_to_load}' or no plottable pairs. Offering dialog.")
                self.trigger_load_data_dialog() # DataMixin - this will again try to plot if successful
        else:
            # No initial file provided, open load dialog or load dummy
            print("No initial file path provided via arguments.")
            self.trigger_load_data_dialog() # DataMixin - offers file dialog

        # Final check: if plots were created, ensure theme is applied.
        # recreate_all_plots and trigger_load_data_dialog should handle theme application internally.
        # This is a fallback.
        if self.figures.any(): # figures is a numpy array of objects
            self.apply_theme_to_all_plots() # PlottingMixin
        elif self.data.empty: # If dialog was cancelled and no data loaded
             # Optionally load dummy data as a last resort if configured to do so
             # self.load_initial_dummy_data() # From DataMixin
             # if not self.data.empty: self.recreate_all_plots()
             pass # For now, leave blank if user cancels everything.

        self.update_status_bar_text() # UIMixin

