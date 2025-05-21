import sys
import argparse
import os
from PyQt5.QtWidgets import QApplication
from visualization_tool import DataVisualizationTool

if __name__ == '__main__':
    app = QApplication(sys.argv)

    parser = argparse.ArgumentParser(description="Advanced Data Visualization Tool")
    parser.add_argument(
        '-n', '--num_pairs', 
        type=int, 
        default=None,
        help="Predefined number of data column pairs to display. Overrides auto-detection up to available pairs."
    )
    parser.add_argument(
        '-t', '--theme', 
        type=str, 
        default='dark', 
        choices=['dark', 'light', 'blue', 'solarized'], # Should match keys in config.THEMES
        help="Initial plot theme."
    )
    parser.add_argument(
        'file_path', 
        nargs='?', # Makes the argument optional
        default=None, 
        help="Path to the data file to load on startup (e.g., CSV, DEP)."
    )
    args = parser.parse_args()

    # Instantiate the main window with command-line arguments
    # These arguments are primarily for initial setup.
    # The window itself will handle loading data if file_path is provided.
    main_window = DataVisualizationTool(
        num_pairs_arg=args.num_pairs,
        initial_theme_arg=args.theme,
        initial_file_path_arg=args.file_path
    )
    
    # The DataVisualizationTool's __init__ sets up the UI.
    # A separate method is called to handle data loading and initial plot creation.
    main_window.handle_initial_startup_tasks()
    
    # The window is already shown maximized by its __init__
    # main_window.show() 

    sys.exit(app.exec_())

