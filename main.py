import os
import sys
import random
import pandas as pd
import numpy as np
import argparse  # New: for parsing command-line options

from PyQt5.QtWidgets import (
    QApplication, QWidget, QGridLayout, QVBoxLayout, QHBoxLayout,
    QPushButton, QInputDialog, QSpacerItem, QSizePolicy, QLabel,
    QLineEdit, QComboBox, QFileDialog, QMessageBox, QGroupBox, QShortcut
)
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from phase import get_amp_phase  # Ensure this module is available
import matplotlib.pyplot as plt  # For isinstance checks (if needed)


class DataVisualizationTool(QWidget):
    """
    A PyQt5-based data visualization tool that supports:
      - Loading data (CSV or space-separated).
      - Displaying column pairs in multiple plot types.
      - Selecting data ranges (via SpanSelector or precise indices).
      - Copying/pasting data (including randomizing paste positions).
      - Computing amplitude and phase via an external function (get_amp_phase).
    """
    def __init__(self, num_pairs=None):  # New: Accept an optional predefined number of pairs
        super().__init__()
        self.predefined_num_pairs = num_pairs  # Store predefined num_pairs if provided
        self.setWindowTitle("Data Visualization Tool")

        # 1) Variable & UI initialization
        self.init_variables()
        self.init_ui()

        # 2) Load initial data & create plots
        self.load_initial_data()

        # 3) Keyboard shortcuts
        self.setup_shortcuts()

        # 4) Show the main window maximized
        self.showMaximized()

    # -------------------------------------------------------------------------
    # Initialization & Setup
    # -------------------------------------------------------------------------
    def init_variables(self):
        """Initialize variables used throughout the class."""
        self.copied_data = pd.DataFrame()
        self.paste_count = 0
        self.calibration = {'voltage': 5.0, 'amplitude': 1.0}
        self.is_calibrated = False
        self.amplitude = []
        self.phase = []
        self.is_zooming = False

        # Will be updated once data is loaded
        self.plot_scatter_grouped = np.empty((1, 2), dtype=object)
        self.selection_lines = []  # For storing vertical lines in grouped plots

    def init_ui(self):
        """Initialize the user interface components."""
        # Main layout
        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)

        # Plot area on the left
        self.init_plot_area()

        # Button panel on the right
        self.init_button_panel()

        # Add widgets to the main layout
        self.main_layout.addWidget(self.plot_widget)
        self.main_layout.addWidget(self.button_widget)

    def init_plot_area(self):
        """Initialize the plot area on the left side."""
        self.plot_widget = QWidget()
        self.plot_layout = QGridLayout()
        self.plot_widget.setLayout(self.plot_layout)

    def init_button_panel(self):
        """Initialize the button panel on the right side."""
        self.button_layout = QVBoxLayout()
        self.button_widget = QGroupBox("Menu")
        self.button_widget.setLayout(self.button_layout)
        self.button_widget.setFixedWidth(300)

        # Main buttons
        self.add_main_buttons()
        # Sub-button placeholder (hidden by default)
        self.add_sub_button_placeholder()
        # Status bar
        self.add_status_bar()
        # Spacer
        self.add_spacer()
        # Connect main buttons
        self.connect_main_buttons()

    def add_main_buttons(self):
        """Add main buttons to the button panel."""
        self.data_interference_button = QPushButton('Data Interference')
        self.precise_selection_button = QPushButton('Precise Data Selection')
        self.save_data_button = QPushButton('Save Data to File')
        self.load_data_button = QPushButton('Load Data')

        for button in [
            self.data_interference_button,
            self.precise_selection_button,
            self.save_data_button,
            self.load_data_button
        ]:
            self.button_layout.addWidget(button)

    def add_sub_button_placeholder(self):
        """Add a placeholder group box for sub-buttons."""
        self.sub_button_layout = QVBoxLayout()
        self.sub_button_widget = QGroupBox()
        self.sub_button_widget.setLayout(self.sub_button_layout)
        self.sub_button_widget.hide()  # Hidden by default
        self.button_layout.addWidget(self.sub_button_widget)

    def add_status_bar(self):
        """Add a status bar to display phase information."""
        self.status_bar = QLabel("Phases: N/A")
        self.status_bar.setStyleSheet("background-color: #333; color: white; padding: 5px;")
        self.status_bar.setAlignment(Qt.AlignLeft)
        self.status_bar.setWordWrap(True)
        self.button_layout.addWidget(self.status_bar)

    def add_spacer(self):
        """Add a vertical spacer to push elements to the top."""
        self.button_layout.addSpacerItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

    def connect_main_buttons(self):
        """Connect main buttons to their respective functions."""
        self.data_interference_button.clicked.connect(self.show_data_interference_buttons)
        self.save_data_button.clicked.connect(self.show_save_data_buttons)
        self.precise_selection_button.clicked.connect(self.show_precise_selection_fields)
        self.load_data_button.clicked.connect(self.load_data)

    def setup_shortcuts(self):
        """Set up keyboard shortcuts for copy/paste functionality."""
        QShortcut(QKeySequence("Ctrl+C"), self).activated.connect(self.copy_data)
        QShortcut(QKeySequence("Ctrl+V"), self).activated.connect(self.paste_data)
        QShortcut(QKeySequence("Ctrl+Shift+V"), self).activated.connect(self.show_randomize_paste_options)

    # -------------------------------------------------------------------------
    # Data Loading & Initialization
    # -------------------------------------------------------------------------
    def load_initial_data(self):
        """Load initial data from the default file path."""
        self.load_data()

    def read_dep_to_df(self, file_path, n_channels, math_n_channels=0):
        import os, numpy as np, pandas as pd

        filesize = os.path.getsize(file_path)
        if filesize <= 1024 or ((filesize - 1024) % (4 * n_channels)) != 0:
            raise ValueError(f"Invalid DEP for {n_channels} channels")

        n_samples = (filesize - 1024) // (4 * n_channels)
        with open(file_path, 'rb') as f:
            f.seek(1024)
            raw = np.fromfile(f, dtype='<i2', count=2 * n_channels * n_samples)

        raw = raw.reshape(n_samples, 2 * n_channels)
        data = {}

        # Raw channels: "0_X", "0_Y", "1_X", "1_Y", ...
        for ch in range(n_channels):
            x_col = f"{ch}_X"
            y_col = f"{ch}_Y"
            data[x_col] = raw[:, 2 * ch].astype(float)
            data[y_col] = raw[:, 2 * ch + 1].astype(float)

        # Math channels: "math0_X", "math0_Y", ...
        if math_n_channels > 0:
            base_X = data["0_X"]
            base_Y = data["0_Y"]
            for j in range(math_n_channels):
                mx = np.zeros(n_samples, dtype=float)
                my = np.zeros(n_samples, dtype=float)
                diffs_X = (base_X[11:] - base_X[:-11]) / 5.0
                diffs_Y = (base_Y[11:] - base_Y[:-11]) / 5.0
                mx[6 : n_samples - 5] = diffs_X
                my[6 : n_samples - 5] = diffs_Y
                data[f"math{j}_X"] = mx
                data[f"math{j}_Y"] = my

        return pd.DataFrame(data)

    def load_data_file(self, file_path):
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
                print(f"Loaded CSV data from {file_path}")

            elif file_path.lower().endswith('.dep'):
                n_ch, ok = QInputDialog.getInt(
                    self, "DEP Loader", "Number of raw channels:", value=8, min=1
                )
                if not ok:
                    return
                m_ch, ok2 = QInputDialog.getInt(
                    self, "DEP Loader", "Number of math channels:", value=0, min=0
                )
                if not ok2:
                    return
                self.data = self.read_dep_to_df(file_path, n_ch, m_ch)
                print(f"Loaded DEP data with {n_ch} raw and {m_ch} math channels")

            else:
                self.data = pd.read_csv(file_path, sep=r'\s+', header=None)
                num_cols = self.data.shape[1]
                columns = []
                for i in range(num_cols):
                    channel = i // 2
                    suffix = 'X' if i % 2 == 0 else 'Y'
                    columns.append(f"{channel}_{suffix}")
                self.data.columns = columns
                self.data = self.data.iloc[::-1].reset_index(drop=True)
                print(f"Loaded space-separated data and renamed columns")

            raw_prefixes = set()
            for col in self.data.columns:
                parts = col.split('_')
                if len(parts) == 2 and parts[1] in ['X', 'Y'] and parts[0].isdigit():
                    raw_prefixes.add(parts[0])
            computed_pairs = len(raw_prefixes)

            if self.predefined_num_pairs is not None:
                self.num_pairs = min(self.predefined_num_pairs, computed_pairs)
            else:
                self.num_pairs = computed_pairs

            self.plot_scatter_grouped = np.empty((self.num_pairs, 2), dtype=object)
            self.selection_lines = [[[] for _ in range(2)] for _ in range(self.num_pairs)]

        except Exception as e:
            QMessageBox.critical(self, "Error loading data", str(e))
            return

    def initialize_plot_arrays(self):
        """Initialize arrays for figures, canvases, axes, and plot lines."""
        self.figures = np.empty((self.num_pairs, 2), dtype=object)
        self.canvases = np.empty((self.num_pairs, 2), dtype=object)
        self.axes_grouped = [[] for _ in range(self.num_pairs)]
        self.axes_colcol = [[] for _ in range(self.num_pairs)]
        self.span_selectors = np.empty((self.num_pairs, 2), dtype=object)
        self.plot_lines_grouped = np.empty((self.num_pairs, 2), dtype=object)
        self.plot_lines_colcol = np.empty((self.num_pairs, 1), dtype=object)
        self.amplitude = np.zeros(self.num_pairs)
        self.phase = np.zeros(self.num_pairs)

    # -------------------------------------------------------------------------
    # Plot Creation
    # -------------------------------------------------------------------------
    def create_plots(self):
        """Create all plots for each pair of columns."""
        for row in range(self.num_pairs):
            self.create_individual_plots(row)

    def create_individual_plots(self, row):
        """
        Create individual plots for a given row (i.e., for a pair of columns):
          1) Two subplots grouped (x_col vs index, y_col vs index)
          2) One subplot (x_col vs y_col)
        """
        x_col = f"{row}_X"
        y_col = f"{row}_Y"


        # -- 1) GROUPED PLOTS ------------------------------------------------
        fig_grouped, canvas_grouped = self.create_grouped_plot(row)
        self.figures[row, 0], self.canvases[row, 0] = fig_grouped, canvas_grouped

        # Plot x_col vs index
        ax1, line1 = self.plot_column_vs_index(
            fig=fig_grouped,
            data_column=self.data[x_col],
            label=f'Column {x_col}',
            row=row,
            subplot_index=0
        )

        # Plot y_col vs index
        ax2, line2 = self.plot_column_vs_index(
            fig=fig_grouped,
            data_column=self.data[y_col],
            label=f'Column {y_col}',
            row=row,
            subplot_index=1,
            sharex=ax1
        )

        # Store references
        self.plot_lines_grouped[row, 0] = line1
        self.plot_lines_grouped[row, 1] = line2
        self.axes_grouped[row].extend([ax1, ax2])

        # Scatter for defect probability columns
        prob_col_name = f'defect_proba_{row + 1}'
        if prob_col_name in self.data.columns:
            print(f"Found {prob_col_name} for pair ({x_col}, {y_col})")
            scatter1 = ax1.scatter(
                self.data.index,
                self.data[x_col],
                c=self.data[prob_col_name],
                cmap='bwr',
                vmin=0,
                vmax=1,
                s=10,
                edgecolors='none',
                alpha=0.6
            )
            scatter2 = ax2.scatter(
                self.data.index,
                self.data[y_col],
                c=self.data[prob_col_name],
                cmap='bwr',
                vmin=0,
                vmax=1,
                s=10,
                edgecolors='none',
                alpha=0.6
            )
            self.plot_scatter_grouped[row, 0] = scatter1
            self.plot_scatter_grouped[row, 1] = scatter2
        else:
            print(f"No {prob_col_name} found for pair ({x_col}, {y_col})")
            self.plot_scatter_grouped[row, 0] = None
            self.plot_scatter_grouped[row, 1] = None

        # Create SpanSelectors
        self.span_selectors[row, 0] = self.create_span_selector(ax1)
        self.span_selectors[row, 1] = self.create_span_selector(ax2)

        # Synchronize axes
        self.synchronize_axes(ax1, ax2)

        # -- 2) x_col vs y_col PLOT -------------------------------------------
        fig_colcol, canvas_colcol = self.create_plot_with_toolbar(row)
        self.figures[row, 1], self.canvases[row, 1] = fig_colcol, canvas_colcol

        ax_colcol, line_colcol = self.plot_column_vs_column(
            fig_colcol,
            self.data[x_col],
            self.data[y_col],
            x_col,
            y_col
        )
        self.plot_lines_colcol[row, 0] = line_colcol
        self.axes_colcol[row].append(ax_colcol)

    def create_grouped_plot(self, row):
        """Create a grouped plot widget with two subplots (col vs index)."""
        fig = Figure(facecolor='black', constrained_layout=True)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self)

        self.plot_layout.addWidget(toolbar, row * 2, 0)
        self.plot_layout.addWidget(canvas, row * 2 + 1, 0)
        return fig, canvas

    def create_plot_with_toolbar(self, row):
        """Create a single subplot widget for col vs col with a navigation toolbar."""
        fig = Figure(facecolor='black', constrained_layout=True)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self)

        self.plot_layout.addWidget(toolbar, row * 2, 1)
        self.plot_layout.addWidget(canvas, row * 2 + 1, 1)
        return fig, canvas

    def plot_column_vs_index(self, fig, data_column, label, row, subplot_index, sharex=None):
        """
        Plot a data column against its index.
        :param fig: Figure to draw on
        :param data_column: The Pandas Series to plot
        :param label: Label for the plot
        :param row: Row index (which column pair we are on)
        :param subplot_index: 0 or 1 (top or bottom in the grouped plot)
        :param sharex: Axis object to share x-axis with, if any
        :return: (axis, line) tuple
        """
        if sharex:
            ax = fig.add_subplot(211 + subplot_index, sharex=sharex)
        else:
            ax = fig.add_subplot(211 + subplot_index)

        line, = ax.plot(
            self.data.index,
            data_column,
            label=label,
            color='white',
            linewidth=1
        )

        self.setup_plot(ax, title=label if subplot_index == 0 else '', xlabel='', ylabel='Voltage')
        if subplot_index == 0:
            ax.tick_params(labelbottom=False, bottom=False)
        self.set_dark_theme(ax)
        return ax, line

    def plot_column_vs_column(self, fig, x_data, y_data, x_col, y_col):
        """
        Plot one data column (x_data) vs another data column (y_data).
        :return: (axis, line) tuple
        """
        ax = fig.add_subplot(111)
        line, = ax.plot(
            x_data,
            y_data,
            'g',
            label=f'Column {x_col} vs Column {y_col}'
        )
        self.setup_plot(ax, f'Column {x_col} vs Column {y_col}', f'Column {x_col}', f'Column {y_col}')
        self.set_dark_theme(ax)
        return ax, line

    def setup_plot(self, ax, title, xlabel='', ylabel=''):
        """Setup plot aesthetics for a given axis."""
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(color='gray', linestyle='--', linewidth=0.5)

    def set_dark_theme(self, ax):
        """Apply a dark theme to the provided axis."""
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    def create_span_selector(self, ax):
        """Create a SpanSelector for horizontal selection in a given axis."""
        return SpanSelector(
            ax,
            self.on_select,
            'horizontal',
            useblit=False,
            props=dict(alpha=0.5, facecolor='red')
        )

    def synchronize_axes(self, ax1, ax2):
        """
        Synchronize x-limits and y-limits between two axes.
        Connect callbacks that will be triggered upon limit changes.
        """
        ax1.callbacks.connect('xlim_changed', self.on_xlim_changed_grouped)
        ax1.callbacks.connect('ylim_changed', self.on_ylim_changed_grouped)
        ax2.callbacks.connect('xlim_changed', self.on_xlim_changed_grouped)
        ax2.callbacks.connect('ylim_changed', self.on_ylim_changed_grouped)

    # -------------------------------------------------------------------------
    # Data Selection & SpanSelector Event
    # -------------------------------------------------------------------------
    def on_select(self, xmin, xmax):
        """
        Handle data selection via SpanSelector.
        Mark the selected range, store copied_data, and compute phases.
        """
        if not isinstance(xmin, (int, float)) or not isinstance(xmax, (int, float)):
            print("Invalid selection")
            return

        xmin, xmax = int(np.round(xmin)), int(np.round(xmax))
        xmin, xmax = max(0, xmin), min(len(self.data) - 1, xmax)
        idx = (self.data.index >= xmin) & (self.data.index <= xmax)
        print(f"Selection from {xmin} to {xmax}, {idx.sum()} entries selected.")

        # Store selected slice
        self.copied_data = self.data.loc[idx].copy()

        # Update selection in the plots
        for row in range(self.num_pairs):
            self.update_selection_plots(row, idx, xmin, xmax)

        # Compute phases for the selected indices
        selected_indices = self.data.index[idx]
        self.update_phases(selected_indices)

    def update_selection_plots(self, row, idx, xmin, xmax):
        """
        Update plots based on the selected range for a given row.
        This includes:
          - Updating col vs col plot to show only selected slice.
          - Drawing vertical lines in the col vs index plots.
          - Redrawing updated canvases.
        """
        x_col = f"{row}_X"
        y_col = f"{row}_Y"

        # 1) Update x_col vs y_col line to show only the selected slice
        line_colcol = self.plot_lines_colcol[row, 0]
        selected_x = self.data.loc[idx, x_col]
        selected_y = self.data.loc[idx, y_col]
        line_colcol.set_xdata(selected_x)
        line_colcol.set_ydata(selected_y)

        ax_colcol = self.axes_colcol[row][0]
        ax_colcol.relim()
        ax_colcol.autoscale_view()

        # 2) For col vs index, add vertical lines at xmin and xmax
        prob_col_name = f'defect_proba_{x_col}'
        has_defect_proba = prob_col_name in self.data.columns

        for subplot_index in range(2):
            ax = self.axes_grouped[row][subplot_index]

            # Clear old vertical lines
            self.clear_vertical_lines(row, subplot_index)

            # Draw new vertical lines
            self.draw_selection_lines(row, subplot_index, ax, xmin, xmax)

            # Update the y-data
            if subplot_index == 0:
                y_data = self.data[x_col]
            else:
                y_data = self.data[y_col]
            self.plot_lines_grouped[row, subplot_index].set_ydata(y_data)
            ax.relim()
            ax.autoscale_view()

            # Update scatter if it exists
            scatter = self.plot_scatter_grouped[row, subplot_index]
            if scatter and has_defect_proba:
                scatter.set_offsets(np.column_stack((self.data.index, y_data)))
                scatter.set_array(self.data[prob_col_name].values)

        # 3) Redraw
        self.canvases[row, 0].draw_idle()  # Grouped col vs index
        self.canvases[row, 1].draw_idle()  # col vs col

    def clear_vertical_lines(self, row, subplot_index):
        """Remove old vertical lines from a subplot."""
        lines = self.selection_lines[row][subplot_index]
        for ln in lines:
            ln.remove()
        self.selection_lines[row][subplot_index] = []

    def draw_selection_lines(self, row, subplot_index, ax, xmin, xmax):
        """Draw green dashed vertical lines on the axis for the selected range."""
        line1 = ax.axvline(x=xmin, color='green', linestyle='--')
        line2 = ax.axvline(x=xmax, color='green', linestyle='--')
        self.selection_lines[row][subplot_index] = [line1, line2]

    def update_phases(self, indices):
        """Compute and update phases for selected indices across all column pairs."""
        phases = self.compute_phases_for_column_pairs(self.data, indices)
        for row, (amp, ph) in phases.items():
            self.amplitude[row] = amp
            self.phase[row] = ph
        self.update_status_bar()

    def compute_phases_for_column_pairs(self, data, indices):
        """
        Compute amplitudes and phases for all column pairs over the specified indices
        using the external get_amp_phase function.
        """
        phases = {}
        for row in range(self.num_pairs):
            x_col = f"{row}_X"
            y_col = f"{row}_Y"
            x = data.loc[indices, x_col].values
            y = data.loc[indices, y_col].values

            if len(x) < 2 or len(y) < 2:
                print(f"Not enough data to compute phase for pair {row}")
                continue

            position = len(x) // 2
            width = len(x) // 2

            try:
                amp, ph = get_amp_phase(
                    list(zip(x, y)),
                    position=position,
                    width=width,
                    is_calibrated=self.is_calibrated,
                    calibration=self.calibration
                )
                phases[row] = (amp, ph)
                print(f"Row {row}: Amplitude={amp:.2f}, Phase={ph:.2f}°")
            except Exception as e:
                print(f"Error computing phase for pair {row}: {e}")
        return phases

    # -------------------------------------------------------------------------
    # Copy & Paste Operations
    # -------------------------------------------------------------------------
    def copy_data(self):
        """Copy the currently selected data to self.copied_data and compute its phase."""
        if self.copied_data.empty:
            print("No data selected to copy.")
            QMessageBox.warning(self, "Copy Data", "No data selected to copy.")
            return

        copied_indices = self.copied_data.index

        # Reset previous slices
        self.reset_previous_copies()

        # Mark newly copied data
        self.data.loc[copied_indices, 'slice_number'] = -1
        print("Data copied and marked with 'slice_number' = -1.")

        # Clear old phases, then compute & assign new phases
        self.clear_phases(copied_indices)
        phases = self.compute_phases_for_column_pairs(self.data, copied_indices)
        self.assign_phases(copied_indices, phases)

    def reset_previous_copies(self):
        """Reset 'slice_number' and phase columns for any previously copied data."""
        if 'slice_number' in self.data.columns:
            previously_copied = self.data['slice_number'] < 0
            self.data.loc[previously_copied, 'slice_number'] = np.nan
            for row in range(self.num_pairs):
                phase_col = f'phase_{row}'
                if phase_col in self.data.columns:
                    self.data.loc[previously_copied, phase_col] = np.nan
        else:
            self.data['slice_number'] = np.nan

    def clear_phases(self, indices):
        """Clear 'phase_{row}' columns for the given indices."""
        for row in range(self.num_pairs):
            phase_col = f'phase_{row}'
            if phase_col in self.data.columns:
                self.data.loc[indices, phase_col] = np.nan

    def assign_phases(self, indices, phases):
        """Assign computed phases to 'phase_{row}' columns for the selected indices."""
        for row, (amp, ph) in phases.items():
            phase_col = f'phase_{row}'
            if phase_col not in self.data.columns:
                self.data[phase_col] = np.nan
            self.data.loc[indices, phase_col] = ph
            print(f"Assigned phase {ph:.2f}° to '{phase_col}' for copied data.")
        self.update_status_bar()

    def paste_data(self, pos=None, scale=None):
        """
        Paste previously copied data into the dataset at a specified position,
        optionally scaled by 'scale'.
        If pos or scale is None, prompt the user.
        """
        if self.copied_data.empty:
            print("No data copied to paste.")
            QMessageBox.warning(self, "Paste Data", "No data copied to paste.")
            return False

        pos = self.get_paste_position(pos)
        if pos is None:
            return False

        scale = self.get_scale_factor(scale)
        if scale is None:
            return False

        paste_success = self.perform_paste(pos, scale)
        if paste_success:
            self.update_all_plots()
        return paste_success

    def get_paste_position(self, pos):
        """Prompt the user for the paste position if not provided."""
        if pos is not None:
            return pos
        pos, ok = QInputDialog.getInt(
            self, 'Paste Position', 'Enter paste index position:', min=0, max=len(self.data) - 1
        )
        return pos if ok else None

    def get_scale_factor(self, scale):
        """Prompt the user for the scale factor if not provided."""
        if scale is not None:
            return scale
        scale, ok = QInputDialog.getDouble(
            self, 'Scale Factor', 'Enter scale factor:', decimals=3
        )
        return scale if ok else None

    def perform_paste(self, pos, scale):
        """Perform the actual paste operation (adding scaled copied data)."""
        paste_data_df = self.copied_data.copy() * scale
        paste_length = len(paste_data_df)
        start_idx, end_idx = pos, min(pos + paste_length, len(self.data))

        if self.check_paste_overlap(start_idx, end_idx):
            return False

        # Only paste as many rows as fit in the data
        paste_data_df = paste_data_df.iloc[:end_idx - start_idx]

        # Add (not overwrite) the data
        self.data.loc[start_idx:end_idx - 1, paste_data_df.columns] += paste_data_df.values
        print(f"Pasted into position {start_idx} to {end_idx}")

        # Mark pasted data & compute phases
        self.mark_pasted_data(start_idx, end_idx)
        self.compute_and_assign_phases(start_idx, end_idx)
        return True

    def check_paste_overlap(self, start_idx, end_idx):
        """
        Check if the paste range overlaps with existing pasted data (slice_number not NaN).
        If overlap is found, warn and return True (indicating conflict).
        """
        if 'slice_number' in self.data.columns:
            overlap = self.data.loc[start_idx:end_idx - 1, 'slice_number'].notna().any()
            if overlap:
                QMessageBox.warning(
                    self, "Paste Conflict",
                    "The selected range contains already pasted data. Operation aborted."
                )
                print("Paste operation aborted: Overlapping with existing pasted data.")
                return True
        return False

    def mark_pasted_data(self, start_idx, end_idx):
        """Mark newly pasted data with a unique slice number."""
        if 'slice_number' not in self.data.columns:
            self.data['slice_number'] = np.nan
        self.paste_count += 1
        self.data.loc[start_idx:end_idx - 1, 'slice_number'] = self.paste_count

    def compute_and_assign_phases(self, start_idx, end_idx):
        """Compute and assign phases for newly pasted data in the range [start_idx, end_idx)."""
        indices = self.data.index[start_idx:end_idx]
        phases = self.compute_phases_for_column_pairs(self.data, indices)
        for row, (amp, ph) in phases.items():
            phase_col = f'phase_{row}'
            if phase_col not in self.data.columns:
                self.data[phase_col] = np.nan
            self.data.loc[indices, phase_col] = ph
            print(f"Computed phase for pasted data in pair {row}: {ph:.2f} degrees")
        self.update_status_bar()

    # -------------------------------------------------------------------------
    # Randomized Pasting
    # -------------------------------------------------------------------------
    def randomize_paste_positions(self, xmin, xmax, num_pastes, scale_type,
                                  scale=None, scale_min=None, scale_max=None):
        """
        Randomly paste the copied data multiple times within the range [xmin, xmax).
        scale_type can be 'Constant' or 'Random'.
        """
        if self.copied_data.empty:
            QMessageBox.warning(self, "Randomize Paste", "No data copied to paste.")
            print("No data copied to paste.")
            return

        paste_length = len(self.copied_data)
        available_positions = self.get_available_positions(xmin, xmax, paste_length)
        if not available_positions:
            QMessageBox.information(self, "Randomize Paste", "No available positions in the specified range.")
            print("No available positions to paste in the specified range.")
            return

        max_pastes = min(num_pastes, len(available_positions))
        if max_pastes < num_pastes:
            QMessageBox.warning(
                self, "Randomize Paste",
                f"Only {max_pastes} out of {num_pastes} pastes can be performed due to limited available space."
            )
            print(f"Only {max_pastes} out of {num_pastes} paste(s) can be performed due to limited space.")

        selected_positions = random.sample(available_positions, max_pastes)
        pastes_done = 0

        for pos in selected_positions:
            current_scale = scale if scale_type == 'Constant' else random.uniform(scale_min, scale_max)
            if self.paste_data(pos, current_scale):
                pastes_done += 1
            else:
                print(f"Paste at position {pos} was not performed.")

        QMessageBox.information(
            self, "Randomize Paste",
            f"Successfully completed {pastes_done} out of {num_pastes} paste(s)."
        )
        print(f"Successfully completed {pastes_done} out of {num_pastes} paste(s).")

    def get_available_positions(self, xmin, xmax, paste_length):
        """Return a list of available (non-overlapping) positions for pasting within [xmin, xmax)."""
        possible_positions = range(xmin, xmax - paste_length + 1)
        available_positions = []
        for pos in possible_positions:
            if 'slice_number' not in self.data.columns or \
               self.data.loc[pos:pos + paste_length - 1, 'slice_number'].isna().all():
                available_positions.append(pos)
        return available_positions

    # -------------------------------------------------------------------------
    # Saving Data
    # -------------------------------------------------------------------------
    def save_data(self, data_to_save):
        """Save the given DataFrame to a CSV file, prompting for a name."""
        if data_to_save.empty:
            QMessageBox.warning(self, "Save Data", "No data selected to save.")
            return

        file_name, ok = QInputDialog.getText(self, 'Save File', 'Enter file name:')
        if not ok or not file_name:
            return

        os.makedirs('saved', exist_ok=True)
        try:
            data_to_save.to_csv(f"saved/{file_name}.csv", index=False)
            QMessageBox.information(self, "Save Data", f"Data saved to saved/{file_name}.csv")
            print(f"Data saved to saved/{file_name}.csv")
        except Exception as e:
            QMessageBox.critical(self, "Save Data", f"Cannot save data to saved/{file_name}.csv:\n{e}")
            print(f"Error saving data: {e}")

    # -------------------------------------------------------------------------
    # Loading New Data
    # -------------------------------------------------------------------------
    def load_data(self):
        """
        Opens a file dialog for selecting a data file (CSV, DAT, or TXT),
        then reloads plots with the new data.
        """
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select a Data File", "",
            "Data Files (*.csv *.dat *.txt *.dep);;All Files (*)", options=options
        )
        if file_name:
            self.load_data_file(file_name)
            self.copied_data = pd.DataFrame()
            self.clear_plots()
            self.initialize_plot_arrays()
            self.create_plots()
            self.setLayout(self.main_layout)
            self.showMaximized()

    def clear_plots(self):
        """Remove all existing plots from the layout before loading new data."""
        for i in reversed(range(self.plot_layout.count())):
            item = self.plot_layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    # -------------------------------------------------------------------------
    # Plot Updating
    # -------------------------------------------------------------------------
    def update_plot(self, row):
        """
        Update plots for a given row (column pair) based on the latest self.data.
        """
        x_col = f"{row}_X"
        y_col = f"{row}_Y"

        # -- Grouped plots (x_col vs index, y_col vs index)
        y_data_0 = self.data[x_col].values
        line0 = self.plot_lines_grouped[row, 0]
        line0.set_ydata(y_data_0)

        scatter0 = self.plot_scatter_grouped[row, 0]
        if scatter0:
            scatter0.set_offsets(np.column_stack((self.data.index, y_data_0)))
            prob_col_name = f'defect_proba_{x_col}'
            if prob_col_name in self.data.columns:
                scatter0.set_array(self.data[prob_col_name].values)

        y_data_1 = self.data[y_col].values
        line1 = self.plot_lines_grouped[row, 1]
        line1.set_ydata(y_data_1)

        scatter1 = self.plot_scatter_grouped[row, 1]
        if scatter1:
            scatter1.set_offsets(np.column_stack((self.data.index, y_data_1)))
            prob_col_name = f'defect_proba_{x_col}'
            if prob_col_name in self.data.columns:
                scatter1.set_array(self.data[prob_col_name].values)

        for ax in self.axes_grouped[row]:
            ax.relim()
            ax.autoscale_view()

        self.canvases[row, 0].draw_idle()

        # -- x_col vs y_col plot
        line_colcol = self.plot_lines_colcol[row, 0]
        line_colcol.set_xdata(self.data[x_col])
        line_colcol.set_ydata(self.data[y_col])
        ax_colcol = self.axes_colcol[row][0]
        ax_colcol.relim()
        ax_colcol.autoscale_view()

        self.canvases[row, 1].draw_idle()

    def update_all_plots(self):
        """Update all plots in the interface."""
        for row in range(self.num_pairs):
            self.update_plot(row)

    # -------------------------------------------------------------------------
    # Axis Synchronization (Group Plots)
    # -------------------------------------------------------------------------
    def on_xlim_changed_grouped(self, axes):
        """Synchronize x-limits among all grouped column-index axes."""
        if self.is_zooming:
            return
        self.is_zooming = True
        try:
            xlim = axes.get_xlim()
            for row in range(self.num_pairs):
                for ax in self.axes_grouped[row]:
                    if ax != axes:
                        ax.set_xlim(xlim)
            # only call draw_idle on non‐None canvases
            for row in range(self.num_pairs):
                canvas = self.canvases[row, 0]
                if canvas is not None:
                    canvas.draw_idle()
        finally:
            self.is_zooming = False

    def on_ylim_changed_grouped(self, axes):
        """Synchronize y-limits among all grouped column-index axes."""
        if self.is_zooming:
            return
        self.is_zooming = True
        try:
            ylim = axes.get_ylim()
            for row in range(self.num_pairs):
                for ax in self.axes_grouped[row]:
                    if ax != axes:
                        ax.set_ylim(ylim)
            # only call draw_idle on non‐None canvases
            for row in range(self.num_pairs):
                canvas = self.canvases[row, 0]
                if canvas is not None:
                    canvas.draw_idle()
        finally:
            self.is_zooming = False


    # -------------------------------------------------------------------------
    # UI - Menus & Sub-Buttons
    # -------------------------------------------------------------------------
    def show_data_interference_buttons(self):
        """Show a menu for data interference: copy, paste, randomize paste."""
        self.show_sub_buttons("Data interference menu", [
            ('Copy', self.copy_data),
            ('Paste', self.paste_data),
            ('Randomize Paste Positions', self.show_randomize_paste_options)
        ])

    def show_save_data_buttons(self):
        """Show a menu for saving data."""
        self.show_sub_buttons("Save menu", [
            ('Save Sliced Data', lambda: self.save_data(self.copied_data)),
            ('Save All Data', lambda: self.save_data(self.data))
        ])

    def show_precise_selection_fields(self):
        """Show fields for precise data selection by index range."""
        self.sub_button_widget.show()
        self.clear_layout(self.sub_button_layout)
        self.sub_button_widget.setTitle("Precise data selection menu")

        from_label, to_label = QLabel('From Index:'), QLabel('To Index:')
        from_input, to_input = QLineEdit(), QLineEdit()
        select_button = QPushButton('Select')

        for widget in [from_label, from_input, to_label, to_input, select_button]:
            self.sub_button_layout.addWidget(widget)

        select_button.clicked.connect(
            lambda: self.precise_selection(from_input.text(), to_input.text())
        )

    def precise_selection(self, xmin_text, xmax_text):
        """Handle user's precise data selection request."""
        try:
            xmin, xmax = int(xmin_text), int(xmax_text)
            if xmin > xmax:
                raise ValueError("From Index cannot be greater than To Index.")
            self.on_select(xmin, xmax)
        except ValueError as ve:
            print(f"Precise selection error: {ve}")
            QMessageBox.warning(self, "Precise Selection", f"Invalid input: {ve}")

    def show_randomize_paste_options(self):
        """Show UI for randomizing paste positions."""
        self.sub_button_widget.show()
        self.clear_layout(self.sub_button_layout)
        self.sub_button_widget.setTitle("Randomize Paste Options")

        # Fields
        labels_texts = [
            ('Paste Position Range:', None),
            ('From Index:', '0'),
            ('To Index:', '5000'),
            ('Number of Pastes:', '1'),
            ('Scale Type:', None),
            ('Constant Scale Factor:', '1'),
            ('Random Scale Min:', '0'),
            ('Random Scale Max:', '1')
        ]
        widgets = {}

        for label_text, default_value in labels_texts:
            label = QLabel(label_text)
            self.sub_button_layout.addWidget(label)
            if default_value is not None:
                input_field = QLineEdit(default_value)
                self.sub_button_layout.addWidget(input_field)
                widgets[label_text] = (label, input_field)
            elif label_text == 'Scale Type:':
                scale_type_dropdown = QComboBox()
                scale_type_dropdown.addItems(['Constant', 'Random'])
                self.sub_button_layout.addWidget(scale_type_dropdown)
                widgets[label_text] = (label, scale_type_dropdown)
            else:
                widgets[label_text] = (label, None)

        # Randomize button
        randomize_button = QPushButton('Randomize and Paste')
        self.sub_button_layout.addWidget(randomize_button)

        # Show/hide fields based on selected scale type
        scale_fields = {
            'Constant': ['Constant Scale Factor:'],
            'Random': ['Random Scale Min:', 'Random Scale Max:']
        }
        scale_type_dropdown = widgets['Scale Type:'][1]

        def update_scale_fields():
            selected_type = scale_type_dropdown.currentText()
            for field in ['Constant Scale Factor:', 'Random Scale Min:', 'Random Scale Max:']:
                label, input_widget = widgets[field]
                is_visible = field in scale_fields[selected_type]
                label.setVisible(is_visible)
                if input_widget:
                    input_widget.setVisible(is_visible)

        scale_type_dropdown.currentIndexChanged.connect(update_scale_fields)
        update_scale_fields()

        # Connect randomize button
        def on_randomize_paste_clicked():
            self.handle_randomize_paste(widgets)

        randomize_button.clicked.connect(on_randomize_paste_clicked)

    def handle_randomize_paste(self, widgets):
        """Extract values from UI and perform randomize paste."""
        try:
            xmin = int(widgets['From Index:'][1].text())
            xmax = int(widgets['To Index:'][1].text())
            num_pastes = int(widgets['Number of Pastes:'][1].text())
            scale_type = widgets['Scale Type:'][1].currentText()

            if xmin > xmax or xmin < 0 or xmax >= len(self.data):
                raise ValueError("Ensure 0 <= From Index <= To Index < data length.")

            if scale_type == 'Constant':
                scale = float(widgets['Constant Scale Factor:'][1].text())
                self.randomize_paste_positions(xmin, xmax, num_pastes, scale_type, scale=scale)
            else:
                scale_min = float(widgets['Random Scale Min:'][1].text())
                scale_max = float(widgets['Random Scale Max:'][1].text())
                if scale_min > scale_max:
                    raise ValueError("Ensure scale min <= scale max.")
                self.randomize_paste_positions(
                    xmin, xmax, num_pastes, scale_type, scale_min=scale_min, scale_max=scale_max
                )
        except ValueError as ve:
            print(f"Randomize paste error: {ve}")
            QMessageBox.warning(self, "Randomize Paste", f"Invalid input: {ve}")

    def show_sub_buttons(self, title, buttons):
        """Display a set of sub-buttons in the side panel."""
        self.sub_button_widget.show()
        self.clear_layout(self.sub_button_layout)
        self.sub_button_widget.setTitle(title)

        for text, func in buttons:
            button = QPushButton(text)
            button.clicked.connect(func)
            self.sub_button_layout.addWidget(button)

    def clear_layout(self, layout):
        """Remove all widgets from the given layout."""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    # -------------------------------------------------------------------------
    # Status Bar Update
    # -------------------------------------------------------------------------
    def update_status_bar(self):
        """Update the status bar with the current phase information."""
        if not self.phase.any():
            self.status_bar.setText("Phases: N/A")
            return

        phase_info = []
        for row in range(self.num_pairs):
            phase_value = self.phase[row]
            if not np.isnan(phase_value):
                phase_info.append(f"\nRow {row}: {phase_value:.2f}°")
            else:
                phase_info.append(f"\nRow {row}: N/A")

        status_text = "Phases:" + "".join(phase_info)
        self.status_bar.setText(status_text)


# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------
if __name__ == '__main__':
    # New: Parse command-line arguments to accept a predefined num_pairs option
    parser = argparse.ArgumentParser(description="Data Visualization Tool")
    parser.add_argument('--num_pairs', type=int, default=None,
                        help="Predefined number of column pairs to plot")
    args, unknown = parser.parse_known_args()

    app = QApplication(sys.argv)
    window = DataVisualizationTool(num_pairs=args.num_pairs)
    window.show()
    sys.exit(app.exec_())
