import os
import sys
import random
import pandas as pd
import numpy as np

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


class DataVisualizationTool(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Visualization Tool")
        self.init_variables()
        self.init_ui()
        self.load_initial_data()
        self.create_plots()
        self.setup_shortcuts()
        self.showMaximized()

    def init_variables(self):
        """Initialize variables used throughout the class."""
        self.copied_data = pd.DataFrame()
        self.paste_count = 0
        self.calibration = {'voltage': 5.0, 'amplitude': 1.0}
        self.is_calibrated = False
        self.amplitude = []
        self.phase = []
        self.is_zooming = False

    def init_ui(self):
        """Initialize the user interface components."""
        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)

        self.init_plot_area()
        self.init_button_panel()

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

        self.add_main_buttons()
        self.add_sub_button_placeholder()
        self.add_status_bar()
        self.add_spacer()
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
        """Add a placeholder for sub-buttons."""
        self.sub_button_layout = QVBoxLayout()
        self.sub_button_widget = QGroupBox()
        self.sub_button_widget.setLayout(self.sub_button_layout)
        self.sub_button_widget.hide()
        self.button_layout.addWidget(self.sub_button_widget)

    def add_status_bar(self):
        """Add a status bar to display phase information."""
        self.status_bar = QLabel("Phases: N/A")
        self.status_bar.setStyleSheet("background-color: #333; color: white; padding: 5px;")
        self.status_bar.setAlignment(Qt.AlignLeft)
        self.status_bar.setWordWrap(True)
        self.button_layout.addWidget(self.status_bar)

    def add_spacer(self):
        """Add a spacer to push elements to the top."""
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
        """Set up keyboard shortcuts for the application."""
        QShortcut(QKeySequence("Ctrl+C"), self).activated.connect(self.copy_data)
        QShortcut(QKeySequence("Ctrl+V"), self).activated.connect(self.paste_data)
        QShortcut(QKeySequence("Ctrl+Shift+V"), self).activated.connect(self.show_randomize_paste_options)

    def load_initial_data(self):
        """Load initial data from the specified file."""
        data_file_path = os.path.join('Andrii', 'raw', 'rawdata_cal.dat')
        self.load_data_file(data_file_path)
        self.initialize_plot_arrays()

    def load_data_file(self, file_path):
        """Load data from the specified file."""
        try:
            self.data = pd.read_csv(file_path, sep=r'\s+', header=None)
            self.data = self.data.iloc[::-1].reset_index(drop=True)
            print(f"Data loaded and reversed successfully from {file_path}")
            self.num_pairs = len(self.data.columns) // 2 - 1
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", f"Data file not found: {file_path}")
            sys.exit(1)
        except pd.errors.ParserError as e:
            QMessageBox.critical(self, "Error", f"Error parsing data file: {e}")
            sys.exit(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")
            sys.exit(1)

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

    def create_plots(self):
        """Create all plots for the data."""
        for row in range(self.num_pairs):
            self.create_individual_plots(row)

    def create_individual_plots(self, row):
        """Create individual plots for a given row."""
        colN = row * 2
        colN1 = colN + 1

        fig_grouped, canvas_grouped = self.create_grouped_plot(row)
        self.figures[row, 0], self.canvases[row, 0] = fig_grouped, canvas_grouped

        ax1, line1 = self.plot_column_vs_index(fig_grouped, self.data[colN], f'Column {colN}', row, 0)
        ax2, line2 = self.plot_column_vs_index(fig_grouped, self.data[colN1], f'Column {colN1}', row, 1, sharex=ax1)

        self.plot_lines_grouped[row, 0], self.plot_lines_grouped[row, 1] = line1, line2
        self.axes_grouped[row].extend([ax1, ax2])

        self.span_selectors[row, 0] = self.create_span_selector(ax1)
        self.span_selectors[row, 1] = self.create_span_selector(ax2)

        self.synchronize_axes(ax1, ax2)

        fig_colcol, canvas_colcol = self.create_plot_with_toolbar(row)
        self.figures[row, 1], self.canvases[row, 1] = fig_colcol, canvas_colcol

        ax_colcol, line_colcol = self.plot_column_vs_column(fig_colcol, self.data[colN], self.data[colN1], colN, colN1)
        self.plot_lines_colcol[row, 0] = line_colcol
        self.axes_colcol[row].append(ax_colcol)

    def create_grouped_plot(self, row):
        """Create a grouped plot widget with two subplots."""
        fig = Figure(facecolor='black', constrained_layout=True)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self)
        self.plot_layout.addWidget(toolbar, row * 2, 0)
        self.plot_layout.addWidget(canvas, row * 2 + 1, 0)
        return fig, canvas

    def create_plot_with_toolbar(self, row):
        """Create a plot with a navigation toolbar."""
        fig = Figure(facecolor='black', constrained_layout=True)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self)
        self.plot_layout.addWidget(toolbar, row * 2, 1)
        self.plot_layout.addWidget(canvas, row * 2 + 1, 1)
        return fig, canvas

    def plot_column_vs_index(self, fig, data_column, label, row, subplot_index, sharex=None):
        """Plot a data column against index."""
        ax = fig.add_subplot(211 + subplot_index, sharex=sharex) if sharex else fig.add_subplot(211 + subplot_index)
        line, = ax.plot(self.data.index, data_column, label=label)
        self.setup_plot(ax, label if subplot_index == 0 else '', '', 'Voltage')
        if subplot_index == 0:
            ax.tick_params(labelbottom=False, bottom=False)
        self.set_dark_theme(ax)
        return ax, line

    def plot_column_vs_column(self, fig, x_data, y_data, colN, colN1):
        """Plot one data column against another."""
        ax = fig.add_subplot(111)
        line, = ax.plot(x_data, y_data, 'g')
        self.setup_plot(ax, f'Column {colN} vs Column {colN1}', f'Column {colN}', f'Column {colN1}')
        self.set_dark_theme(ax)
        return ax, line

    def setup_plot(self, ax, title, xlabel='', ylabel=''):
        """Setup plot aesthetics."""
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(color='gray', linestyle='--', linewidth=0.5)

    def set_dark_theme(self, ax):
        """Set dark theme for the plot axes."""
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')

    def create_span_selector(self, ax):
        """Create a SpanSelector for a given axis."""
        return SpanSelector(
            ax, self.on_select, 'horizontal', useblit=False, props=dict(alpha=0.5, facecolor='red')
        )

    def synchronize_axes(self, ax1, ax2):
        """Synchronize x-limits and y-limits between two axes."""
        ax1.callbacks.connect('xlim_changed', self.on_xlim_changed_grouped)
        ax1.callbacks.connect('ylim_changed', self.on_ylim_changed_grouped)
        ax2.callbacks.connect('xlim_changed', self.on_xlim_changed_grouped)
        ax2.callbacks.connect('ylim_changed', self.on_ylim_changed_grouped)

    def on_select(self, xmin, xmax):
        """Handle data selection via SpanSelector."""
        # Add validation to ensure xmin and xmax are valid numbers
        if not isinstance(xmin, (int, float)) or not isinstance(xmax, (int, float)):
            print("Invalid selection")
            return

        print(f"Selection from {xmin} to {xmax}")
        xmin, xmax = int(np.round(xmin)), int(np.round(xmax))

        # Clamp values to valid range
        xmin, xmax = max(0, xmin), min(len(self.data) - 1, xmax)

        idx = (self.data.index >= xmin) & (self.data.index <= xmax)

        print(f"Selected indices: {idx.sum()} entries")

        # Store all selected columns
        self.copied_data = self.data.loc[idx].copy()

        # Update specific plots
        for row in range(self.num_pairs):
            self.update_selection_plots(row, idx, xmin, xmax)

        # Compute phases for selected indices
        selected_indices = self.data.index[idx]
        self.update_phases(selected_indices)

    def update_selection_plots(self, row, idx, xmin, xmax):
        """Update plots based on the selection."""
        colN = row * 2
        colN1 = colN + 1

        # Update colN vs colN+1 plot
        line = self.plot_lines_colcol[row, 0]
        line.set_xdata(self.data.loc[idx, colN])
        line.set_ydata(self.data.loc[idx, colN1])
        ax = self.axes_colcol[row][0]
        ax.relim()
        ax.autoscale_view()
        self.canvases[row, 1].draw_idle()

        # Update vertical lines in the grouped plots
        for ax in self.axes_grouped[row]:
            self.clear_vertical_lines(ax)
            ax.axvline(x=xmin, color='green', linestyle='--')
            ax.axvline(x=xmax, color='green', linestyle='--')
            self.canvases[row, 0].draw_idle()

    def clear_vertical_lines(self, ax):
        """Clear vertical lines from an axis."""
        lines_to_remove = [line for line in ax.lines[1:] if line.get_linestyle() == '--']
        for line in lines_to_remove:
            line.remove()

    def update_phases(self, indices):
        """Compute and update phases for selected indices."""
        phases = self.compute_phases_for_column_pairs(self.data, indices)
        for row, (amplitude, phase) in phases.items():
            self.amplitude[row] = amplitude
            self.phase[row] = phase
        self.update_status_bar()

    def compute_phases_for_column_pairs(self, data, indices):
        """Compute amplitudes and phases for all column pairs over specified indices."""
        phases = {}
        for row in range(self.num_pairs):
            colN, colN1 = row * 2, row * 2 + 1
            x = data.loc[indices, colN].values
            y = data.loc[indices, colN1].values

            if len(x) < 2 or len(y) < 2:
                print(f"Not enough data to compute phase for pair {row}")
                continue

            position, width = len(x) // 2, len(x) // 2
            try:
                amplitude, phase = get_amp_phase(
                    list(zip(x, y)),
                    position=position,
                    width=width,
                    is_calibrated=self.is_calibrated,
                    calibration=self.calibration
                )
                phases[row] = (amplitude, phase)
                print(f"Row {row}: Amplitude: {amplitude:.2f}, Phase: {phase:.2f}°")
            except Exception as e:
                print(f"Error computing phase for pair {row}: {e}")
        return phases

    def copy_data(self):
        """Copy selected data and compute its phase."""
        if self.copied_data.empty:
            print("No data selected to copy.")
            return

        copied_indices = self.copied_data.index
        self.reset_previous_copies()
        self.data.loc[copied_indices, 'slice_number'] = -1
        print("Data copied and marked with 'slice_number' = -1.")

        # Clear and compute phases
        self.clear_phases(copied_indices)
        phases = self.compute_phases_for_column_pairs(self.data, copied_indices)
        self.assign_phases(copied_indices, phases)

    def reset_previous_copies(self):
        """Reset 'slice_number' and 'phase_{row}' columns for previously copied data."""
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
        """Assign computed phases to 'phase_{row}' columns."""
        for row, (amplitude, phase) in phases.items():
            phase_col = f'phase_{row}'
            if phase_col not in self.data.columns:
                self.data[phase_col] = np.nan
            self.data.loc[indices, phase_col] = phase
            print(f"Assigned phase {phase:.2f}° to '{phase_col}' for copied data.")
        self.update_status_bar()

    def paste_data(self, pos=None, scale=None):
        """Paste copied data into the dataset."""
        if self.copied_data.empty:
            print("No data copied to paste.")
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
        """Prompt user for paste position if not provided."""
        if pos is not None:
            return pos
        pos, ok = QInputDialog.getInt(
            self, 'Paste Position', 'Enter paste index position:', min=0, max=len(self.data) - 1
        )
        return pos if ok else None

    def get_scale_factor(self, scale):
        """Prompt user for scale factor if not provided."""
        if scale is not None:
            return scale
        scale, ok = QInputDialog.getDouble(
            self, 'Scale Factor', 'Enter scale factor:', decimals=3
        )
        return scale if ok else None

    def perform_paste(self, pos, scale):
        """Perform the paste operation."""
        paste_data_df = self.copied_data.copy() * scale
        paste_length = len(paste_data_df)
        start_idx, end_idx = pos, min(pos + paste_length, len(self.data))

        if self.check_paste_overlap(start_idx, end_idx):
            return False

        paste_data_df = paste_data_df.iloc[:end_idx - start_idx]
        self.data.loc[start_idx:end_idx - 1, paste_data_df.columns] += paste_data_df.values
        print(f"Pasted into position {start_idx} to {end_idx}")

        self.mark_pasted_data(start_idx, end_idx)
        self.compute_and_assign_phases(start_idx, end_idx)
        return True

    def check_paste_overlap(self, start_idx, end_idx):
        """Check for overlap with existing pasted data."""
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
        """Mark pasted data with a new slice number."""
        if 'slice_number' not in self.data.columns:
            self.data['slice_number'] = np.nan
        self.paste_count += 1
        self.data.loc[start_idx:end_idx - 1, 'slice_number'] = self.paste_count

    def compute_and_assign_phases(self, start_idx, end_idx):
        """Compute and assign phases for the pasted data."""
        indices = self.data.index[start_idx:end_idx]
        phases = self.compute_phases_for_column_pairs(self.data, indices)
        for row, (amplitude, phase) in phases.items():
            phase_col = f'phase_{row}'
            if phase_col not in self.data.columns:
                self.data[phase_col] = np.nan
            self.data.loc[indices, phase_col] = phase
            print(f"Computed phase for pasted data in pair {row}: {phase:.2f} degrees")
        self.update_status_bar()

    def save_data(self, data_to_save):
        """Save the specified data to a file."""
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
        except Exception as e:
            QMessageBox.critical(self, "Save Data", f"Cannot save data to saved/{file_name}.csv:\n{e}")

    def load_data(self):
        """Load new data from a file."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", "Data Files (*.csv *.dat *.txt);;All Files (*)", options=options
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
        """Clear existing plots from the layout."""
        for i in reversed(range(self.plot_layout.count())):
            widget = self.plot_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

    def update_plot(self, row):
        """Update a single plot based on current data."""
        colN, colN1 = row * 2, row * 2 + 1

        # Update grouped plots
        for col, data in enumerate([(self.data.index, self.data[colN]), (self.data.index, self.data[colN1])]):
            x_data, y_data = data
            line = self.plot_lines_grouped[row, col]
            line.set_xdata(x_data)
            line.set_ydata(y_data)
            ax = self.axes_grouped[row][col]
            ax.relim()
            ax.autoscale_view()

        # Update colN vs colN+1 plot
        line_colcol = self.plot_lines_colcol[row, 0]
        line_colcol.set_xdata(self.data[colN])
        line_colcol.set_ydata(self.data[colN1])
        ax_colcol = self.axes_colcol[row][0]
        ax_colcol.relim()
        ax_colcol.autoscale_view()

        # Redraw canvases
        self.canvases[row, 0].draw_idle()
        self.canvases[row, 1].draw_idle()

    def update_all_plots(self):
        """Update all plots."""
        for row in range(self.num_pairs):
            self.update_plot(row)

    def on_xlim_changed_grouped(self, axes):
        """Synchronize x-limits across all Column vs Index plots."""
        if self.is_zooming:
            return
        self.is_zooming = True
        try:
            xlim = axes.get_xlim()
            for row in range(self.num_pairs):
                for ax in self.axes_grouped[row]:
                    if ax != axes:
                        ax.set_xlim(xlim)
            for row in range(self.num_pairs):
                self.canvases[row, 0].draw_idle()
        finally:
            self.is_zooming = False

    def on_ylim_changed_grouped(self, axes):
        """Synchronize y-limits across all Column vs Index plots."""
        if self.is_zooming:
            return
        self.is_zooming = True
        try:
            ylim = axes.get_ylim()
            for row in range(self.num_pairs):
                for ax in self.axes_grouped[row]:
                    if ax != axes:
                        ax.set_ylim(ylim)
            for row in range(self.num_pairs):
                self.canvases[row, 0].draw_idle()
        finally:
            self.is_zooming = False

    def get_row_from_figure(self, fig):
        """Get the row index for a given figure."""
        for row in range(self.num_pairs):
            if fig == self.figures[row, 0] or fig == self.figures[row, 1]:
                return row
        return -1  # Not found

    def randomize_paste_positions(self, xmin, xmax, num_pastes, scale_type, scale=None, scale_min=None, scale_max=None):
        """Randomly paste data at different positions."""
        if self.copied_data.empty:
            QMessageBox.warning(self, "Randomize Paste", "No data copied to paste.")
            print("No data copied to paste.")
            return

        paste_length = len(self.copied_data)
        available_positions = self.get_available_positions(xmin, xmax, paste_length)

        if not available_positions:
            QMessageBox.information(
                self, "Randomize Paste", "No available positions to paste in the specified range."
            )
            print("No available positions to paste in the specified range.")
            return

        max_pastes = min(num_pastes, len(available_positions))
        if max_pastes < num_pastes:
            QMessageBox.warning(
                self, "Randomize Paste",
                f"Only {max_pastes} out of {num_pastes} pastes can be performed due to limited available space."
            )
            print(f"Only {max_pastes} out of {num_pastes} pastes can be performed due to limited space.")

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
        """Get a list of available positions for pasting."""
        possible_positions = range(xmin, xmax - paste_length + 1)
        available_positions = []

        for pos in possible_positions:
            if 'slice_number' not in self.data.columns or \
               self.data.loc[pos:pos + paste_length - 1, 'slice_number'].isna().all():
                available_positions.append(pos)
        return available_positions

    # UI Functions to show various menus and options
    def show_data_interference_buttons(self):
        """Show data interference menu."""
        self.show_sub_buttons("Data interference menu", [
            ('Copy', self.copy_data),
            ('Paste', self.paste_data),
            ('Randomize Paste Positions', self.show_randomize_paste_options)
        ])

    def show_save_data_buttons(self):
        """Show save data menu."""
        self.show_sub_buttons("Save menu", [
            ('Save Sliced Data', lambda: self.save_data(self.copied_data)),
            ('Save All Data', lambda: self.save_data(self.data))
        ])

    def show_precise_selection_fields(self):
        """Show precise data selection menu."""
        self.sub_button_widget.show()
        self.clear_layout(self.sub_button_layout)
        self.sub_button_widget.setTitle("Precise data selection menu")

        from_label, to_label = QLabel('From Index:'), QLabel('To Index:')
        from_input, to_input = QLineEdit(), QLineEdit()
        select_button = QPushButton('Select')

        for widget in [from_label, from_input, to_label, to_input, select_button]:
            self.sub_button_layout.addWidget(widget)

        select_button.clicked.connect(lambda: self.precise_selection(from_input.text(), to_input.text()))

    def precise_selection(self, xmin_text, xmax_text):
        """Handle precise data selection."""
        try:
            xmin, xmax = int(xmin_text), int(xmax_text)
            self.on_select(xmin, xmax)
        except ValueError:
            print("Please enter valid integer indices.")
            QMessageBox.warning(
                self, "Precise Selection",
                "Please enter valid integer indices."
            )

    def show_randomize_paste_options(self):
        """Show options for randomizing paste positions."""
        self.sub_button_widget.show()
        self.clear_layout(self.sub_button_layout)
        self.sub_button_widget.setTitle("Randomize Paste Options")

        # Widgets and labels
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

        # Update visibility based on scale type
        scale_fields = {
            'Constant': ['Constant Scale Factor:'],
            'Random': ['Random Scale Min:', 'Random Scale Max:']
        }

        def update_scale_fields():
            selected_type = scale_type_dropdown.currentText()
            for field in ['Constant Scale Factor:', 'Random Scale Min:', 'Random Scale Max:']:
                label, input_widget = widgets[field]
                is_visible = field in scale_fields[selected_type]
                label.setVisible(is_visible)
                input_widget.setVisible(is_visible)

        scale_type_dropdown.currentIndexChanged.connect(update_scale_fields)
        update_scale_fields()

        # Connect randomize button
        def on_randomize_paste_clicked():
            self.handle_randomize_paste(widgets)

        randomize_button.clicked.connect(on_randomize_paste_clicked)

    def handle_randomize_paste(self, widgets):
        """Handle randomize paste based on user inputs."""
        try:
            xmin = int(widgets['From Index:'][1].text())
            xmax = int(widgets['To Index:'][1].text())
            num_pastes = int(widgets['Number of Pastes:'][1].text())
            scale_type = widgets['Scale Type:'][1].currentText()

            if xmin > xmax or xmin < 0 or xmax >= len(self.data):
                print("Invalid range. Ensure 0 <= From Index <= To Index < data length.")
                QMessageBox.warning(
                    self, "Randomize Paste",
                    "Invalid range. Ensure 0 <= From Index <= To Index < data length."
                )
                return

            if scale_type == 'Constant':
                scale = float(widgets['Constant Scale Factor:'][1].text())
                self.randomize_paste_positions(xmin, xmax, num_pastes, scale_type, scale=scale)
            else:
                scale_min = float(widgets['Random Scale Min:'][1].text())
                scale_max = float(widgets['Random Scale Max:'][1].text())
                if scale_min > scale_max:
                    print("Invalid scale range. Ensure scale min <= scale max.")
                    QMessageBox.warning(
                        self, "Randomize Paste",
                        "Invalid scale range. Ensure scale min <= scale max."
                    )
                    return
                self.randomize_paste_positions(xmin, xmax, num_pastes, scale_type, scale_min=scale_min, scale_max=scale_max)
        except ValueError:
            print("Please enter valid integer and float values.")
            QMessageBox.warning(
                self, "Randomize Paste",
                "Please enter valid integer and float values."
            )

    def show_sub_buttons(self, title, buttons):
        """Show sub-buttons in the side panel."""
        self.sub_button_widget.show()
        self.clear_layout(self.sub_button_layout)
        self.sub_button_widget.setTitle(title)
        for text, func in buttons:
            button = QPushButton(text)
            button.clicked.connect(func)
            self.sub_button_layout.addWidget(button)

    def clear_layout(self, layout):
        """Clear all widgets from a layout."""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def update_status_bar(self):
        """Update the status bar with current phase information."""
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
        status_text = "Phases: " + ", ".join(phase_info)
        self.status_bar.setText(status_text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DataVisualizationTool()
    window.show()
    sys.exit(app.exec_())
