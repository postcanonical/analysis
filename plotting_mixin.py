import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import SpanSelector
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt # For isinstance checks and cm

class PlottingMixin:
    
    def initialize_plot_infrastructure(self):
        """Initializes arrays to hold plot-related objects, called after num_pairs is known."""
        if self.num_pairs <= 0: # Ensure num_pairs is valid
            self.num_pairs = 0 
        
        self.figures = np.empty((self.num_pairs, 2), dtype=object)
        self.canvases = np.empty((self.num_pairs, 2), dtype=object)
        self.axes_grouped = [[] for _ in range(self.num_pairs)] # List of lists for grouped axes
        self.axes_colcol = [[] for _ in range(self.num_pairs)]  # List of lists for col-col axes
        self.span_selectors = np.empty((self.num_pairs, 2), dtype=object)
        self.plot_lines_grouped = np.empty((self.num_pairs, 2), dtype=object)
        self.plot_lines_colcol = np.empty((self.num_pairs, 1), dtype=object) # Only one line artist per col-col plot
        
        # Initialize selection_lines correctly based on num_pairs
        self.selection_lines = [[[] for _ in range(2)] for _ in range(self.num_pairs)]


    def apply_theme_to_all_plots(self):
        if not hasattr(self, 'figures') or not self.figures.any() or \
           not hasattr(self, 'axes_grouped') or not hasattr(self, 'axes_colcol') or \
           self.num_pairs == 0:
            return

        theme = self.themes[self.current_theme]
        for row in range(self.num_pairs):
            if row >= self.figures.shape[0]: continue

            # Grouped plots (X vs Index, Y vs Index)
            if self.figures[row, 0] is not None and row < len(self.axes_grouped) and self.axes_grouped[row]:
                self.figures[row, 0].set_facecolor(theme['figure_bg'])
                for ax_idx, ax in enumerate(self.axes_grouped[row]):
                    if ax is None: continue
                    self._apply_theme_to_single_axis(ax, theme)
                    # Update line/collection colors
                    if row < self.plot_lines_grouped.shape[0] and ax_idx < self.plot_lines_grouped.shape[1]:
                        plot_obj = self.plot_lines_grouped[row, ax_idx]
                        if isinstance(plot_obj, LineCollection):
                            new_cmap = mcolors.LinearSegmentedColormap.from_list(
                                f'defect_cmap_grouped_{row}_{ax_idx}_{theme["line_color"].replace("#","")}',
                                [theme['line_color'], 'red'], N=256) # Assuming 'red' is still the defect color
                            plot_obj.set_cmap(new_cmap)
                        elif isinstance(plot_obj, plt.Line2D): # Check for standard lines
                            plot_obj.set_color(theme['line_color'])
                if self.canvases[row, 0]: self.canvases[row, 0].draw_idle()

            # Column vs Column plots
            if self.figures[row, 1] is not None and row < len(self.axes_colcol) and self.axes_colcol[row] and self.axes_colcol[row][0]:
                self.figures[row, 1].set_facecolor(theme['figure_bg'])
                ax_col = self.axes_colcol[row][0] # Assuming one axis per col-col plot
                if ax_col is None: continue
                self._apply_theme_to_single_axis(ax_col, theme)
                # Update line/collection colors
                if row < self.plot_lines_colcol.shape[0]:
                    plot_obj_colcol = self.plot_lines_colcol[row, 0]
                    if isinstance(plot_obj_colcol, LineCollection):
                        new_cmap_colcol = mcolors.LinearSegmentedColormap.from_list(
                            f'col_vs_col_defect_cmap_{row}_{theme["line_color"].replace("#","")}',
                            [theme['line_color'], 'red'], N=256)
                        plot_obj_colcol.set_cmap(new_cmap_colcol)
                    elif isinstance(plot_obj_colcol, plt.Line2D):
                        plot_obj_colcol.set_color(theme['line_color'])
                if self.canvases[row, 1]: self.canvases[row, 1].draw_idle()
            
            # Update selection line colors
            if hasattr(self, 'selection_lines') and row < len(self.selection_lines):
                for subplot_idx in range(len(self.selection_lines[row])): # Should be 2 for grouped plots
                    for line_obj in self.selection_lines[row][subplot_idx]:
                        if line_obj: line_obj.set_color(theme['selection_color'])
            
            # Update SpanSelector facecolor
            if hasattr(self, 'span_selectors') and self.span_selectors is not None:
                 if row < self.span_selectors.shape[0]:
                    for col_idx_span in range(self.span_selectors.shape[1]): # Should be 2 for grouped plots
                        span_selector_widget = self.span_selectors[row, col_idx_span]
                        if span_selector_widget and hasattr(span_selector_widget, 'props'): # Check if SpanSelector exists
                            span_selector_widget.props['facecolor'] = theme['span_color']
                            # If the span is active, also update its visible patch
                            if span_selector_widget.active and hasattr(span_selector_widget, 'patch') and span_selector_widget.patch:
                                span_selector_widget.patch.set_facecolor(theme['span_color'])


        # Redraw all canvases after theme changes
        for r_idx in range(self.num_pairs): # Iterate up to num_pairs
            if r_idx < self.canvases.shape[0]:
                if self.canvases[r_idx, 0]: self.canvases[r_idx, 0].draw_idle()
                if self.canvases[r_idx, 1]: self.canvases[r_idx, 1].draw_idle()

    def _apply_theme_to_single_axis(self, ax, theme):
        ax.set_facecolor(theme['axes_bg'])
        ax.tick_params(axis='x', colors=theme['text_color'])
        ax.tick_params(axis='y', colors=theme['text_color'])
        ax.xaxis.label.set_color(theme['text_color'])
        ax.yaxis.label.set_color(theme['text_color'])
        ax.title.set_color(theme['text_color'])
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color=theme['grid_color'])
        for spine in ax.spines.values():
            spine.set_edgecolor(theme['text_color'])

    def recreate_all_plots(self):
        if self.data.empty or self.num_pairs == 0 or not self.current_plot_prefixes:
            # If there's no data or no pairs, ensure plots are cleared
            self.clear_all_plots()
            return

        # Ensure plot infrastructure is ready for the current num_pairs
        # self.initialize_plot_infrastructure() # Called by load_data_from_file or initial setup

        # Clear existing plot widgets from the layout before creating new ones
        self.clear_all_plots() # This clears artists and also widgets from plot_layout

        for idx, prefix in enumerate(self.current_plot_prefixes):
            if idx >= self.num_pairs: # Only create plots up to num_pairs
                break
            self._create_plots_for_pair(idx, prefix)
        
        self.apply_theme_to_all_plots() # Apply current theme to newly created plots

    def _create_plots_for_pair(self, plot_row_idx, data_prefix):
        x_col_name = f"{data_prefix}_X"
        y_col_name = f"{data_prefix}_Y"

        if x_col_name not in self.data.columns or y_col_name not in self.data.columns:
            # Ensure no stale plot objects if data columns are missing
            self.figures[plot_row_idx, 0] = None
            self.canvases[plot_row_idx, 0] = None
            self.figures[plot_row_idx, 1] = None
            self.canvases[plot_row_idx, 1] = None
            if plot_row_idx < len(self.axes_grouped): self.axes_grouped[plot_row_idx] = []
            if plot_row_idx < len(self.axes_colcol): self.axes_colcol[plot_row_idx] = []
            return

        # Create grouped plot (X vs Index, Y vs Index)
        fig_grouped, canvas_grouped = self._create_figure_and_canvas(plot_row_idx, 0) # 0 for grouped plot column
        self.figures[plot_row_idx, 0] = fig_grouped
        self.canvases[plot_row_idx, 0] = canvas_grouped
        
        # Plot X vs Index
        ax1_grouped, line1_grouped = self._plot_data_vs_index(fig_grouped, self.data[x_col_name], 
                                                             f'{x_col_name} vs Index', 
                                                             plot_row_idx, data_prefix, 0) # subplot_index 0
        # Plot Y vs Index (sharing X-axis with X vs Index plot)
        ax2_grouped, line2_grouped = self._plot_data_vs_index(fig_grouped, self.data[y_col_name],
                                                             f'{y_col_name} vs Index',
                                                             plot_row_idx, data_prefix, 1, sharex=ax1_grouped) # subplot_index 1
        
        self.plot_lines_grouped[plot_row_idx, 0] = line1_grouped
        self.plot_lines_grouped[plot_row_idx, 1] = line2_grouped
        if plot_row_idx < len(self.axes_grouped):
             self.axes_grouped[plot_row_idx] = [ax1_grouped, ax2_grouped]
        else: # Should not happen if initialized correctly
             self.axes_grouped.append([ax1_grouped, ax2_grouped])


        if ax1_grouped: self.span_selectors[plot_row_idx, 0] = self._create_span_selector_for_axis(ax1_grouped)
        if ax2_grouped: self.span_selectors[plot_row_idx, 1] = self._create_span_selector_for_axis(ax2_grouped)
        if ax1_grouped and ax2_grouped: self._synchronize_x_axes(ax1_grouped, ax2_grouped)

        # Create column vs column plot (X vs Y)
        fig_colcol, canvas_colcol = self._create_figure_and_canvas(plot_row_idx, 1) # 1 for col-col plot column
        self.figures[plot_row_idx, 1] = fig_colcol
        self.canvases[plot_row_idx, 1] = canvas_colcol

        ax_colcol, line_colcol = self._plot_data_col_vs_col(fig_colcol, self.data[x_col_name], self.data[y_col_name],
                                                          x_col_name, y_col_name, plot_row_idx, data_prefix)
        self.plot_lines_colcol[plot_row_idx, 0] = line_colcol
        if plot_row_idx < len(self.axes_colcol):
            self.axes_colcol[plot_row_idx] = [ax_colcol]
        else: # Should not happen
            self.axes_colcol.append([ax_colcol])


    def _create_figure_and_canvas(self, plot_row_idx, plot_col_idx_in_layout):
        # Uses self.plot_layout (QGridLayout from UIMixin)
        fig = Figure(facecolor=self.themes[self.current_theme]['figure_bg'], constrained_layout=True)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self.plot_widget) # Parent is plot_widget for toolbar

        # Add to the main plot layout grid
        # Each pair of plots (grouped + colcol) takes two rows in the grid: one for toolbars, one for canvases
        # Grouped plots are in column 0, ColCol plots in column 1
        self.plot_layout.addWidget(toolbar, plot_row_idx * 2, plot_col_idx_in_layout)
        self.plot_layout.addWidget(canvas, plot_row_idx * 2 + 1, plot_col_idx_in_layout)
        return fig, canvas

    def _plot_data_vs_index(self, fig, data_series, label, plot_row_idx, data_prefix, subplot_index, sharex=None):
        # subplot_index: 0 for top (X vs Index), 1 for bottom (Y vs Index)
        ax = fig.add_subplot(2, 1, subplot_index + 1, sharex=sharex if sharex else None)
        
        if data_series.empty:
            self._setup_basic_plot_appearance(ax, label + " (No Data)", 'Index' if subplot_index == 1 else '', 'Value')
            self._apply_theme_to_single_axis(ax, self.themes[self.current_theme])
            return ax, None

        line_artist = None
        indices = self.data.index.values # Use main dataframe's index for consistency
        values = data_series.values
        
        defect_proba_col = f'defect_proba_{data_prefix}'
        has_defect_proba = defect_proba_col in self.data.columns and not self.data[defect_proba_col].isnull().all()

        if has_defect_proba:
            probabilities = self.data[defect_proba_col].fillna(0).values
            # Ensure probabilities align with the current data_series if its index is different (e.g. after slicing)
            # For simplicity, assume data_series uses the same index range as self.data for probabilities
            
            segments = [[(indices[i], values[i]), (indices[i+1], values[i+1])] for i in range(len(indices)-1)]
            # Ensure probabilities correspond to segments, so len(probabilities)-1 if using point probabilities for segment start
            color_values_for_lc = [probabilities[i] for i in range(len(indices)-1)]


            if not segments: # Fallback if only one data point
                line_artist, = ax.plot(indices, values, color=self.themes[self.current_theme]['line_color'], linewidth=1)
            else:
                theme_line_color = self.themes[self.current_theme]['line_color']
                defect_color = 'red' # Standard defect color
                cmap_name = f'defect_cmap_grouped_{plot_row_idx}_{subplot_index}_{theme_line_color.replace("#","")}'
                cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, [theme_line_color, defect_color], N=256)
                
                lc = LineCollection(segments, cmap=cmap, linewidth=1.5, zorder=10)
                lc.set_array(np.array(color_values_for_lc))
                lc.set_norm(mcolors.Normalize(vmin=0, vmax=1))
                line_artist = ax.add_collection(lc)
            
            ax.set_xlim(indices.min(), indices.max())
            ax.set_ylim(values.min(), values.max()) # Autoscale Y

        else: # No defect probability
            line_artist, = ax.plot(indices, values, label=label, color=self.themes[self.current_theme]['line_color'], linewidth=1)
            # ax.autoscale(enable=True, axis='both', tight=True) # autoscale if no defect proba

        self._setup_basic_plot_appearance(ax, label if subplot_index == 0 else '',  # Title for top plot only
                                       'Index' if subplot_index == 1 else '',   # X-label for bottom plot only
                                       'Value') # Y-label for both
        if subplot_index == 0: # Top plot (X vs Index)
            ax.tick_params(labelbottom=False, bottom=False) # Hide x-axis labels and ticks

        self._apply_theme_to_single_axis(ax, self.themes[self.current_theme])
        return ax, line_artist

    def _plot_data_col_vs_col(self, fig, x_data_series, y_data_series, x_label, y_label, plot_row_idx, data_prefix):
        ax = fig.add_subplot(1, 1, 1)
        
        if x_data_series.empty or y_data_series.empty:
            self._setup_basic_plot_appearance(ax, f'{x_label} vs {y_label} (No Data)', x_label, y_label)
            self._apply_theme_to_single_axis(ax, self.themes[self.current_theme])
            return ax, None

        line_artist = None
        defect_proba_col = f'defect_proba_{data_prefix}'
        has_defect_proba = defect_proba_col in self.data.columns and not self.data[defect_proba_col].isnull().all()

        # Ensure data alignment for plotting, especially if series come from different operations
        common_index = x_data_series.index.intersection(y_data_series.index)
        current_x_data = x_data_series.loc[common_index].values
        current_y_data = y_data_series.loc[common_index].values
        
        if len(current_x_data) == 0: # No common data points
            self._setup_basic_plot_appearance(ax, f'{x_label} vs {y_label} (No Data)', x_label, y_label)
            self._apply_theme_to_single_axis(ax, self.themes[self.current_theme])
            return ax, None


        if has_defect_proba:
            # Align probabilities with the common_index of x and y data
            probabilities = self.data.loc[common_index, defect_proba_col].fillna(0).values
            
            segments = [[(current_x_data[i], current_y_data[i]), (current_x_data[i+1], current_y_data[i+1])] for i in range(len(current_x_data)-1)]
            color_values_for_lc = [probabilities[i] for i in range(len(current_x_data)-1)]

            if not segments: # Fallback for single point or no segments
                 line_artist, = ax.plot(current_x_data, current_y_data, color=self.themes[self.current_theme]['line_color'])
            else:
                theme_line_color = self.themes[self.current_theme]['line_color']
                defect_color = 'red'
                cmap_name = f'col_vs_col_defect_cmap_{plot_row_idx}_{theme_line_color.replace("#","")}'
                cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, [theme_line_color, defect_color], N=256)
                
                lc = LineCollection(segments, cmap=cmap, linewidth=1.5, zorder=10)
                lc.set_array(np.array(color_values_for_lc))
                lc.set_norm(mcolors.Normalize(vmin=0, vmax=1))
                line_artist = ax.add_collection(lc)
            
            if len(current_x_data) > 0:
                ax.set_xlim(current_x_data.min(), current_x_data.max())
                ax.set_ylim(current_y_data.min(), current_y_data.max())

        else: # No defect probability
            line_artist, = ax.plot(current_x_data, current_y_data, color=self.themes[self.current_theme]['line_color'])
            if len(current_x_data) > 0: # Ensure data exists before min/max
                ax.set_xlim(current_x_data.min(), current_x_data.max())
                ax.set_ylim(current_y_data.min(), current_y_data.max())
            # ax.autoscale(enable=True, axis='both', tight=True)


        self._setup_basic_plot_appearance(ax, f'{x_label} vs {y_label}', x_label, y_label)
        self._apply_theme_to_single_axis(ax, self.themes[self.current_theme])
        return ax, line_artist

    def _setup_basic_plot_appearance(self, ax, title, xlabel='', ylabel=''):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        # Grid is applied by _apply_theme_to_single_axis
        ax.tick_params(axis='both', which='major', labelsize=8)

    def _create_span_selector_for_axis(self, ax):
        # self.on_span_select is a method in ActionsMixin
        return SpanSelector(ax, self.on_span_select, 'horizontal', useblit=False, # useblit=False often more stable
                            props=dict(alpha=0.3, facecolor=self.themes[self.current_theme]['span_color']),
                            interactive=True, drag_from_anywhere=True)

    def _synchronize_x_axes(self, ax1, ax2):
        if ax1 and ax2:
            # self.handle_xlim_change_for_grouped_plots is a method in ActionsMixin
            ax1.callbacks.connect('xlim_changed', self.handle_xlim_change_for_grouped_plots)
            ax2.callbacks.connect('xlim_changed', self.handle_xlim_change_for_grouped_plots)
    
    def clear_all_plots(self):
        """Clears all plot figures, canvases, axes, and removes widgets from plot_layout."""
        if hasattr(self, 'plot_layout') and self.plot_layout is not None:
            while self.plot_layout.count():
                item = self.plot_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater() # Important for C++ object cleanup

        # Reset plot object storage arrays
        # self.num_pairs should be accurate here, or set to 0 if no data
        current_num_pairs = self.num_pairs if hasattr(self, 'num_pairs') else 0

        self.figures = np.empty((current_num_pairs, 2), dtype=object)
        self.canvases = np.empty((current_num_pairs, 2), dtype=object)
        self.axes_grouped = [[] for _ in range(current_num_pairs)]
        self.axes_colcol = [[] for _ in range(current_num_pairs)]
        self.span_selectors = np.empty((current_num_pairs, 2), dtype=object)
        self.plot_lines_grouped = np.empty((current_num_pairs, 2), dtype=object)
        self.plot_lines_colcol = np.empty((current_num_pairs, 1), dtype=object)
        self.selection_lines = [[[] for _ in range(2)] for _ in range(current_num_pairs)]

        # If num_pairs itself should be reset (e.g. no data loaded), it should be done
        # by the data loading logic. Here, we just clear based on current num_pairs.

    def update_single_plot_pair(self, plot_row_idx):
        """Updates both grouped and col-col plots for a specific data pair index."""
        if self.data.empty or plot_row_idx >= self.num_pairs or \
           plot_row_idx >= len(self.current_plot_prefixes) or \
           plot_row_idx >= self.figures.shape[0]:
            return

        data_prefix = self.current_plot_prefixes[plot_row_idx]
        x_col_name = f"{data_prefix}_X"
        y_col_name = f"{data_prefix}_Y"

        if x_col_name not in self.data.columns or y_col_name not in self.data.columns:
            return # Cannot update if data is missing

        # --- Update Grouped Plot (X vs Index, Y vs Index) ---
        fig_grouped = self.figures[plot_row_idx, 0]
        canvas_grouped = self.canvases[plot_row_idx, 0]

        if fig_grouped and canvas_grouped and plot_row_idx < len(self.axes_grouped):
            # Clear existing axes from the figure
            for ax_g in self.axes_grouped[plot_row_idx]:
                if ax_g: fig_grouped.delaxes(ax_g)
            
            # Re-plot X vs Index
            ax1_new, line1_new = self._plot_data_vs_index(fig_grouped, self.data[x_col_name],
                                                         f'{x_col_name} vs Index', plot_row_idx, data_prefix, 0)
            # Re-plot Y vs Index
            ax2_new, line2_new = self._plot_data_vs_index(fig_grouped, self.data[y_col_name],
                                                         f'{y_col_name} vs Index', plot_row_idx, data_prefix, 1, sharex=ax1_new)
            
            self.axes_grouped[plot_row_idx] = [ax1_new, ax2_new]
            self.plot_lines_grouped[plot_row_idx, 0] = line1_new
            self.plot_lines_grouped[plot_row_idx, 1] = line2_new

            # Re-create span selectors and synchronize axes
            if ax1_new: self.span_selectors[plot_row_idx, 0] = self._create_span_selector_for_axis(ax1_new)
            else: self.span_selectors[plot_row_idx, 0] = None # Ensure it's None if axis is None

            if ax2_new: self.span_selectors[plot_row_idx, 1] = self._create_span_selector_for_axis(ax2_new)
            else: self.span_selectors[plot_row_idx, 1] = None

            if ax1_new and ax2_new: self._synchronize_x_axes(ax1_new, ax2_new)
            
            try: fig_grouped.tight_layout() # Adjust layout
            except Exception: pass # Sometimes tight_layout can fail with rapidly changing plots
            canvas_grouped.draw_idle()

        # --- Update Column vs Column Plot (X vs Y) ---
        fig_colcol = self.figures[plot_row_idx, 1]
        canvas_colcol = self.canvases[plot_row_idx, 1]

        if fig_colcol and canvas_colcol and plot_row_idx < len(self.axes_colcol) and self.axes_colcol[plot_row_idx]:
            ax_cc_old = self.axes_colcol[plot_row_idx][0] # Assuming one axis
            if ax_cc_old: fig_colcol.delaxes(ax_cc_old)

            ax_cc_new, line_cc_new = self._plot_data_col_vs_col(fig_colcol, self.data[x_col_name], self.data[y_col_name],
                                                              x_col_name, y_col_name, plot_row_idx, data_prefix)
            self.axes_colcol[plot_row_idx] = [ax_cc_new]
            self.plot_lines_colcol[plot_row_idx, 0] = line_cc_new
            
            try: fig_colcol.tight_layout()
            except Exception: pass
            canvas_colcol.draw_idle()
        
        # After updating plots, re-apply the theme to ensure consistency for new elements
        self.apply_theme_to_all_plots()


    def refresh_all_plots_content(self):
        if self.data.empty or self.num_pairs == 0:
            self.clear_all_plots() # Clear if no data
            return
        
        for plot_idx in range(self.num_pairs):
            if plot_idx < len(self.current_plot_prefixes): # Ensure prefix exists
                self.update_single_plot_pair(plot_idx)
        
        self.apply_theme_to_all_plots() # Ensure theme is consistent

    def clear_selection_lines_for_pair_subplot(self, pair_idx, subplot_idx_in_grouped_plot):
        # subplot_idx_in_grouped_plot is 0 for X vs Index, 1 for Y vs Index
        if pair_idx < len(self.selection_lines) and \
           subplot_idx_in_grouped_plot < len(self.selection_lines[pair_idx]):
            for line_artist in self.selection_lines[pair_idx][subplot_idx_in_grouped_plot]:
                if line_artist and line_artist.axes: # Check if line exists and is part of an axes
                    try:
                        line_artist.remove()
                    except ValueError: # May already be removed or not part of axes
                        pass
            self.selection_lines[pair_idx][subplot_idx_in_grouped_plot] = [] # Reset the list

    def draw_vertical_selection_lines(self, pair_idx, subplot_idx_in_grouped_plot, axis_obj, xmin_val, xmax_val):
        theme_selection_color = self.themes[self.current_theme]['selection_color']
        line1 = axis_obj.axvline(x=xmin_val, color=theme_selection_color, linestyle='--', linewidth=1.5, zorder=20)
        line2 = axis_obj.axvline(x=xmax_val, color=theme_selection_color, linestyle='--', linewidth=1.5, zorder=20)
        
        if pair_idx < len(self.selection_lines) and \
           subplot_idx_in_grouped_plot < len(self.selection_lines[pair_idx]):
            self.selection_lines[pair_idx][subplot_idx_in_grouped_plot].extend([line1, line2])


    def reset_all_span_selectors(self):
        """Resets span selectors, typically after a precise selection."""
        if not hasattr(self, 'span_selectors') or self.span_selectors is None:
            return

        for r in range(self.num_pairs):
            for c_idx in range(2): # 0 for X-plot, 1 for Y-plot in grouped view
                if r < self.span_selectors.shape[0] and c_idx < self.span_selectors.shape[1]:
                    span = self.span_selectors[r, c_idx]
                    if span is not None:
                        # Disconnect old span selector's events
                        if hasattr(span, 'disconnect_events'): # older mpl
                             span.disconnect_events()
                        elif hasattr(span, 'observers') and isinstance(span.observers, dict): # newer mpl with explicit observers
                            for obs_id in list(span.observers.keys()):
                                span.ax.callbacks.disconnect(obs_id)
                            span.observers.clear()


                        # Try to remove visual elements if they exist
                        if hasattr(span, 'artists'): # older mpl
                            for artist in span.artists: artist.remove()
                        if hasattr(span, '_selection_artists'): # newer mpl
                            for artist in span._selection_artists: artist.remove()
                        if hasattr(span, 'rect') and span.rect: span.rect.remove() # common
                        if hasattr(span, 'patch') and span.patch: span.patch.remove() # common for some versions
                        
                        # Clear blitting background
                        span.background = None
                        
                        # Recreate the span selector on the correct axis
                        if r < len(self.axes_grouped) and c_idx < len(self.axes_grouped[r]):
                            ax_for_span = self.axes_grouped[r][c_idx]
                            if ax_for_span:
                                self.span_selectors[r, c_idx] = self._create_span_selector_for_axis(ax_for_span)
                            else:
                                self.span_selectors[r, c_idx] = None
                        else:
                             self.span_selectors[r, c_idx] = None


        # Redraw relevant canvases
        for r_idx in range(self.num_pairs):
            if r_idx < self.canvases.shape[0] and self.canvases[r_idx, 0]:
                self.canvases[r_idx, 0].draw_idle()
