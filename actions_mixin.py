import numpy as np
import pandas as pd
import random
from PyQt5.QtWidgets import QInputDialog, QMessageBox
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut

from phase import get_amp_phase # External module
from matplotlib.collections import LineCollection # For type checking in plot updates
import matplotlib.colors as mcolors # For colormaps

class ActionsMixin:

    def setup_event_shortcuts(self):
        """Sets up keyboard shortcuts for common actions."""
        QShortcut(QKeySequence("Ctrl+C"), self).activated.connect(self.trigger_copy_data)
        QShortcut(QKeySequence("Ctrl+V"), self).activated.connect(self.trigger_paste_data)
        QShortcut(QKeySequence("Ctrl+Shift+V"), self).activated.connect(self.show_randomize_paste_ui) # From UIMixin


    def on_span_select(self, xmin, xmax):
        """Callback for the SpanSelector, handles data selection and plot updates."""
        if self.data.empty or not isinstance(xmin, (int, float)) or not isinstance(xmax, (int, float)):
            return

        # Ensure xmin and xmax are valid indices within the data
        # Convert float to int; matplotlib span can give float values.
        xmin_idx = int(np.round(xmin))
        xmax_idx = int(np.round(xmax))

        if self.data.index.empty: return # No data index to select from

        min_data_idx = self.data.index.min()
        max_data_idx = self.data.index.max()

        # Clamp selection to data boundaries
        xmin_idx = max(min_data_idx, xmin_idx)
        xmax_idx = min(max_data_idx, xmax_idx)
        
        if xmin_idx >= xmax_idx: # Invalid or zero-length selection
            self.copied_data = pd.DataFrame() # Clear copied data
            # Clear visual indicators of selection (green lines, specific ColCol plot update)
            for pair_idx_clear in range(self.num_pairs):
                 if pair_idx_clear < len(self.axes_grouped) and self.axes_grouped[pair_idx_clear]:
                    for subplot_idx_clear, _ in enumerate(self.axes_grouped[pair_idx_clear]):
                        self.clear_selection_lines_for_pair_subplot(pair_idx_clear, subplot_idx_clear) # From PlottingMixin
                    if self.canvases[pair_idx_clear, 0]: self.canvases[pair_idx_clear, 0].draw_idle()
                 
                 # For ColCol plot, revert to showing full data or clear if appropriate
                 # This might mean replotting the full data for that specific ColCol plot
                 if pair_idx_clear < self.figures.shape[0] and self.figures[pair_idx_clear,1]:
                     self.update_single_plot_pair(pair_idx_clear) # Replots based on full data or current state

            # Reset phase/amplitude display as no valid selection is active
            self.amplitude.fill(np.nan)
            self.phase.fill(np.nan)
            self.update_status_bar_text() # From UIMixin
            return

        # Valid selection range
        index_mask = (self.data.index >= xmin_idx) & (self.data.index <= xmax_idx)
        self.copied_data = self.data.loc[index_mask].copy()

        # Update plots to reflect the new selection
        for pair_idx_update in range(self.num_pairs):
            self._update_plots_for_selection(pair_idx_update, index_mask, xmin_idx, xmax_idx)

        selected_indices = self.data.index[index_mask]
        if not selected_indices.empty:
            self._calculate_and_update_phases_for_indices(selected_indices)
        else: # Should not happen if xmin_idx < xmax_idx but good to handle
            self.amplitude.fill(np.nan)
            self.phase.fill(np.nan)
            self.update_status_bar_text()


    def _update_plots_for_selection(self, pair_idx, index_mask, xmin_val_selected, xmax_val_selected):
        """Updates visual elements on plots for a given pair after a selection."""
        if pair_idx >= len(self.current_plot_prefixes): return

        data_prefix = self.current_plot_prefixes[pair_idx]
        x_col = f"{data_prefix}_X"
        y_col = f"{data_prefix}_Y"

        if x_col not in self.data.columns or y_col not in self.data.columns: return

        # Update Column-vs-Column plot to show only selected data
        if pair_idx < self.figures.shape[0] and self.figures[pair_idx, 1] and \
           pair_idx < len(self.axes_colcol) and self.axes_colcol[pair_idx] and self.axes_colcol[pair_idx][0]:
            
            ax_cc = self.axes_colcol[pair_idx][0]
            ax_cc.clear() # Clear previous content

            selected_subset_df = self.data.loc[index_mask]
            if not selected_subset_df.empty:
                selected_x_data = selected_subset_df[x_col]
                selected_y_data = selected_subset_df[y_col]
                
                # Check for defect probability for the selected subset
                defect_proba_col_name = f'defect_proba_{data_prefix}'
                has_defect_proba_selected = defect_proba_col_name in selected_subset_df.columns and \
                                            not selected_subset_df[defect_proba_col_name].isnull().all()

                line_artist_sel = None
                if has_defect_proba_selected and not selected_x_data.empty:
                    probabilities_sel = selected_subset_df.loc[selected_x_data.index, defect_proba_col_name].fillna(0).values
                    current_x_sel = selected_x_data.values
                    current_y_sel = selected_y_data.values
                    
                    segments_sel = [[(current_x_sel[i], current_y_sel[i]), (current_x_sel[i+1], current_y_sel[i+1])] for i in range(len(current_x_sel)-1)]
                    colors_for_lc_sel = [probabilities_sel[i] for i in range(len(current_x_sel)-1)]

                    if segments_sel:
                        theme_line_color_sel = self.themes[self.current_theme]['line_color']
                        defect_color_sel = 'red' # Consistent defect color
                        cmap_name_sel = f'selected_col_vs_col_cmap_{pair_idx}_{theme_line_color_sel.replace("#","")}'
                        cmap_sel = mcolors.LinearSegmentedColormap.from_list(cmap_name_sel, [theme_line_color_sel, defect_color_sel], N=256)
                        
                        lc_sel = LineCollection(segments_sel, cmap=cmap_sel, linewidth=1.5, zorder=10)
                        lc_sel.set_array(np.array(colors_for_lc_sel))
                        lc_sel.set_norm(mcolors.Normalize(vmin=0, vmax=1))
                        ax_cc.add_collection(lc_sel)
                    elif len(current_x_sel) > 0 : # Single point or no segments, plot as line
                         ax_cc.plot(current_x_sel, current_y_sel, color=self.themes[self.current_theme]['line_color'], linewidth=1)

                elif not selected_x_data.empty: # No defect probability, or data too short for segments
                    ax_cc.plot(selected_x_data, selected_y_data, color=self.themes[self.current_theme]['line_color'], linewidth=1)

                # Set limits for the col-col plot based on selected data
                if not selected_x_data.empty:
                    ax_cc.set_xlim(selected_x_data.min(), selected_x_data.max())
                    ax_cc.set_ylim(selected_y_data.min(), selected_y_data.max())
                else: # Fallback to full data limits if selection somehow yields nothing plottable
                    ax_cc.set_xlim(self.data[x_col].min(), self.data[x_col].max())
                    ax_cc.set_ylim(self.data[y_col].min(), self.data[y_col].max())

            else: # If selected_subset_df is empty, set to full data limits
                ax_cc.set_xlim(self.data[x_col].min(), self.data[x_col].max())
                ax_cc.set_ylim(self.data[y_col].min(), self.data[y_col].max())

            self._setup_basic_plot_appearance(ax_cc, f'Selected: {x_col} vs {y_col}', x_col, y_col) # From PlottingMixin
            self._apply_theme_to_single_axis(ax_cc, self.themes[self.current_theme]) # From PlottingMixin
            
            if self.canvases[pair_idx, 1]: self.canvases[pair_idx, 1].draw_idle()

        # Update vertical selection lines on grouped plots
        if pair_idx < len(self.axes_grouped) and self.axes_grouped[pair_idx]:
            for subplot_idx_grouped, ax_grouped in enumerate(self.axes_grouped[pair_idx]):
                if ax_grouped is None: continue
                self.clear_selection_lines_for_pair_subplot(pair_idx, subplot_idx_grouped) # From PlottingMixin
                if not self.copied_data.empty: # Draw new lines if there's a selection
                    self.draw_vertical_selection_lines(pair_idx, subplot_idx_grouped, ax_grouped, 
                                                       xmin_val_selected, xmax_val_selected) # From PlottingMixin
            if self.canvases[pair_idx, 0]: self.canvases[pair_idx, 0].draw_idle()


    def _calculate_and_update_phases_for_indices(self, data_indices):
        """Computes phases for the given data indices and updates the display."""
        if data_indices.empty:
            self.amplitude.fill(np.nan)
            self.phase.fill(np.nan)
        else:
            phase_results = self._compute_phase_amplitude_for_all_pairs(self.data, data_indices)
            # Ensure amplitude and phase arrays are correctly sized for num_pairs
            self._resize_phase_amplitude_arrays() # from DataMixin, ensures correct size
            
            for p_idx, (amp, ph) in phase_results.items():
                if p_idx < self.num_pairs: # Ensure index is within bounds
                    self.amplitude[p_idx] = amp
                    self.phase[p_idx] = ph
        
        self.update_status_bar_text() # From UIMixin

    def _compute_phase_amplitude_for_all_pairs(self, dataframe, df_indices_for_calc):
        """Helper to compute amplitude and phase for all relevant column pairs over given indices."""
        phase_results_map = {} # Store as {pair_index: (amplitude, phase)}
        if df_indices_for_calc.empty or not self.current_plot_prefixes:
            return phase_results_map

        for pair_idx, data_prefix in enumerate(self.current_plot_prefixes):
            if pair_idx >= self.num_pairs: break # Only compute for active pairs

            x_col = f"{data_prefix}_X"
            y_col = f"{data_prefix}_Y"

            if x_col not in dataframe.columns or y_col not in dataframe.columns:
                phase_results_map[pair_idx] = (np.nan, np.nan)
                continue

            # Ensure indices are valid for the dataframe
            valid_indices_for_pair = dataframe.index.intersection(df_indices_for_calc)
            if valid_indices_for_pair.empty:
                phase_results_map[pair_idx] = (np.nan, np.nan)
                continue
            
            x_values = dataframe.loc[valid_indices_for_pair, x_col].values
            y_values = dataframe.loc[valid_indices_for_pair, y_col].values

            if len(x_values) < 2: # Need at least two points for phase calculation
                phase_results_map[pair_idx] = (np.nan, np.nan)
                continue
            
            # Prepare data for get_amp_phase: list of (x,y) tuples
            xy_data_list = list(zip(x_values, y_values))
            
            # Determine position and width for get_amp_phase
            # Example: use middle of the selection and half its width
            position_param = len(xy_data_list) // 2
            width_param = max(1, len(xy_data_list) // 2) # Ensure width is at least 1
            
            if width_param == 0 : # Should be caught by len(x_values) < 2, but defensive
                phase_results_map[pair_idx] = (np.nan, np.nan)
                continue

            try:
                amp, ph = get_amp_phase(xy_data_list, position_param, width_param, 
                                        self.is_calibrated, self.calibration)
                phase_results_map[pair_idx] = (amp, ph)
            except Exception as e:
                print(f"Error computing phase for pair '{data_prefix}': {e}")
                phase_results_map[pair_idx] = (np.nan, np.nan)
        
        return phase_results_map

    def trigger_copy_data(self):
        if self.copied_data.empty:
            QMessageBox.warning(self, "Copy Data", "No data selected to copy. Use mouse on plots or 'Precise Selection'.")
            return False # Indicate failure

        # Mark the original data region corresponding to this copy
        copied_indices = self.copied_data.index
        self._reset_markers_for_previous_copies() # Clear old '-1' markers

        if 'slice_number' not in self.data.columns:
            self.data['slice_number'] = np.nan # Add column if it doesn't exist
        
        # Mark current copy operation with -1 (temporary marker for "just copied")
        self.data.loc[copied_indices, 'slice_number'] = -1 
        
        # Clear any existing phase values for the copied region in the main dataframe
        self._clear_phase_columns_for_indices(copied_indices)
        
        # Optionally, re-calculate and store phases for the newly copied slice in the main dataframe
        # This is if you want the 'copied' section in self.data to have its phases stored if not already
        # phase_info_for_copied_slice = self._compute_phase_amplitude_for_all_pairs(self.data, copied_indices)
        # self._assign_phase_data_to_columns(copied_indices, phase_info_for_copied_slice)
        
        QMessageBox.information(self, "Copy Data", f"{len(copied_indices)} rows copied to buffer.")
        return True # Indicate success

    def _reset_markers_for_previous_copies(self):
        if 'slice_number' in self.data.columns:
            # Find rows previously marked as copied (-1) and reset them to NaN (or original state if known)
            previously_copied_mask = self.data['slice_number'] == -1
            if previously_copied_mask.any():
                self.data.loc[previously_copied_mask, 'slice_number'] = np.nan # Or some other default
                # Also clear phase data for these reset regions if appropriate
                self._clear_phase_columns_for_indices(self.data.index[previously_copied_mask])

    def _clear_phase_columns_for_indices(self, indices_to_clear):
        if indices_to_clear.empty or not self.current_plot_prefixes:
            return
        for pair_idx, data_prefix in enumerate(self.current_plot_prefixes):
            if pair_idx >= self.num_pairs: break
            phase_col_name = f'phase_{data_prefix}' # Assuming a naming convention
            if phase_col_name in self.data.columns:
                self.data.loc[indices_to_clear, phase_col_name] = np.nan

    def _assign_phase_data_to_columns(self, indices_to_assign, phase_info_map):
        # phase_info_map is {pair_idx: (amplitude, phase)}
        if indices_to_assign.empty or not phase_info_map or not self.current_plot_prefixes:
            return

        for pair_idx, (amp, ph_val) in phase_info_map.items():
            if pair_idx < len(self.current_plot_prefixes) and pair_idx < self.num_pairs:
                data_prefix = self.current_plot_prefixes[pair_idx]
                phase_col_name = f'phase_{data_prefix}'
                # amplitude_col_name = f'amplitude_{data_prefix}' # If storing amplitude too

                if phase_col_name not in self.data.columns:
                    self.data[phase_col_name] = np.nan # Add column
                # if amplitude_col_name not in self.data.columns:
                #     self.data[amplitude_col_name] = np.nan

                self.data.loc[indices_to_assign, phase_col_name] = ph_val
                # self.data.loc[indices_to_assign, amplitude_col_name] = amp
        
        # After assigning, status bar might need update if this affects overall view
        # self.update_status_bar_text() # Or rely on selection change to do this.

    def trigger_paste_data(self, paste_position=None, scale_factor=None):
        if self.copied_data.empty:
            QMessageBox.warning(self, "Paste Data", "No data in copy buffer to paste.")
            return False

        # Determine paste position
        actual_paste_start_index = self._get_paste_start_index_from_user(paste_position)
        if actual_paste_start_index is None: return False # User cancelled or invalid input

        # Determine scale factor
        actual_scale_factor = self._get_scale_factor_from_user(scale_factor)
        if actual_scale_factor is None: return False # User cancelled or invalid input

        # Perform the paste operation
        success = self._execute_paste_operation(actual_paste_start_index, actual_scale_factor)
        if success:
            self.refresh_all_plots_content() # From PlottingMixin
            QMessageBox.information(self, "Paste Data", "Data pasted successfully.")
        # Errors handled within _execute_paste_operation
        return success

    def _get_paste_start_index_from_user(self, predefined_position):
        if predefined_position is not None: # Used by randomized paste
            if 0 <= predefined_position < len(self.data):
                return predefined_position
            else: # Should not happen if available indices are calculated correctly
                QMessageBox.warning(self, 'Paste Position Error',
                                    f"Internal Error: Provided paste position {predefined_position} is out of bounds (0-{len(self.data)-1}).")
                return None
        
        # Interactive input from user
        max_paste_idx = len(self.data) - 1 if not self.data.empty else 0
        if max_paste_idx < 0 : max_paste_idx = 0 # if data is empty, can only paste at 0 (conceptually)

        pos_val, ok = QInputDialog.getInt(self, 'Paste Position', 
                                          f'Enter start index for paste (0 to {max_paste_idx}):',
                                          min=0, max=max_paste_idx)
        return pos_val if ok else None

    def _get_scale_factor_from_user(self, predefined_scale):
        if predefined_scale is not None: # Used by randomized paste
            return float(predefined_scale)
        
        # Interactive input
        scale_val, ok = QInputDialog.getDouble(self, 'Scale Factor', 
                                               'Enter scale factor for pasted data:',
                                               value=1.0, decimals=3, min=-1e6, max=1e6)
        return scale_val if ok else None

    def _execute_paste_operation(self, paste_start_index, scale_factor_val):
        if self.copied_data.empty: return False # Should be checked before calling

        # Select only numeric columns from copied_data for scaling
        numeric_cols_copied = self.copied_data.select_dtypes(include=np.number).columns
        segment_to_paste = self.copied_data.copy()
        segment_to_paste[numeric_cols_copied] *= scale_factor_val

        paste_length = len(segment_to_paste)
        
        # Determine the actual end index in the target dataframe, avoid overflow
        paste_end_exclusive = min(paste_start_index + paste_length, len(self.data))
        
        # Target indices in the main dataframe
        target_indices = self.data.index[paste_start_index : paste_end_exclusive]

        if target_indices.empty:
            QMessageBox.warning(self, "Paste Error", "Invalid paste range or target data too short.")
            return False
        
        # Trim segment_to_paste if it's longer than the available target space
        actual_segment_to_paste_df = segment_to_paste.iloc[:len(target_indices)]
        actual_segment_to_paste_df.index = target_indices # Align index with target

        # Check for overlap with existing "pasted" (not "copied") data
        if self._check_for_paste_overlap(target_indices):
            # Message is shown by _check_for_paste_overlap
            return False 
        
        # Add pasted data to the main dataframe for numeric columns
        # Ensure we only try to add to columns that exist in self.data
        cols_to_update_in_main_data = actual_segment_to_paste_df.columns.intersection(self.data.columns)
        numeric_cols_to_update = self.data[cols_to_update_in_main_data].select_dtypes(include=np.number).columns
        
        # Use .add() method for robust addition, or direct assignment if replacing
        # self.data.loc[target_indices, numeric_cols_to_update] += actual_segment_to_paste_df[numeric_cols_to_update].values
        # For "interference", it's usually an addition. If it's a replacement, use direct assignment.
        # Assuming addition:
        for col in numeric_cols_to_update:
            if col in actual_segment_to_paste_df:
                 self.data.loc[target_indices, col] = self.data.loc[target_indices, col].values + actual_segment_to_paste_df[col].values


        # Mark this pasted region and calculate its phases
        self._mark_data_as_pasted(target_indices)
        self._recalculate_and_assign_phases_for_pasted_region(target_indices)
        
        return True

    def _check_for_paste_overlap(self, target_indices_for_paste):
        if 'slice_number' in self.data.columns and not target_indices_for_paste.empty:
            # Check if any part of the target range has a slice_number > 0 (already a finalized paste)
            if (self.data.loc[target_indices_for_paste, 'slice_number'] > 0).any():
                QMessageBox.warning(self, "Paste Conflict", 
                                    "The target paste area overlaps with a previously pasted segment. "
                                    "Pasting here would overwrite or interfere with existing modifications. "
                                    "Operation aborted.")
                return True # Overlap detected
        return False # No overlap

    def _mark_data_as_pasted(self, pasted_data_indices):
        if 'slice_number' not in self.data.columns:
            self.data['slice_number'] = np.nan # Initialize if not present
        
        self.paste_count += 1 # Increment global paste counter
        self.data.loc[pasted_data_indices, 'slice_number'] = self.paste_count # Mark with current paste ID

    def _recalculate_and_assign_phases_for_pasted_region(self, pasted_data_indices):
        if pasted_data_indices.empty: return
        
        phase_info_for_pasted = self._compute_phase_amplitude_for_all_pairs(self.data, pasted_data_indices)
        self._assign_phase_data_to_columns(pasted_data_indices, phase_info_for_pasted)


    def trigger_randomize_paste(self):
        """Handles the click from the 'Randomize and Paste' button in the UI."""
        if self.copied_data.empty:
            QMessageBox.warning(self, "Randomize Paste", "No data in copy buffer. Please copy a selection first.")
            return

        try:
            # Retrieve values from the UI widgets stored in self.rand_paste_widgets (UIMixin)
            params = {k: w.text() if isinstance(w, QLineEdit) else w.currentText() 
                      for k, w in self.rand_paste_widgets.items()}

            from_idx = int(params['from_index'])
            to_idx = int(params['to_index'])
            num_pastes_to_try = int(params['num_pastes'])
            scale_type = params['scale_type']

            if self.data.empty: raise ValueError("Main data is empty. Cannot determine paste range.")
            
            min_data_idx, max_data_idx = self.data.index.min(), self.data.index.max()
            if not (min_data_idx <= from_idx <= max_data_idx and 
                    min_data_idx <= to_idx <= max_data_idx and 
                    from_idx <= to_idx):
                raise ValueError(f"Index range for randomization [{from_idx}-{to_idx}] is invalid "
                                 f"for data range [{min_data_idx}-{max_data_idx}].")

            if num_pastes_to_try <= 0: raise ValueError("Number of pastes must be greater than 0.")

            constant_scale_val = None
            random_scale_min_val, random_scale_max_val = None, None

            if scale_type == 'Constant':
                constant_scale_val = float(params['constant_scale'])
            else: # Random
                random_scale_min_val = float(params['random_scale_min'])
                random_scale_max_val = float(params['random_scale_max'])
                if random_scale_min_val > random_scale_max_val:
                    raise ValueError("Random scale min cannot be greater than random scale max.")
            
            # Store current options as last used for next time
            self.last_random_paste_options.update(params)

            # Execute the randomization and pasting
            self._perform_randomized_pastes(from_idx, to_idx, num_pastes_to_try, scale_type,
                                           constant_scale_val, random_scale_min_val, random_scale_max_val)

        except ValueError as ve:
            QMessageBox.warning(self, "Randomize Paste Error", str(ve))
        except Exception as e: # Catch any other unexpected errors
            QMessageBox.critical(self, "Randomize Paste Error", f"An unexpected error occurred: {e}")


    def _perform_randomized_pastes(self, range_min_idx, range_max_idx, num_pastes_requested, 
                                  scale_type_str, const_scale=None, rand_min_scale=None, rand_max_scale=None):
        
        if self.copied_data.empty: return # Should be checked earlier

        length_of_copied_segment = len(self.copied_data)
        if length_of_copied_segment == 0: return # Nothing to paste

        # Find all possible non-overlapping start positions within the specified range
        available_start_indices = self._find_available_paste_starts(range_min_idx, range_max_idx, length_of_copied_segment)

        if not available_start_indices:
            QMessageBox.information(self, "Randomize Paste", 
                                    "No non-overlapping positions available in the specified range to paste the copied segment.")
            return

        num_pastes_to_perform = min(num_pastes_requested, len(available_start_indices))
        if num_pastes_to_perform < num_pastes_requested:
            QMessageBox.warning(self, "Randomize Paste", 
                                f"Could only find {num_pastes_to_perform} suitable positions out of {num_pastes_requested} requested.")

        # Randomly select start indices from the available ones
        selected_start_indices = random.sample(available_start_indices, num_pastes_to_perform)
        
        successful_pastes_count = 0
        for start_idx in selected_start_indices:
            current_scale_factor = const_scale if scale_type_str == 'Constant' else random.uniform(rand_min_scale, rand_max_scale)
            if self._execute_paste_operation(start_idx, current_scale_factor):
                successful_pastes_count += 1
        
        QMessageBox.information(self, "Randomize Paste Complete", 
                                f"Completed {successful_pastes_count} out of {num_pastes_requested} requested pastes.")
        
        if successful_pastes_count > 0:
            self.refresh_all_plots_content() # Update all plots after modifications


    def _find_available_paste_starts(self, search_range_min_idx, search_range_max_idx, length_of_segment_to_paste):
        if self.data.empty or length_of_segment_to_paste == 0:
            return []

        # The latest possible start index for the segment to fit within self.data
        true_max_data_index = self.data.index.max() # Actual last index in data
        # Max start index such that segment_to_paste still fits within data
        absolute_max_start_idx = true_max_data_index - length_of_segment_to_paste + 1
        
        # Effective end of search range for start indices
        # Cannot start paste if the segment would go beyond available data or specified range_max_idx
        effective_search_end_idx = min(search_range_max_idx, absolute_max_start_idx)
        
        possible_start_indices = [s_idx for s_idx in range(search_range_min_idx, effective_search_end_idx + 1) 
                                  if s_idx <= absolute_max_start_idx] # Double check fit

        if not possible_start_indices: return []
        
        # If 'slice_number' column doesn't exist or no concept of "pasted" regions, all are available
        if 'slice_number' not in self.data.columns:
            return possible_start_indices

        valid_non_overlapping_starts = []
        for potential_start in possible_start_indices:
            # Define the target indices for this potential paste
            target_indices_for_check = self.data.index[potential_start : potential_start + length_of_segment_to_paste]
            
            if len(target_indices_for_check) < length_of_segment_to_paste: # Segment doesn't fit
                continue 
            
            # Check if this range overlaps with any region marked with slice_number > 0
            if not (self.data.loc[target_indices_for_check, 'slice_number'] > 0).any():
                valid_non_overlapping_starts.append(potential_start)
        
        return valid_non_overlapping_starts

    def perform_precise_selection(self, xmin_str, xmax_str):
        """Performs selection based on string inputs for min and max index."""
        try:
            xmin_val = int(xmin_str)
            xmax_val = int(xmax_str)

            if xmin_val > xmax_val:
                raise ValueError("Precise Selection: 'From Index' cannot be greater than 'To Index'.")
            if self.data.empty:
                raise ValueError("Precise Selection: No data is loaded to select from.")

            min_data_idx = self.data.index.min()
            max_data_idx = self.data.index.max()

            if not (min_data_idx <= xmin_val <= max_data_idx and min_data_idx <= xmax_val <= max_data_idx):
                raise ValueError(f"Precise Selection: Indices [{xmin_val}, {xmax_val}] are out of the valid data range [{min_data_idx}, {max_data_idx}].")
            
            # Apply the selection (this will update copied_data, plots, and phases)
            self.on_span_select(xmin_val, xmax_val) # Core selection logic
            
            # Crucially, reset span selectors on the plots to clear any visual drag-selection
            # and ensure they are ready for new interactions.
            self.reset_all_span_selectors() # From PlottingMixin
            
            # Store the successfully used values for next time (e.g., in randomize paste defaults)
            self.last_random_paste_options['from_index'] = str(xmin_val)
            self.last_random_paste_options['to_index'] = str(xmax_val)

            QMessageBox.information(self, "Precise Selection", f"Data selected from index {xmin_val} to {xmax_val}.")

        except ValueError as ve:
            QMessageBox.warning(self, "Precise Selection Error", str(ve))
        except Exception as e: # Catch any other unexpected errors
            QMessageBox.critical(self, "Precise Selection Error", f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc() # For debugging

    def handle_xlim_change_for_grouped_plots(self, changed_axes_object):
        """Synchronizes X-axis limits across all grouped plots when one is zoomed/panned."""
        if self.is_zooming or not hasattr(self,'axes_grouped') or not self.axes_grouped: # self.is_zooming is a flag to prevent recursion
            return
        
        self.is_zooming = True # Set lock
        try:
            new_xlim_tuple = changed_axes_object.get_xlim()
            
            # Iterate through all pairs of grouped plots
            for pair_idx in range(self.num_pairs):
                if pair_idx < len(self.axes_grouped) and self.axes_grouped[pair_idx]: # Check if pair exists
                    # axes_grouped[pair_idx] contains [ax_X_vs_idx, ax_Y_vs_idx]
                    for ax_in_pair in self.axes_grouped[pair_idx]:
                        if ax_in_pair and ax_in_pair != changed_axes_object: # If axis exists and is not the one that triggered
                            if ax_in_pair.get_xlim() != new_xlim_tuple: # Avoid redundant updates
                                ax_in_pair.set_xlim(new_xlim_tuple)
            
            # Redraw canvases for all grouped plots that might have changed
            for pair_idx_redraw in range(self.num_pairs):
                 if pair_idx_redraw < self.canvases.shape[0] and self.canvases[pair_idx_redraw, 0]: # Check canvas exists for grouped plot
                    self.canvases[pair_idx_redraw, 0].draw_idle()
        
        finally:
            self.is_zooming = False # Release lock
