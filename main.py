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
from matplotlib.collections import LineCollection # Added for type checking
import matplotlib.colors as mcolors # Added for colormaps and normalization
import matplotlib.pyplot as plt # For isinstance checks and cm

# Ensure this module is available or comment out if not used for testing
from phase import get_amp_phase

class DataVisualizationTool(QWidget):
    """
    A PyQt5-based data visualization tool that supports:
      - Loading data (CSV or space-separated).
      - Displaying column pairs in multiple plot types.
      - Selecting data ranges (via SpanSelector or precise indices).
      - Copying/pasting data (including randomizing paste positions).
      - Computing amplitude and phase via an external function (get_amp_phase).
      - Customizable themes for the UI and plots.
    """
    def __init__(self, num_pairs=None):  # New: Accept an optional predefined number of pairs
        super().__init__()
        self.predefined_num_pairs = num_pairs  # Store predefined num_pairs if provided
        self.setWindowTitle("Data Visualization Tool")

        # 1) Variable & UI initialization
        self.init_variables()
        self.init_themes()  # New: Initialize themes
        self.init_ui() # This will set the initial UI style based on OS/Qt defaults

        # 2) Initial data loading is now handled in the if __name__ == '__main__' block
        # self.load_initial_data() # Removed from here

        # 3) Keyboard shortcuts
        self.setup_shortcuts()

        # 4) Show the main window maximized
        self.showMaximized()
        # —————————————————————————————————————————————————————————
        # Memory for last-random-paste options
        # —————————————————————————————————————————————————————————
        self.last_random_paste_options = {
            'from_index': '0',
            'to_index': '5000',
            'num_pastes': '1',
            'scale_type': 'Constant',
            'constant_scale': '1',
            'random_scale_min': '0',
            'random_scale_max': '1',
        }

    # -------------------------------------------------------------------------
    # Initialization & Setup
    # -------------------------------------------------------------------------
    def init_variables(self):
        """Initialize variables used throughout the class."""
        self.data = pd.DataFrame() # Initialize self.data
        self.copied_data = pd.DataFrame()
        self.paste_count = 0
        self.calibration = {'voltage': 5.0, 'amplitude': 1.0}
        self.is_calibrated = False
        self.amplitude = np.array([]) # Initialize as numpy array
        self.phase = np.array([]) # Initialize as numpy array
        self.is_zooming = False
        self.current_theme = "dark"  # Default plot theme
        self.num_pairs = 0 # Initialize num_pairs
        self.current_plot_prefixes = [] # Initialize list to store prefixes like "0", "1", "math0"

        # Will be updated once data is loaded
        self.plot_scatter_grouped = np.empty((1, 2), dtype=object)
        self.selection_lines = []  # For storing vertical lines in grouped plots

    def init_themes(self):
        """Initialize theme presets for plots."""
        # Theme definitions
        self.themes = {
            "dark": {
                "figure_bg": "black",
                "axes_bg": "black",
                "text_color": "white",
                "grid_color": "gray",
                "line_color": "white", 
                "selection_color": "green",
                "span_color": "red"
            },
            "light": {
                "figure_bg": "white",
                "axes_bg": "white",
                "text_color": "black",
                "grid_color": "lightgray",
                "line_color": "blue", 
                "selection_color": "green",
                "span_color": "red"
            },
            "blue": {
                "figure_bg": "#1e3a5f",
                "axes_bg": "#1e3a5f",
                "text_color": "white",
                "grid_color": "#5a88c6",
                "line_color": "#00ffff", 
                "selection_color": "yellow",
                "span_color": "yellow"
            },
            "solarized": {
                "figure_bg": "#002b36",
                "axes_bg": "#002b36",
                "text_color": "#839496",
                "grid_color": "#586e75",
                "line_color": "#2aa198", 
                "selection_color": "#d33682",
                "span_color": "#cb4b16"
            }
        }

    def init_ui(self):
        """Initialize the user interface components."""
        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)
        self.init_plot_area()
        self.init_button_panel()
        self.main_layout.addWidget(self.plot_widget)
        self.main_layout.addWidget(self.button_widget)

    def init_plot_area(self):
        self.plot_widget = QWidget()
        self.plot_layout = QGridLayout()
        self.plot_widget.setLayout(self.plot_layout)

    def init_button_panel(self):
        self.button_layout = QVBoxLayout()
        self.button_widget = QGroupBox("Menu")
        self.button_widget.setLayout(self.button_layout)
        self.button_widget.setFixedWidth(300)
        self.add_theme_selector()
        self.add_main_buttons()
        self.add_sub_button_placeholder()
        self.add_status_bar()
        self.add_spacer()
        self.connect_main_buttons()

    def add_theme_selector(self):
        theme_layout = QHBoxLayout()
        theme_label = QLabel("Plot Theme:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(list(self.themes.keys()))
        self.theme_combo.setCurrentText(self.current_theme)
        self.theme_combo.currentTextChanged.connect(self.change_theme)
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_combo)
        self.button_layout.addLayout(theme_layout)

    def add_main_buttons(self):
        self.data_interference_button = QPushButton('Data Interference')
        self.precise_selection_button = QPushButton('Precise Data Selection')
        self.save_data_button = QPushButton('Save Data to File')
        self.load_data_button = QPushButton('Load Data')
        for button in [self.data_interference_button, self.precise_selection_button, self.save_data_button, self.load_data_button]:
            self.button_layout.addWidget(button)

    def add_sub_button_placeholder(self):
        self.sub_button_layout = QVBoxLayout()
        self.sub_button_widget = QGroupBox()
        self.sub_button_widget.setLayout(self.sub_button_layout)
        self.sub_button_widget.hide()
        self.button_layout.addWidget(self.sub_button_widget)

    def add_status_bar(self):
        self.status_bar = QLabel("Phases: N/A")
        self.status_bar.setStyleSheet("background-color: #DDDDDD; color: black; padding: 5px; border: 1px solid #AAAAAA;")
        self.status_bar.setAlignment(Qt.AlignLeft)
        self.status_bar.setWordWrap(True)
        self.button_layout.addWidget(self.status_bar)

    def add_spacer(self):
        self.button_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

    def connect_main_buttons(self):
        self.data_interference_button.clicked.connect(self.show_data_interference_buttons)
        self.save_data_button.clicked.connect(self.show_save_data_buttons)
        self.precise_selection_button.clicked.connect(self.show_precise_selection_fields)
        self.load_data_button.clicked.connect(self.load_data)

    def setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+C"), self).activated.connect(self.copy_data)
        QShortcut(QKeySequence("Ctrl+V"), self).activated.connect(self.paste_data)
        QShortcut(QKeySequence("Ctrl+Shift+V"), self).activated.connect(self.show_randomize_paste_options)

    def change_theme(self, theme_name):
        if theme_name not in self.themes: return
        self.current_theme = theme_name
        self.update_plot_theme()
        print(f"Plot theme changed to: {theme_name}")

    def update_plot_theme(self):
        if not hasattr(self, 'figures') or not self.figures.any() or \
           not hasattr(self, 'axes_grouped') or not hasattr(self, 'axes_colcol'):
            return
        theme = self.themes[self.current_theme]
        for row in range(self.num_pairs):
            if row >= self.figures.shape[0]: continue 
            if self.figures[row, 0] is not None and row < len(self.axes_grouped) and self.axes_grouped[row]:
                self.figures[row, 0].set_facecolor(theme['figure_bg'])
                for ax_idx, ax in enumerate(self.axes_grouped[row]):
                    if ax is None: continue
                    self.apply_theme_to_axis(ax)
                    if row < self.plot_lines_grouped.shape[0] and ax_idx < self.plot_lines_grouped.shape[1]:
                        plot_obj = self.plot_lines_grouped[row, ax_idx]
                        if isinstance(plot_obj, LineCollection):
                            new_cmap = mcolors.LinearSegmentedColormap.from_list(
                                f'defect_cmap_grouped_{row}_{ax_idx}_{theme["line_color"].replace("#","")}',
                                [theme['line_color'], 'red'], N=256)
                            plot_obj.set_cmap(new_cmap) 
                        elif isinstance(plot_obj, plt.Line2D):
                            plot_obj.set_color(theme['line_color'])
                if self.canvases[row, 0]: self.canvases[row, 0].draw_idle()

            if self.figures[row, 1] is not None and row < len(self.axes_colcol) and self.axes_colcol[row]:
                self.figures[row, 1].set_facecolor(theme['figure_bg'])
                ax_col = self.axes_colcol[row][0]
                if ax_col is None: continue
                self.apply_theme_to_axis(ax_col)
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
            
            if row < len(self.selection_lines):
                for subplot_idx in range(len(self.selection_lines[row])): 
                    for line_obj in self.selection_lines[row][subplot_idx]:
                        if line_obj: line_obj.set_color(theme['selection_color'])
            
            if hasattr(self, 'span_selectors') and self.span_selectors is not None:
                 if row < self.span_selectors.shape[0]:
                    for col_idx_span in range(self.span_selectors.shape[1]):
                        span_selector_widget = self.span_selectors[row, col_idx_span]
                        if span_selector_widget and hasattr(span_selector_widget, 'props'): 
                            span_selector_widget.props['facecolor'] = theme['span_color']
                            if span_selector_widget.active and hasattr(span_selector_widget, 'patch') and span_selector_widget.patch:
                                span_selector_widget.patch.set_facecolor(theme['span_color'])


        for r_idx in range(self.num_pairs): 
            if r_idx < self.canvases.shape[0]:
                if self.canvases[r_idx, 0]: self.canvases[r_idx, 0].draw_idle()
                if self.canvases[r_idx, 1]: self.canvases[r_idx, 1].draw_idle()

    def apply_theme_to_axis(self, ax):
        theme = self.themes[self.current_theme]
        ax.set_facecolor(theme['axes_bg'])
        ax.tick_params(axis='x', colors=theme['text_color']); ax.tick_params(axis='y', colors=theme['text_color'])
        ax.xaxis.label.set_color(theme['text_color']); ax.yaxis.label.set_color(theme['text_color'])
        ax.title.set_color(theme['text_color'])
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color=theme['grid_color'])
        for spine in ax.spines.values(): spine.set_edgecolor(theme['text_color'])

    def load_initial_data(self): 
        if self.data.empty: 
            print("Loading dummy data (load_initial_data called).")
            n_samples = 1000
            self.num_pairs = self.predefined_num_pairs if self.predefined_num_pairs is not None else 2
            self.initialize_plot_arrays() 
            dummy_dict = {}
            self.current_plot_prefixes = [str(i) for i in range(self.num_pairs)]
            for i in range(self.num_pairs):
                prefix = self.current_plot_prefixes[i]
                dummy_dict[f'{prefix}_X'] = np.random.randn(n_samples).cumsum()
                dummy_dict[f'{prefix}_Y'] = np.random.randn(n_samples).cumsum()
                probs = np.random.rand(n_samples)
                probs[n_samples//10 : 2*n_samples//10] = np.clip(np.random.rand(n_samples//10)*2,0,1) 
                probs[n_samples//2 : n_samples//2 + n_samples//20] = np.clip(np.random.rand(n_samples//20)*3,0,1)
                dummy_dict[f'defect_proba_{prefix}'] = probs
            self.data = pd.DataFrame(dummy_dict)
            self.create_plots() 

    def read_dep_to_df(self, file_path, n_channels, math_n_channels=0):
        import os, numpy as np, pandas as pd 
        filesize = os.path.getsize(file_path); bytes_per_sample_per_pair = 4 
        total_bytes_per_sample = n_channels * bytes_per_sample_per_pair
        if filesize <= 1024 or ((filesize - 1024) % total_bytes_per_sample) != 0:
            raise ValueError(f"Invalid DEP for {n_channels} pairs. Filesize: {filesize}")
        n_samples = (filesize - 1024) // total_bytes_per_sample
        with open(file_path, 'rb') as f:
            f.seek(1024); raw = np.fromfile(f, dtype='<i2', count=n_samples * n_channels * 2)
        raw = raw.reshape(n_samples, n_channels * 2); data_dict = {}
        for ch_pair_idx in range(n_channels): 
            data_dict[f"{ch_pair_idx}_X"] = raw[:, 2*ch_pair_idx].astype(float)
            data_dict[f"{ch_pair_idx}_Y"] = raw[:, 2*ch_pair_idx + 1].astype(float)
        df = pd.DataFrame(data_dict)
        if math_n_channels > 0 and "0_X" in df.columns and "0_Y" in df.columns:
            base_X = df["0_X"].values; base_Y = df["0_Y"].values
            for j in range(math_n_channels):
                mx = np.zeros(n_samples,dtype=float); my = np.zeros(n_samples,dtype=float)
                if n_samples > 11: 
                    diffs_X=(base_X[11:]-base_X[:-11])/5.0; diffs_Y=(base_Y[11:]-base_Y[:-11])/5.0
                    start_mx=5; end_mx=start_mx+len(diffs_X)
                    mx[start_mx:end_mx]=diffs_X; my[start_mx:end_mx]=diffs_Y
                df[f"math{j}_X"]=mx; df[f"math{j}_Y"]=my
        return df

    def load_data_file(self, file_path):
        try:
            if file_path.endswith('.csv'): self.data = pd.read_csv(file_path)
            elif file_path.lower().endswith('.dep'):
                n_ch,ok = QInputDialog.getInt(self,"DEP Loader","Pairs:",value=8,min=1)
                if not ok: self.data=pd.DataFrame(); return 
                m_ch,ok2 = QInputDialog.getInt(self,"DEP Loader","Math ch:",value=0,min=0)
                if not ok2: self.data=pd.DataFrame(); return 
                self.data = self.read_dep_to_df(file_path,n_ch,m_ch)
            else: 
                self.data = pd.read_csv(file_path,sep=r'\s+',header=None)
                if self.data.shape[1]%2!=0: QMessageBox.warning(self,"Load Error","Odd cols."); self.data=pd.DataFrame(); return
                self.data.columns = [f"{i//2}_{'X' if i%2==0 else 'Y'}" for i in range(self.data.shape[1])]
            if self.data.empty: return
            raw_p={c.split('_')[0] for c in self.data.columns if c.split('_')[0].isdigit() and c.endswith(('_X','_Y'))}
            math_p={c.split('_')[0] for c in self.data.columns if c.startswith("math") and c.endswith(('_X','_Y'))}
            self.current_plot_prefixes = []
            for p_set in [sorted(list(raw_p),key=int), sorted(list(math_p))]: 
                for p in p_set:
                    if f"{p}_X" in self.data.columns and f"{p}_Y" in self.data.columns: self.current_plot_prefixes.append(p)
            comp_pairs = len(self.current_plot_prefixes)
            self.num_pairs = min(self.predefined_num_pairs,comp_pairs) if self.predefined_num_pairs is not None else comp_pairs
            self.current_plot_prefixes = self.current_plot_prefixes[:self.num_pairs]
            if self.num_pairs==0:
                if not self.data.empty: QMessageBox.warning(self,"Load Info","No valid pairs.")
                self.data=pd.DataFrame(); return
            self.initialize_plot_arrays()
        except Exception as e: 
            QMessageBox.critical(self,"Load Error",str(e))
            self.data=pd.DataFrame(); self.num_pairs=0; self.clear_plots(); self.initialize_plot_arrays()

    def initialize_plot_arrays(self):
        self.figures=np.empty((self.num_pairs,2),dtype=object); self.canvases=np.empty((self.num_pairs,2),dtype=object)
        self.axes_grouped=[[] for _ in range(self.num_pairs)]; self.axes_colcol=[[] for _ in range(self.num_pairs)] 
        self.span_selectors=np.empty((self.num_pairs,2),dtype=object)
        self.plot_lines_grouped=np.empty((self.num_pairs,2),dtype=object); self.plot_lines_colcol=np.empty((self.num_pairs,1),dtype=object)
        self.amplitude=np.full(self.num_pairs,np.nan); self.phase=np.full(self.num_pairs,np.nan)
        self.selection_lines=[[[] for _ in range(2)] for _ in range(self.num_pairs)]

    def create_plots(self):
        if self.data.empty or self.num_pairs==0 or not self.current_plot_prefixes: return
        for idx, prefix in enumerate(self.current_plot_prefixes):
            if idx >= self.num_pairs: break 
            self.create_individual_plots(idx, prefix)
        self.update_plot_theme()

    def create_individual_plots(self, plot_row_idx, data_prefix):
        x_col=f"{data_prefix}_X"; y_col=f"{data_prefix}_Y"
        if x_col not in self.data.columns or y_col not in self.data.columns:
            self.figures[plot_row_idx,0]=None; self.canvases[plot_row_idx,0]=None; self.figures[plot_row_idx,1]=None; self.canvases[plot_row_idx,1]=None; return
        self.axes_grouped[plot_row_idx]=[]; self.axes_colcol[plot_row_idx]=[]
        fig_g, can_g = self.create_grouped_plot(plot_row_idx); self.figures[plot_row_idx,0],self.canvases[plot_row_idx,0]=fig_g,can_g
        ax1,ln1=self.plot_column_vs_index(fig_g,self.data[x_col],f'{x_col} vs Idx',plot_row_idx,data_prefix,0)
        ax2,ln2=self.plot_column_vs_index(fig_g,self.data[y_col],f'{y_col} vs Idx',plot_row_idx,data_prefix,1,sharex=ax1)
        self.plot_lines_grouped[plot_row_idx,0]=ln1; self.plot_lines_grouped[plot_row_idx,1]=ln2
        self.axes_grouped[plot_row_idx].extend([ax1,ax2])
        if ax1: self.span_selectors[plot_row_idx,0]=self.create_span_selector(ax1)
        if ax2: self.span_selectors[plot_row_idx,1]=self.create_span_selector(ax2)
        if ax1 and ax2: self.synchronize_axes(ax1,ax2)
        fig_c, can_c = self.create_plot_with_toolbar(plot_row_idx); self.figures[plot_row_idx,1],self.canvases[plot_row_idx,1]=fig_c,can_c
        ax_cc,ln_cc=self.plot_column_vs_column(fig_c,self.data[x_col],self.data[y_col],x_col,y_col,plot_row_idx,data_prefix)
        self.plot_lines_colcol[plot_row_idx,0]=ln_cc; self.axes_colcol[plot_row_idx].append(ax_cc)

    def create_grouped_plot(self, plot_row_idx):
        fig=Figure(facecolor=self.themes[self.current_theme]['figure_bg'],constrained_layout=True)
        can=FigureCanvas(fig); tb=NavigationToolbar(can,self)
        self.plot_layout.addWidget(tb,plot_row_idx*2,0); self.plot_layout.addWidget(can,plot_row_idx*2+1,0)
        return fig,can

    def create_plot_with_toolbar(self, plot_row_idx):
        fig=Figure(facecolor=self.themes[self.current_theme]['figure_bg'],constrained_layout=True)
        can=FigureCanvas(fig); tb=NavigationToolbar(can,self)
        self.plot_layout.addWidget(tb,plot_row_idx*2,1); self.plot_layout.addWidget(can,plot_row_idx*2+1,1)
        return fig,can

    def plot_column_vs_index(self, fig, data_col, lbl, p_row_idx, d_prefix, sub_idx, sharex=None):
        ax=fig.add_subplot(211+sub_idx,sharex=sharex if sharex else None)
        if data_col.empty: self.setup_plot(ax,lbl+" (No Data)",'Idx' if sub_idx==1 else '','Volt'); self.apply_theme_to_axis(ax); return ax,None
        prob_col=f'defect_proba_{d_prefix}'; has_prob=prob_col in self.data.columns and not self.data[prob_col].isnull().all()
        ln_art=None; idxs=self.data.index.values; vals=data_col.values
        if has_prob:
            probs=self.data[prob_col].fillna(0).values
            segs=[[(idxs[i],vals[i]),(idxs[i+1],vals[i+1])] for i in range(len(idxs)-1)]
            c_lc=[probs[i] for i in range(len(idxs)-1)]
            if not segs: ln_art,=ax.plot(idxs,vals,color=self.themes[self.current_theme]['line_color'],lw=1)
            else:
                th_ln_c=self.themes[self.current_theme]['line_color']; dfct_c='red'
                cmap_n=f'dfct_cmap_init_{p_row_idx}_{sub_idx}_{th_ln_c.replace("#","")}'
                cmap=mcolors.LinearSegmentedColormap.from_list(cmap_n,[th_ln_c,dfct_c],N=256)
                lc=LineCollection(segs,cmap=cmap,lw=1.5,zorder=10)
                lc.set_array(np.array(c_lc)); lc.set_norm(mcolors.Normalize(vmin=0,vmax=1)) 
                ln_art=ax.add_collection(lc)
            ax.set_xlim(idxs.min(),idxs.max()); ax.set_ylim(vals.min(),vals.max())
        else: ln_art,=ax.plot(idxs,vals,label=lbl,color=self.themes[self.current_theme]['line_color'],lw=1)
        self.setup_plot(ax,lbl if sub_idx==0 else '','Idx' if sub_idx==1 else '','Volt')
        if sub_idx==0: ax.tick_params(labelbottom=False,bottom=False)
        self.apply_theme_to_axis(ax); return ax,ln_art

    def plot_column_vs_column(self, fig, x_data, y_data, x_col_name, y_col_name, plot_row_idx, data_prefix):
        ax = fig.add_subplot(111)
        if x_data.empty or y_data.empty:
            self.setup_plot(ax, f'{x_col_name} vs {y_col_name} (No Data)', x_col_name, y_col_name)
            self.apply_theme_to_axis(ax); return ax, None
        
        line_artist = None
        prob_col_name = f'defect_proba_{data_prefix}'
        has_defect_proba = prob_col_name in self.data.columns and not self.data[prob_col_name].isnull().all()

        if has_defect_proba:
            valid_indices = x_data.index.intersection(y_data.index).intersection(self.data.index)
            if valid_indices.empty: has_defect_proba = False 
            else:
                probabilities = self.data.loc[valid_indices, prob_col_name].fillna(0).values
                current_x_data = x_data.loc[valid_indices].values; current_y_data = y_data.loc[valid_indices].values
                segments = [[(current_x_data[i], current_y_data[i]), (current_x_data[i+1], current_y_data[i+1])] for i in range(len(current_x_data) - 1)]
                colors_for_lc = [probabilities[i] for i in range(len(current_x_data) - 1)] 
                
                if not segments: 
                    line_artist, = ax.plot(current_x_data, current_y_data, color=self.themes[self.current_theme]['line_color'])
                else:
                    theme_line_color = self.themes[self.current_theme]['line_color']
                    defect_color = 'red'
                    cmap_name = f'col_vs_col_defect_cmap_{plot_row_idx}_{theme_line_color.replace("#","")}'
                    cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, [theme_line_color, defect_color], N=256)
                    
                    lc = LineCollection(segments, cmap=cmap, linewidth=1.5, zorder=10)
                    lc.set_array(np.array(colors_for_lc)); lc.set_norm(mcolors.Normalize(vmin=0, vmax=1))
                    line_artist = ax.add_collection(lc)
                ax.set_xlim(current_x_data.min(), current_x_data.max()); ax.set_ylim(current_y_data.min(), current_y_data.max())
        
        if not has_defect_proba: 
            line_artist, = ax.plot(x_data, y_data, color=self.themes[self.current_theme]['line_color'], label=f'{x_col_name} vs {y_col_name}')
            ax.set_xlim(x_data.min(), x_data.max()); ax.set_ylim(y_data.min(), y_data.max())

        self.setup_plot(ax, f'{x_col_name} vs {y_col_name}', x_col_name, y_col_name)
        self.apply_theme_to_axis(ax)
        return ax, line_artist

    def setup_plot(self, ax, title, xlabel='', ylabel=''):
        ax.set_title(title,fontsize=10); ax.set_xlabel(xlabel,fontsize=9); ax.set_ylabel(ylabel,fontsize=9)
        ax.grid(color=self.themes[self.current_theme]['grid_color'],ls='--',lw=0.5)
        ax.tick_params(axis='both',which='major',labelsize=8)

    def create_span_selector(self, ax):
        return SpanSelector(ax,self.on_select,'horizontal',useblit=False,props=dict(alpha=0.3,facecolor=self.themes[self.current_theme]['span_color']),interactive=True,drag_from_anywhere=True)

    def synchronize_axes(self, ax1, ax2):
        if ax1 and ax2: ax1.callbacks.connect('xlim_changed',self.on_xlim_changed_grouped); ax2.callbacks.connect('xlim_changed',self.on_xlim_changed_grouped)

    def on_select(self, xmin, xmax):
        if self.data.empty or not isinstance(xmin,(int,float)) or not isinstance(xmax,(int,float)): return
        xmin_idx,xmax_idx=int(np.round(xmin)),int(np.round(xmax))
        if self.data.index.empty: return
        min_d_idx,max_d_idx=self.data.index.min(),self.data.index.max()
        xmin_idx=max(min_d_idx,xmin_idx); xmax_idx=min(max_d_idx,xmax_idx) 
        if xmin_idx>=xmax_idx: 
            self.copied_data=pd.DataFrame()
            for p_idx_c in range(self.num_pairs):
                 if p_idx_c<len(self.axes_grouped) and self.axes_grouped[p_idx_c]:
                    for s_idx_c,_ in enumerate(self.axes_grouped[p_idx_c]): self.clear_vertical_lines(p_idx_c,s_idx_c)
                    if self.canvases[p_idx_c,0]: self.canvases[p_idx_c,0].draw_idle()
                 if p_idx_c<self.figures.shape[0] and self.figures[p_idx_c,1]: self.update_plot(p_idx_c) 
            self.amplitude.fill(np.nan); self.phase.fill(np.nan); self.update_status_bar(); return
        idx_mask=(self.data.index>=xmin_idx)&(self.data.index<=xmax_idx)
        self.copied_data=self.data.loc[idx_mask].copy()
        for p_idx in range(self.num_pairs): self.update_selection_plots(p_idx,idx_mask,xmin_idx,xmax_idx)
        sel_idxs=self.data.index[idx_mask]
        if not sel_idxs.empty: self.update_phases(sel_idxs)
        else: self.amplitude.fill(np.nan); self.phase.fill(np.nan); self.update_status_bar()

    def update_selection_plots(self, p_idx, idx_mask, xmin_v, xmax_v):
        if p_idx>=len(self.current_plot_prefixes): return 
        d_prefix=self.current_plot_prefixes[p_idx]; x_c,y_c=f"{d_prefix}_X",f"{d_prefix}_Y"
        if x_c not in self.data.columns or y_c not in self.data.columns: return
        if p_idx<self.figures.shape[0] and self.figures[p_idx,1] and p_idx<len(self.axes_colcol) and self.axes_colcol[p_idx] and self.axes_colcol[p_idx][0]:
            ax_cc=self.axes_colcol[p_idx][0]; ax_cc.clear()
            sel_data_sub=self.data.loc[idx_mask]
            if not sel_data_sub.empty:
                sel_x=sel_data_sub[x_c]; sel_y=sel_data_sub[y_c]; prob_c_n=f'defect_proba_{d_prefix}'
                has_prob_s=prob_c_n in sel_data_sub.columns and not sel_data_sub[prob_c_n].isnull().all()
                if has_prob_s and not sel_x.empty: # Ensure sel_x is not empty for indexing probabilities
                    sel_probs=sel_data_sub.loc[sel_x.index,prob_c_n].fillna(0).values
                    segs=[[(sel_x.values[i],sel_y.values[i]),(sel_x.values[i+1],sel_y.values[i+1])] for i in range(len(sel_x)-1)]
                    if segs:
                        theme_line_color_sel = self.themes[self.current_theme]['line_color']
                        cmap_sel = mcolors.LinearSegmentedColormap.from_list(
                            f'sel_cc_cmap_{p_idx}_{theme_line_color_sel.replace("#","")}', 
                            [theme_line_color_sel, 'red'], N=256)
                        lc_s=LineCollection(segs,cmap=cmap_sel,lw=1.5,zorder=10)
                        lc_s.set_array(np.array([sel_probs[i] for i in range(len(sel_x)-1)])); lc_s.set_norm(mcolors.Normalize(vmin=0,vmax=1))
                        ax_cc.add_collection(lc_s)
                elif not sel_x.empty: ax_cc.plot(sel_x,sel_y,color=self.themes[self.current_theme]['line_color'],lw=1)
                
                if not sel_x.empty: ax_cc.set_xlim(sel_x.min(),sel_x.max()); ax_cc.set_ylim(sel_y.min(),sel_y.max())
                else: ax_cc.set_xlim(self.data[x_c].min(),self.data[x_c].max()); ax_cc.set_ylim(self.data[y_c].min(),self.data[y_c].max())
                self.setup_plot(ax_cc,f'Sel: {x_c} v {y_c}',x_c,y_c); self.apply_theme_to_axis(ax_cc)
                if self.canvases[p_idx,1]: self.canvases[p_idx,1].draw_idle()

        if p_idx<len(self.axes_grouped) and self.axes_grouped[p_idx]:
            for s_idx_g,ax_g in enumerate(self.axes_grouped[p_idx]):
                if ax_g is None: continue
                self.clear_vertical_lines(p_idx,s_idx_g) 
                if not self.copied_data.empty: self.draw_selection_lines(p_idx,s_idx_g,ax_g,xmin_v,xmax_v)
            if self.canvases[p_idx,0]: self.canvases[p_idx,0].draw_idle()

    def clear_vertical_lines(self, p_idx, s_idx_g):
        if p_idx<len(self.selection_lines) and s_idx_g<len(self.selection_lines[p_idx]):
            for ln in self.selection_lines[p_idx][s_idx_g]:
                if ln and ln.axes: 
                    try: ln.remove()
                    except: pass 
            self.selection_lines[p_idx][s_idx_g]=[]

    def draw_selection_lines(self, p_idx, s_idx_g, ax, xmin_v, xmax_v):
        clr=self.themes[self.current_theme]['selection_color']
        ln1=ax.axvline(x=xmin_v,color=clr,ls='--',lw=1.5,zorder=20)
        ln2=ax.axvline(x=xmax_v,color=clr,ls='--',lw=1.5,zorder=20)
        if p_idx<len(self.selection_lines) and s_idx_g<len(self.selection_lines[p_idx]): self.selection_lines[p_idx][s_idx_g].extend([ln1,ln2])

    def update_phases(self, idxs):
        if idxs.empty: self.amplitude.fill(np.nan); self.phase.fill(np.nan)
        else:
            ph_data=self.compute_phases_for_column_pairs(self.data,idxs)
            self.amplitude.fill(np.nan); self.phase.fill(np.nan)
            for p_idx,(amp,ph) in ph_data.items():
                if p_idx<self.num_pairs: self.amplitude[p_idx]=amp; self.phase[p_idx]=ph
        self.update_status_bar()

    def compute_phases_for_column_pairs(self, df, df_idxs):
        ph_res={};
        if df_idxs.empty or not self.current_plot_prefixes: return ph_res
        for p_idx,d_prefix in enumerate(self.current_plot_prefixes):
            if p_idx>=self.num_pairs: break 
            x_c,y_c=f"{d_prefix}_X",f"{d_prefix}_Y"
            if x_c not in df.columns or y_c not in df.columns: ph_res[p_idx]=(np.nan,np.nan); continue
            val_idxs_p=df.index.intersection(df_idxs)
            if val_idxs_p.empty: ph_res[p_idx]=(np.nan,np.nan); continue
            x_v=df.loc[val_idxs_p,x_c].values; y_v=df.loc[val_idxs_p,y_c].values
            if len(x_v)<2: ph_res[p_idx]=(np.nan,np.nan); continue
            xy_l=list(zip(x_v,y_v)); pos=len(xy_l)//2; width=max(1,len(xy_l)//2) if len(xy_l)>0 else 0
            if width==0: ph_res[p_idx]=(np.nan,np.nan); continue
            try: amp,ph=get_amp_phase(xy_l,pos,width,self.is_calibrated,self.calibration); ph_res[p_idx]=(amp,ph)
            except Exception as e: print(f"Phase err {d_prefix}: {e}"); ph_res[p_idx]=(np.nan,np.nan)
        return ph_res

    def copy_data(self):
        if self.copied_data.empty: QMessageBox.warning(self,"Copy","No data selected."); return
        cp_idxs=self.copied_data.index; self.reset_previous_copies()
        if 'slice_number' not in self.data.columns: self.data['slice_number']=np.nan
        self.data.loc[cp_idxs,'slice_number']=-1 
        self.clear_phases_for_indices(cp_idxs)
        ph_info=self.compute_phases_for_column_pairs(self.data,cp_idxs) 
        self.assign_phases_to_indices(cp_idxs,ph_info)
        QMessageBox.information(self,"Copy",f"{len(cp_idxs)} rows copied.")

    def reset_previous_copies(self):
        if 'slice_number' in self.data.columns:
            prev_mask=self.data['slice_number']==-1
            if prev_mask.any(): self.data.loc[prev_mask,'slice_number']=np.nan; self.clear_phases_for_indices(self.data.index[prev_mask])

    def clear_phases_for_indices(self, idxs_c):
        if idxs_c.empty or not self.current_plot_prefixes: return
        for p_idx,d_prefix in enumerate(self.current_plot_prefixes):
            if p_idx>=self.num_pairs: break
            ph_c=f'phase_{d_prefix}' 
            if ph_c in self.data.columns: self.data.loc[idxs_c,ph_c]=np.nan

    def assign_phases_to_indices(self, idxs_a, ph_info):
        if idxs_a.empty or not ph_info or not self.current_plot_prefixes: return
        for p_idx,(amp,ph) in ph_info.items():
            if p_idx<len(self.current_plot_prefixes) and p_idx<self.num_pairs:
                d_prefix=self.current_plot_prefixes[p_idx]; ph_c=f'phase_{d_prefix}' 
                if ph_c not in self.data.columns: self.data[ph_c]=np.nan
                self.data.loc[idxs_a,ph_c]=ph
        self.update_status_bar() 

    def paste_data(self, pos=None, scale=None):
        if self.copied_data.empty: QMessageBox.warning(self,"Paste","No data copied."); return False
        p_start_idx=self.get_paste_position(pos); curr_scale=self.get_scale_factor(scale)
        if p_start_idx is None or curr_scale is None: return False
        if self.perform_paste(p_start_idx,curr_scale): self.update_all_plots(); return True
        return False

    def get_paste_position(self, pos_v):
        if pos_v is not None:
            if 0<=pos_v<len(self.data): return pos_v
            QMessageBox.warning(self,'Paste Pos Err',f"Pos {pos_v} out of bounds."); return None
        max_p=len(self.data)-1 if not self.data.empty else 0
        p_in,ok=QInputDialog.getInt(self,'Paste Pos',f'Enter start idx (0 to {max_p}):',min=0,max=max_p if max_p>=0 else 0)
        return p_in if ok else None

    def get_scale_factor(self, scale_v):
        if scale_v is not None: return float(scale_v)
        s_in,ok=QInputDialog.getDouble(self,'Scale Factor','Enter scale factor:',value=1.0,decimals=3)
        return s_in if ok else None

    def perform_paste(self, p_start_idx, scale_f):
        if self.copied_data.empty: return False
        num_cols=self.copied_data.select_dtypes(include=np.number).columns
        p_seg=self.copied_data.copy(); p_seg[num_cols]*=scale_f
        p_len=len(p_seg); p_end_exc=min(p_start_idx+p_len,len(self.data))
        tgt_idxs=self.data.index[p_start_idx:p_end_exc]
        if tgt_idxs.empty: QMessageBox.warning(self,"Paste Err","Invalid paste range."); return False
        act_p_df=p_seg.iloc[:len(tgt_idxs)]; act_p_df.index=tgt_idxs 
        if self.check_paste_overlap(tgt_idxs): return False
        cols_upd=act_p_df.columns.intersection(self.data.columns)
        num_cols_upd=self.data[cols_upd].select_dtypes(include=np.number).columns
        self.data.loc[tgt_idxs,num_cols_upd]+=act_p_df[num_cols_upd].values
        self.mark_pasted_data(tgt_idxs); self.compute_and_assign_phases_for_pasted(tgt_idxs) 
        return True

    def check_paste_overlap(self, tgt_idxs_p):
        if 'slice_number' in self.data.columns and not tgt_idxs_p.empty:
            if (self.data.loc[tgt_idxs_p,'slice_number']>0).any(): QMessageBox.warning(self,"Paste Conflict","Target overlaps existing."); return True
        return False 

    def mark_pasted_data(self, p_idxs):
        if 'slice_number' not in self.data.columns: self.data['slice_number']=np.nan
        self.paste_count+=1; self.data.loc[p_idxs,'slice_number']=self.paste_count

    def compute_and_assign_phases_for_pasted(self, p_idxs):
        if p_idxs.empty: return
        ph_info=self.compute_phases_for_column_pairs(self.data,p_idxs)
        self.assign_phases_to_indices(p_idxs,ph_info)

    def randomize_paste_positions(self, xmin_idx,xmax_idx,num_p_try,s_type,const_s=None,rand_s_min=None,rand_s_max=None):
        if self.copied_data.empty: QMessageBox.warning(self,"Rand Paste","No data copied."); return
        p_len=len(self.copied_data);
        if p_len==0: return
        avail_starts=self.get_available_paste_start_indices(xmin_idx,xmax_idx,p_len)
        if not avail_starts: QMessageBox.information(self,"Rand Paste","No non-overlapping positions."); return
        num_do=min(num_p_try,len(avail_starts))
        if num_do<num_p_try: QMessageBox.warning(self,"Rand Paste",f"Can only do {num_do} of {num_p_try} pastes.")
        sel_starts=random.sample(avail_starts,num_do); done_c=0
        for start_idx in sel_starts:
            scale=const_s if s_type=='Constant' else random.uniform(rand_s_min,rand_s_max)
            if self.perform_paste(start_idx,scale): done_c+=1
        QMessageBox.information(self,"Rand Paste",f"Completed {done_c} of {num_p_try} pastes.")
        if done_c>0: self.update_all_plots()

    def get_available_paste_start_indices(self, r_min_idx,r_max_idx,len_p):
        if self.data.empty or len_p==0: return []
        true_max_d_idx=self.data.index.max()
        search_end_idx=min(r_max_idx,true_max_d_idx-len_p+1)
        poss_starts=[s for s in range(r_min_idx,search_end_idx+1)]
        if not poss_starts or 'slice_number' not in self.data.columns: return poss_starts
        val_starts=[]
        for p_idx in poss_starts:
            tgt_idxs=self.data.index[p_idx:p_idx+len_p]
            if len(tgt_idxs)<len_p: continue
            if not (self.data.loc[tgt_idxs,'slice_number']>0).any(): val_starts.append(p_idx)
        return val_starts

    def save_data(self, df_save):
        if df_save.empty: QMessageBox.warning(self,"Save","No data to save."); return
        def_fn="export.csv"
        if hasattr(self,'current_file_path') and self.current_file_path:
            base=os.path.splitext(os.path.basename(self.current_file_path))[0]; def_fn=f"{base}_mod.csv"
        f_p,_=QFileDialog.getSaveFileName(self,"Save As",def_fn,"CSV (*.csv);;All (*)")
        if not f_p: return
        try: df_save.to_csv(f_p,index=True); QMessageBox.information(self,"Save",f"Saved to {f_p}")
        except Exception as e: QMessageBox.critical(self,"Save Err",f"Failed: {e}")

    def load_data(self): # This method now opens the file dialog
        f_p,_=QFileDialog.getOpenFileName(self,"Select Data File","","Data (*.csv *.dep *.dat *.txt);;All (*)")
        if f_p:
            self.current_file_path=f_p
            self.clear_plots() 
            self.load_data_file(f_p) # Sets self.data, self.num_pairs, self.current_plot_prefixes

            if self.data.empty or self.num_pairs == 0:
                QMessageBox.information(self,"Load","No data or no valid pairs to plot from selected file.")
                self.copied_data=pd.DataFrame()
                self.amplitude.fill(np.nan); self.phase.fill(np.nan)
                self.update_status_bar()
                return # Stop if no data to plot

            self.copied_data=pd.DataFrame() 
            self.paste_count=0 
            self.create_plots() # This will also apply current plot theme
            QMessageBox.information(self,"Load",f"Loaded {f_p}\n{self.num_pairs} pairs.")
        else: # User cancelled the dialog
            QMessageBox.information(self,"Load","File loading cancelled by user.")
            # Optionally, load dummy data here if no file is chosen and window is empty
            if self.data.empty:
                print("No file chosen, loading dummy data as fallback.")
                self.load_initial_data() # Explicitly load dummy if dialog cancelled and no data


    def clear_plots(self):
        while self.plot_layout.count():
            item=self.plot_layout.takeAt(0); widget=item.widget()
            if widget: widget.deleteLater()
        self.figures=np.array([]); self.canvases=np.array([]); self.axes_grouped=[]; self.axes_colcol=[]
        self.span_selectors=np.array([]); self.plot_lines_grouped=np.array([]); self.plot_lines_colcol=np.array([])
        self.current_plot_prefixes=[]

    def update_plot(self, p_idx_upd):
        if self.data.empty or p_idx_upd>=self.num_pairs or p_idx_upd>=len(self.current_plot_prefixes) or p_idx_upd>=self.figures.shape[0]: return
        d_prefix=self.current_plot_prefixes[p_idx_upd]; x_c,y_c=f"{d_prefix}_X",f"{d_prefix}_Y"
        if x_c not in self.data.columns or y_c not in self.data.columns: return
        fig_g=self.figures[p_idx_upd,0]; can_g=self.canvases[p_idx_upd,0]
        if not fig_g or not can_g or not self.axes_grouped[p_idx_upd]: return
        old_axes_g=self.axes_grouped[p_idx_upd]
        for ax_g in old_axes_g: 
            if ax_g: fig_g.delaxes(ax_g) 
        new_ax_l_g=[]; ax1_n,ln1_n=self.plot_column_vs_index(fig_g,self.data[x_c],f'{x_c} vs Idx',p_idx_upd,d_prefix,0)
        new_ax_l_g.append(ax1_n); ax2_n,ln2_n=self.plot_column_vs_index(fig_g,self.data[y_c],f'{y_c} vs Idx',p_idx_upd,d_prefix,1,sharex=ax1_n)
        new_ax_l_g.append(ax2_n); self.axes_grouped[p_idx_upd]=new_ax_l_g
        self.plot_lines_grouped[p_idx_upd,0]=ln1_n; self.plot_lines_grouped[p_idx_upd,1]=ln2_n
        if ax1_n: self.span_selectors[p_idx_upd,0]=self.create_span_selector(ax1_n)
        if ax2_n: self.span_selectors[p_idx_upd,1]=self.create_span_selector(ax2_n)
        if ax1_n and ax2_n: self.synchronize_axes(ax1_n,ax2_n)
        try: fig_g.tight_layout()
        except: pass
        can_g.draw_idle()
        fig_c=self.figures[p_idx_upd,1]; can_c=self.canvases[p_idx_upd,1]
        if not fig_c or not can_c or not self.axes_colcol[p_idx_upd] or not self.axes_colcol[p_idx_upd][0]: return
        ax_cc_old=self.axes_colcol[p_idx_upd][0]
        if ax_cc_old: fig_c.delaxes(ax_cc_old)
        tmp_ax_cc,tmp_ln_cc=self.plot_column_vs_column(fig_c,self.data[x_c],self.data[y_c],x_c,y_c,p_idx_upd,d_prefix)
        self.axes_colcol[p_idx_upd][0]=tmp_ax_cc; self.plot_lines_colcol[p_idx_upd,0]=tmp_ln_cc
        try: fig_c.tight_layout()
        except: pass
        can_c.draw_idle()

    def update_all_plots(self):
        if self.data.empty or self.num_pairs==0: return
        for p_idx in range(self.num_pairs):
            if p_idx<len(self.current_plot_prefixes): self.update_plot(p_idx)

    def on_xlim_changed_grouped(self, ch_axes):
        if self.is_zooming or not hasattr(self,'axes_grouped') or not self.axes_grouped: return
        self.is_zooming=True
        try:
            new_xlim=ch_axes.get_xlim()
            for i in range(self.num_pairs):
                if i<len(self.axes_grouped) and self.axes_grouped[i]: 
                    for ax_p in self.axes_grouped[i]:
                        if ax_p and ax_p!=ch_axes and ax_p.get_xlim()!=new_xlim: ax_p.set_xlim(new_xlim)
            for i in range(self.num_pairs):
                if i<self.canvases.shape[0] and self.canvases[i,0]: self.canvases[i,0].draw_idle()
        finally: self.is_zooming=False

    def show_data_interference_buttons(self): self.show_sub_buttons("Data Interference",[('Copy (Ctrl+C)',self.copy_data),('Paste (Ctrl+V)',self.paste_data),('Rand. Paste (Ctrl+Shift+V)',self.show_randomize_paste_options)])
    def show_save_data_buttons(self): self.show_sub_buttons("Save Data",[('Save Selected Slice',lambda:self.save_data(self.copied_data)),('Save All Modified',lambda:self.save_data(self.data))])
    def show_precise_selection_fields(self):
        self.sub_button_widget.show(); self.clear_layout(self.sub_button_layout)
        self.sub_button_widget.setTitle("Precise Select"); form_l=QGridLayout() 
        from_l=QLabel('From:'); to_l=QLabel('To:')
        def_f=self.last_random_paste_options.get('from_index','0'); def_t=self.last_random_paste_options.get('to_index',str(max(0,len(self.data)-1)) if not self.data.empty else '0')
        self.precise_from_input=QLineEdit(def_f); self.precise_to_input=QLineEdit(def_t)
        form_l.addWidget(from_l,0,0); form_l.addWidget(self.precise_from_input,0,1)
        form_l.addWidget(to_l,1,0); form_l.addWidget(self.precise_to_input,1,1)
        self.sub_button_layout.addLayout(form_l); sel_b=QPushButton('Select Range')
        sel_b.clicked.connect(lambda:self.precise_selection(self.precise_from_input.text(),self.precise_to_input.text()))
        self.sub_button_layout.addWidget(sel_b)

    def reset_span_selectors(self):
        """Thoroughly clean up and reset all span selectors."""
        if hasattr(self, 'span_selectors') and self.span_selectors is not None:
            for row in range(self.num_pairs):
                for col_idx in range(2):
                    if row < self.span_selectors.shape[0] and col_idx < self.span_selectors.shape[1]:
                        span = self.span_selectors[row, col_idx]
                        if span is not None:
                            # Get the axis for this position
                            if row < len(self.axes_grouped) and col_idx < len(self.axes_grouped[row]):
                                ax = self.axes_grouped[row][col_idx]
                                if ax is not None:
                                    # Attempt to fully clean up the old span selector
                                    try:
                                        # Disconnect all event handlers
                                        if hasattr(span, 'disconnect_events'):
                                            span.disconnect_events()
                                        
                                        # Remove all visual elements
                                        if hasattr(span, 'artists'):
                                            for artist in span.artists:
                                                artist.remove()
                                        
                                        # Remove selection artists (newer matplotlib versions)
                                        if hasattr(span, '_selection_artists'):
                                            for artist in span._selection_artists:
                                                artist.remove()
                                        
                                        # Remove rectangle directly
                                        if hasattr(span, 'rect') and span.rect:
                                            span.rect.remove()
                                        
                                        # Remove patch if it exists
                                        if hasattr(span, 'patch') and span.patch:
                                            span.patch.remove()
                                            
                                        # Clear background for blitting
                                        if hasattr(span, 'background'):
                                            span.background = None
                                    except Exception as e:
                                        print(f"Error cleaning up span: {e}")
                                    
                                    # Create a new fresh span selector
                                    self.span_selectors[row, col_idx] = self.create_span_selector(ax)
            
            # Force redraw all canvases
            for r_idx in range(self.num_pairs):
                if r_idx < self.canvases.shape[0]:
                    if self.canvases[r_idx, 0]: 
                        self.canvases[r_idx, 0].draw_idle()
                    if self.canvases[r_idx, 1]: 
                        self.canvases[r_idx, 1].draw_idle()

    def precise_selection(self, xmin_text, xmax_text):
        try:
            xmin_val, xmax_val = int(xmin_text), int(xmax_text)
            if xmin_val > xmax_val:
                raise ValueError("From Index > To Index.")
            if self.data.empty:
                raise ValueError("No data.")
            min_i, max_i = self.data.index.min(), self.data.index.max()
            if not (min_i <= xmin_val <= max_i and min_i <= xmax_val <= max_i):
                raise ValueError(f"Idx out of range [{min_i},{max_i}].")
            
            # Apply the selection through on_select to update data and green lines
            self.on_select(xmin_val, xmax_val)
            
            # Reset all span selectors to clean state
            self.reset_span_selectors()
            
            # Store the values for future use
            self.last_random_paste_options['from_index'] = str(xmin_val)
            self.last_random_paste_options['to_index'] = str(xmax_val)
                                    
        except ValueError as ve:
            QMessageBox.warning(self, "Precise Select Err", str(ve))
        except Exception as e:
            QMessageBox.critical(self, "Precise Selection Error", f"An unexpected error occurred: {e}")
            print(f"Unexpected error in precise_selection: {e}")
            import traceback
            traceback.print_exc()

    def show_randomize_paste_options(self):
        self.sub_button_widget.show(); self.clear_layout(self.sub_button_layout)
        self.sub_button_widget.setTitle("Rand. Paste Opts"); form_l=QGridLayout(); self.rand_paste_widgets={}
        def add_r(lbl,k,d_v,r,is_c=False,itms=None):
            l=QLabel(lbl);form_l.addWidget(l,r,0);v=self.last_random_paste_options.get(k,d_v)
            if is_c: w=QComboBox();w.addItems(itms or []);idx=w.findText(v);w.setCurrentIndex(idx if idx!=-1 else 0)
            else: w=QLineEdit(v)
            form_l.addWidget(w,r,1);self.rand_paste_widgets[k]=w
        def_t=str(max(0,len(self.data)-1)) if not self.data.empty else '0'
        add_r('From:','from_index','0',0);add_r('To:','to_index',def_t,1);add_r('#Pastes:','num_pastes','1',2)
        add_r('Scale:','scale_type','Constant',3,True,['Constant','Random']);add_r('Const Scale:','constant_scale','1.0',4)
        add_r('Rand Min:','random_scale_min','0.5',5);add_r('Rand Max:','random_scale_max','1.5',6)
        self.sub_button_layout.addLayout(form_l);rand_b=QPushButton('Rand & Paste');rand_b.clicked.connect(self.handle_randomize_paste_clicked)
        self.sub_button_layout.addWidget(rand_b);s_type_c=self.rand_paste_widgets['scale_type']
        def tgl_flds():
            is_c=s_type_c.currentText()=='Constant'
            form_l.itemAtPosition(4,0).widget().setVisible(is_c);self.rand_paste_widgets['constant_scale'].setVisible(is_c)
            form_l.itemAtPosition(5,0).widget().setVisible(not is_c);self.rand_paste_widgets['random_scale_min'].setVisible(not is_c)
            form_l.itemAtPosition(6,0).widget().setVisible(not is_c);self.rand_paste_widgets['random_scale_max'].setVisible(not is_c)
        s_type_c.currentTextChanged.connect(tgl_flds);tgl_flds()

    def handle_randomize_paste_clicked(self):
        try:
            vals={k:w.text() if isinstance(w,QLineEdit) else w.currentText() for k,w in self.rand_paste_widgets.items()}
            frm,to,num_p=int(vals['from_index']),int(vals['to_index']),int(vals['num_pastes'])
            s_type=vals['scale_type']
            if self.data.empty: raise ValueError("No data.")
            min_d,max_d=self.data.index.min(),self.data.index.max()
            if not(min_d<=frm<=max_d and min_d<=to<=max_d and frm<=to): raise ValueError("Idx range invalid.")
            if num_p<=0: raise ValueError("Pastes must be >0.")
            c_s,r_min,r_max=None,None,None
            if s_type=='Constant': c_s=float(vals['constant_scale'])
            else: r_min,r_max=float(vals['random_scale_min']),float(vals['random_scale_max']); assert r_min<=r_max,"Rand min>max"
            self.last_random_paste_options.update(vals)
            self.randomize_paste_positions(frm,to,num_p,s_type,c_s,r_min,r_max)
        except (ValueError,AssertionError) as ve: QMessageBox.warning(self,"Rand. Paste Err",str(ve))
        except Exception as e: QMessageBox.critical(self,"Rand. Paste Err",f"Unexpected: {e}")

    def show_sub_buttons(self, title, btns_cfg):
        self.sub_button_widget.show(); self.clear_layout(self.sub_button_layout) 
        self.sub_button_widget.setTitle(title)
        for txt,func in btns_cfg: btn=QPushButton(txt);btn.clicked.connect(func);self.sub_button_layout.addWidget(btn)
        self.sub_button_layout.addStretch(1)

    def clear_layout(self, lo_clear):
        if lo_clear is None: return
        while lo_clear.count():
            item=lo_clear.takeAt(0)
            if item: widget=item.widget(); sub_lo=item.layout()
            if widget: widget.deleteLater()
            elif sub_lo: self.clear_layout(sub_lo)

    def update_status_bar(self):
        if self.phase is None or self.amplitude is None or not self.current_plot_prefixes or self.num_pairs==0: self.status_bar.setText("Phases: N/A"); return
        parts=[f"P{self.current_plot_prefixes[i]}: A={'%.2f'%self.amplitude[i] if np.isfinite(self.amplitude[i]) else 'N/A'}, Ph={'%.2f°'%self.phase[i] if np.isfinite(self.phase[i]) else 'N/A'}" 
               for i in range(self.num_pairs) if i<len(self.current_plot_prefixes) and i<len(self.phase) and i<len(self.amplitude)]
        self.status_bar.setText("\n".join(parts) if parts else "Phases: N/A")

if __name__ == '__main__':
    app=QApplication(sys.argv)
    parser=argparse.ArgumentParser(description="Data Vis Tool")
    parser.add_argument('-n','--num_pairs',type=int,default=None,help="Num pairs")
    parser.add_argument('-t','--theme',type=str,default='dark',choices=['dark','light','blue','solarized'],help="Plot theme")
    parser.add_argument('file_path',nargs='?',default=None,help="Data file path")
    args=parser.parse_args()

    window=DataVisualizationTool(num_pairs=args.num_pairs) 
    if args.theme!=window.current_theme: # Set initial plot theme if specified
        window.change_theme(args.theme)

    if args.file_path:
        print(f"Attempting to load initial file: {args.file_path}")
        window.current_file_path=args.file_path
        window.clear_plots() 
        window.load_data_file(args.file_path) 
        if not window.data.empty and window.num_pairs > 0:
            window.create_plots() 
        else: 
            print(f"Failed to load data from: {args.file_path} or no plottable pairs found. Opening file dialog.")
            window.load_data() 
    else: 
        print("No file path provided. Opening file dialog.")
        window.load_data() 
    
    if not window.figures.any() and not window.data.empty and window.num_pairs > 0 :
        print("Data exists but no plots. Attempting to create plots.")
        window.create_plots() 
    elif window.figures.any(): 
        window.update_plot_theme()

    window.show()
    sys.exit(app.exec_())

