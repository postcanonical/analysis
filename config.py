import numpy as np

THEMES = {
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

DEFAULT_RANDOM_PASTE_OPTIONS = {
    'from_index': '0',
    'to_index': '5000',
    'num_pastes': '1',
    'scale_type': 'Constant',
    'constant_scale': '1',
    'random_scale_min': '0',
    'random_scale_max': '1',
}

DEFAULT_CALIBRATION = {'voltage': 5.0, 'amplitude': 1.0}