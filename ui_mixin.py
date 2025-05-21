from PyQt5.QtWidgets import (
    QWidget, QGridLayout, QVBoxLayout, QHBoxLayout,
    QPushButton, QInputDialog, QSpacerItem, QSizePolicy, QLabel,
    QLineEdit, QComboBox, QFileDialog, QMessageBox, QGroupBox
)
from PyQt5.QtCore import Qt

class UIMixin:
    def init_ui_elements(self):
        """Initialize the user interface components."""
        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)
        
        self._init_plot_area_ui()
        self._init_button_panel_ui()
        
        self.main_layout.addWidget(self.plot_widget)
        self.main_layout.addWidget(self.button_widget)
        # Apply initial theme to UI elements if necessary (beyond plots)
        # For now, Qt's default styling or OS styling is used for widgets.

    def _init_plot_area_ui(self):
        self.plot_widget = QWidget()
        self.plot_layout = QGridLayout() # This layout will hold plot canvases and toolbars
        self.plot_widget.setLayout(self.plot_layout)

    def _init_button_panel_ui(self):
        self.button_layout = QVBoxLayout()
        self.button_widget = QGroupBox("Menu")
        self.button_widget.setLayout(self.button_layout)
        self.button_widget.setFixedWidth(300)

        self._add_theme_selector_ui()
        self._add_main_buttons_ui()
        self._add_sub_button_placeholder_ui()
        self._add_status_bar_ui()
        self._add_spacer_ui()
        self._connect_main_buttons_ui()

    def _add_theme_selector_ui(self):
        theme_layout = QHBoxLayout()
        theme_label = QLabel("Plot Theme:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(list(self.themes.keys()))
        self.theme_combo.setCurrentText(self.current_theme)
        self.theme_combo.currentTextChanged.connect(self.on_theme_change) # Renamed for clarity
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_combo)
        self.button_layout.addLayout(theme_layout)

    def _add_main_buttons_ui(self):
        self.data_interference_button = QPushButton('Data Interference')
        self.precise_selection_button = QPushButton('Precise Data Selection')
        self.save_data_button = QPushButton('Save Data to File')
        self.load_data_button = QPushButton('Load Data')
        for button in [self.data_interference_button, self.precise_selection_button, self.save_data_button, self.load_data_button]:
            self.button_layout.addWidget(button)

    def _add_sub_button_placeholder_ui(self):
        self.sub_button_layout = QVBoxLayout()
        self.sub_button_widget = QGroupBox() # Title set dynamically
        self.sub_button_widget.setLayout(self.sub_button_layout)
        self.sub_button_widget.hide()
        self.button_layout.addWidget(self.sub_button_widget)

    def _add_status_bar_ui(self):
        self.status_bar = QLabel("Phases: N/A")
        self.status_bar.setStyleSheet("background-color: #DDDDDD; color: black; padding: 5px; border: 1px solid #AAAAAA;")
        self.status_bar.setAlignment(Qt.AlignLeft)
        self.status_bar.setWordWrap(True)
        self.button_layout.addWidget(self.status_bar)

    def _add_spacer_ui(self):
        self.button_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

    def _connect_main_buttons_ui(self):
        self.data_interference_button.clicked.connect(self.show_data_interference_options_ui)
        self.save_data_button.clicked.connect(self.show_save_data_options_ui)
        self.precise_selection_button.clicked.connect(self.show_precise_selection_ui)
        self.load_data_button.clicked.connect(self.trigger_load_data_dialog) # Connects to DataMixin method

    def on_theme_change(self, theme_name):
        if theme_name not in self.themes:
            return
        self.current_theme = theme_name
        self.apply_theme_to_all_plots() # Method from PlottingMixin
        print(f"Plot theme changed to: {theme_name}")

    def show_data_interference_options_ui(self):
        buttons_config = [
            ('Copy', self.trigger_copy_data),
            ('Paste', self.trigger_paste_data),
            ('Randomize Paste', self.show_randomize_paste_ui)
        ]
        self._display_sub_buttons("Data Interference", buttons_config)

    def show_save_data_options_ui(self):
        buttons_config = [
            ('Save Selected Slice', lambda: self.trigger_save_data(self.copied_data)),
            ('Save All Modified Data', lambda: self.trigger_save_data(self.data))
        ]
        self._display_sub_buttons("Save Data Options", buttons_config)

    def show_precise_selection_ui(self):
        self.sub_button_widget.show()
        self._clear_layout_from_widget(self.sub_button_layout)
        self.sub_button_widget.setTitle("Precise Data Selection")

        form_layout = QGridLayout()
        from_label = QLabel('From Index:')
        to_label = QLabel('To Index:')
        
        default_from = self.last_random_paste_options.get('from_index', '0')
        default_to = self.last_random_paste_options.get('to_index', str(max(0, len(self.data) - 1)) if not self.data.empty else '0')

        self.precise_from_input = QLineEdit(default_from)
        self.precise_to_input = QLineEdit(default_to)

        form_layout.addWidget(from_label, 0, 0)
        form_layout.addWidget(self.precise_from_input, 0, 1)
        form_layout.addWidget(to_label, 1, 0)
        form_layout.addWidget(self.precise_to_input, 1, 1)
        
        self.sub_button_layout.addLayout(form_layout)
        
        select_button = QPushButton('Select Range')
        select_button.clicked.connect(
            lambda: self.perform_precise_selection(self.precise_from_input.text(), self.precise_to_input.text())
        ) # Connects to ActionsMixin method
        self.sub_button_layout.addWidget(select_button)
        self.sub_button_layout.addStretch(1)

    def show_randomize_paste_ui(self):
        self.sub_button_widget.show()
        self._clear_layout_from_widget(self.sub_button_layout)
        self.sub_button_widget.setTitle("Randomize Paste Options")

        form_layout = QGridLayout()
        self.rand_paste_widgets = {}

        def add_row(label_text, key, default_value, row_index, is_combo=False, items=None):
            label = QLabel(label_text)
            form_layout.addWidget(label, row_index, 0)
            value = self.last_random_paste_options.get(key, default_value)
            if is_combo:
                widget = QComboBox()
                widget.addItems(items or [])
                idx = widget.findText(value)
                widget.setCurrentIndex(idx if idx != -1 else 0)
            else:
                widget = QLineEdit(str(value)) # Ensure string for QLineEdit
            form_layout.addWidget(widget, row_index, 1)
            self.rand_paste_widgets[key] = widget

        default_to_idx = str(max(0, len(self.data) - 1)) if not self.data.empty else '0'
        add_row('From Index:', 'from_index', self.last_random_paste_options.get('from_index','0'), 0)
        add_row('To Index:', 'to_index', self.last_random_paste_options.get('to_index',default_to_idx), 1)
        add_row('Number of Pastes:', 'num_pastes', self.last_random_paste_options.get('num_pastes','1'), 2)
        add_row('Scale Type:', 'scale_type', self.last_random_paste_options.get('scale_type','Constant'), 3, True, ['Constant', 'Random'])
        add_row('Constant Scale:', 'constant_scale', self.last_random_paste_options.get('constant_scale','1.0'), 4)
        add_row('Random Scale Min:', 'random_scale_min', self.last_random_paste_options.get('random_scale_min','0.5'), 5)
        add_row('Random Scale Max:', 'random_scale_max', self.last_random_paste_options.get('random_scale_max','1.5'), 6)
        
        self.sub_button_layout.addLayout(form_layout)

        randomize_button = QPushButton('Randomize and Paste')
        randomize_button.clicked.connect(self.trigger_randomize_paste) # Connects to ActionsMixin
        self.sub_button_layout.addWidget(randomize_button)

        scale_type_combo = self.rand_paste_widgets['scale_type']

        def toggle_scale_fields():
            is_constant = scale_type_combo.currentText() == 'Constant'
            form_layout.itemAtPosition(4, 0).widget().setVisible(is_constant) # Constant Scale Label
            self.rand_paste_widgets['constant_scale'].setVisible(is_constant)
            form_layout.itemAtPosition(5, 0).widget().setVisible(not is_constant) # Random Min Label
            self.rand_paste_widgets['random_scale_min'].setVisible(not is_constant)
            form_layout.itemAtPosition(6, 0).widget().setVisible(not is_constant) # Random Max Label
            self.rand_paste_widgets['random_scale_max'].setVisible(not is_constant)

        scale_type_combo.currentTextChanged.connect(toggle_scale_fields)
        toggle_scale_fields() # Initial state
        self.sub_button_layout.addStretch(1)


    def _display_sub_buttons(self, title, buttons_config):
        self.sub_button_widget.show()
        self._clear_layout_from_widget(self.sub_button_layout)
        self.sub_button_widget.setTitle(title)
        for text, func in buttons_config:
            button = QPushButton(text)
            button.clicked.connect(func)
            self.sub_button_layout.addWidget(button)
        self.sub_button_layout.addStretch(1)

    def _clear_layout_from_widget(self, layout_to_clear):
        if layout_to_clear is None:
            return
        while layout_to_clear.count():
            item = layout_to_clear.takeAt(0)
            if item:
                widget = item.widget()
                sub_layout = item.layout()
                if widget:
                    widget.deleteLater()
                elif sub_layout:
                    self._clear_layout_from_widget(sub_layout) # Recurse for nested layouts

    def update_status_bar_text(self):
        # This method now assumes self.amplitude, self.phase, self.current_plot_prefixes, self.num_pairs are up-to-date
        # These variables are managed by DataMixin and the main class
        import numpy as np # Local import for safety
        if self.phase is None or self.amplitude is None or \
           not hasattr(self, 'current_plot_prefixes') or not self.current_plot_prefixes or \
           not hasattr(self, 'num_pairs') or self.num_pairs == 0:
            self.status_bar.setText("Phases: N/A")
            return

        parts = []
        for i in range(self.num_pairs):
            if i < len(self.current_plot_prefixes) and i < len(self.phase) and i < len(self.amplitude):
                prefix = self.current_plot_prefixes[i]
                amp_val = self.amplitude[i]
                phase_val = self.phase[i]
                
                amp_str = '%.2f' % amp_val if np.isfinite(amp_val) else 'N/A'
                phase_str = '%.2f°' % phase_val if np.isfinite(phase_val) else 'N/A'
                parts.append(f"P{prefix}: A={amp_str}, Ph={phase_str}")
        
        self.status_bar.setText("\n".join(parts) if parts else "Phases: N/A")

