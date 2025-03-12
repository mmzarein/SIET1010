from kivymd.uix.screen import MDScreen


class GeneralSettingsScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_enter(self):
        self.all_pass_value = self.manager.config_manager.getboolean(
            'SIET1010',
            'all_pass'
        )
        self.ids.all_pass_btn.icon = 'check' if self.all_pass_value else 'close'
        self.all_pass()

    def save(self):
        self.is_all_pass = False if self.ids.all_pass_btn.icon == 'check' else True
        self.manager.config_manager.set(
            'SIET1010',
            'all_pass',
            self.is_all_pass
        )
        if not self.is_all_pass:
            self.manager.config_manager.set(
                'SIET1010',
                'high_frequency',
                self.ids.high_freq_field.text
            )
            self.manager.config_manager.set(
                'SIET1010',
                'low_frequency',
                self.ids.low_freq_field.text
            )
        self.manager.config_manager.save()

    def all_pass(self):
        is_checked = self.ids.all_pass_btn.icon == 'check'

        self.ids.all_pass_btn.icon = 'close' if is_checked else 'check'
        bg_color = self.manager.app.STALE_BLUE if is_checked else self.manager.app.PURE_LIGHT
        text_color = self.manager.app.PURE_LIGHT if is_checked else self.manager.app.VOID_BLACK
        opacity = .6 if is_checked else 1
        disabled = is_checked

        self.ids.all_pass_btn.md_bg_color = bg_color
        self.ids.all_pass_btn.text_color = text_color
        self.ids.all_pass_btn.icon_color = text_color
        self.ids.frequencies_label.opacity = opacity
        self.ids.low_freq_field.disabled = disabled
        self.ids.high_freq_field.disabled = disabled

    def update_freq_labels(self, target, value):
        if target == 'low':
            self.ids.low_freq_label.text = f'Low (kHz): {value:.2f}'
            self.ids.high_min_label.text = f'Min: {value:.2f} kHz'
            self.ids.high_freq_slider.min = value
        else:
            self.ids.high_freq_label.text = f'High (kHz): {value:.2f}'


class AdvancedSettingsScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._is_off = True

    def on_enter(self):
        self.resolution = float(self.manager.config_manager.get(
            'SIET1010',
            'resolution'
        ))
        self.update_label('resolution', self.resolution)
        self.ids.resolution_slider.value = self.resolution

        self.sensitivity = float(self.manager.config_manager.get(
            'SIET1010',
            'sensitivity'
        ))
        self.update_label('sensitivity', self.sensitivity)
        self.ids.sensitivity_slider.value = self.sensitivity

    def back_to_general(self):
        self.manager.current = 'general_settings'
        self.manager.transition.direction = 'down'

    def save(self):
        self.manager.config_manager.set(
            'SIET1010', 'resolution', round(self.ids.resolution_slider.value, 2)
        )
        self.manager.config_manager.set(
            'SIET1010', 'sensitivity', round(self.ids.sensitivity_slider.value, 2)
        )
        self.manager.config_manager.save()

    def update_label(self, label, value):
        if label == 'resolution':
            self.ids.resolution_label.text = f'Resolution: {value:.1f}'
        else:
            self.ids.sensitivity_label.text = f'Sensitivity: {value:.1f}'
