import os
import shutil
import subprocess
import threading
from kivy.clock import mainthread
from kivymd.uix.screen import MDScreen
from kivymd.uix.dialog import MDDialog
from kivymd.uix.textfield import MDTextField


class Updater:
    def __init__(self, clone_dir):
        self.clone_dir = clone_dir

        self.update_dialog = MDDialog(
            title='Update',
            text='The app is updating itself in the most inefficient and insecure way possible. Do not close the app, for God\'s sake!'
        )

    def update(self):
        threading.Thread(target=self._perform_update, daemon=True).start()
        self.update_dialog.open()

    def _perform_update(self):
        try:
            self._clone_or_pull_repo()
            self._install_dependencies()
        except Exception as e:
            print(f"Update failed: {e}")

    def _clone_or_pull_repo(self):
        os.system('rm -rf ./update')
        self.update_dialog.text = 'Cloning from GitHub!'
        os.system('git clone https://github.com/mmzarein/SIET1010.git ./update')
        os.system('cd ./update/ && git checkout refactor && cd ..')
        self.update_dialog.text = 'Installing the update!'
        os.system(f'rsync -av --remove-source-files ./update/* {self.clone_dir}')
        self.update_dialog.text = 'Clean Up!'
        os.system('rm -rf ./update')

    def _install_dependencies(self):
        requirements_path = os.path.join(self.clone_dir, "requirements.txt")
        if os.path.exists(requirements_path):
            print("Installing dependencies...")
            self.update_dialog.text = 'Installing the Dependencies!'
            subprocess.run(["pip", "install", "-r", requirements_path], check=True)
        else:
            print("No requirements.txt found, skipping dependency installation.")

        self.update_dialog.text = 'Done!'


class GeneralSettingsScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_enter(self):
        self.low_frequency = float(self.manager.config_manager.get(
            'SIET1010',
            'low_frequency'
        ))
        self.high_frequency = float(self.manager.config_manager.get(
            'SIET1010',
            'high_frequency'
        ))
        self.update_label('low_freq_label', self.low_frequency)
        self.ids.low_freq.value = self.low_frequency
        self.update_label('high_freq_label', self.high_frequency)
        self.ids.high_freq.value = self.high_frequency
        self.all_pass_value = self.manager.config_manager.getboolean(
            'SIET1010',
            'all_pass'
        )
        self.ids.all_pass_btn.icon = 'check' if self.all_pass_value else 'close'
        self.all_pass()

    def save(self):
        self.manager.config_manager.set(
            'SIET1010', 'low_frequency', round(self.ids.low_freq.value, 2)
        )
        self.manager.config_manager.set(
            'SIET1010', 'high_frequency', round(self.ids.high_freq.value, 2)
        )
        self.manager.config_manager.save()

    def update_label(self, label, value):
        if label == 'low_freq_label':
            self.ids.low_freq_label.text = f'Low Frequency: {value:.2f} kHz'
            self.ids.high_freq.min = value
            self.ids.high_freq_min.text = f'Min: {value:.2f} kHz'
        else:
            self.ids.high_freq_label.text = f'High Frequency: {value:.2f} kHz'
            self.ids.low_freq.max = value
            self.ids.low_freq_max.text = f'Max: {value:.2f} kHz'

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
        self.ids.low_freq.disabled = disabled
        self.ids.high_freq.disabled = disabled


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
        elif label == 'sensitivity':
            self.ids.sensitivity_label.text = f'Sensitivity: ± {value:.1f}'

    def update(self):
        # Define your repository details
        CLONE_DIR = "/home/pi/SIET1010/"

        updater = Updater(CLONE_DIR)

        updater.update()

