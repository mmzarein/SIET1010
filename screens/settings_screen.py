import os
import shutil
import subprocess
import threading
import git
from kivy.app import App
from kivy.clock import mainthread
from kivymd.uix.screen import MDScreen

class Updater:
    def __init__(self, repo_url, branch, username, token, clone_dir):
        self.repo_url = repo_url.replace("https://", f"https://{username}:{token}@")
        self.branch = branch
        self.clone_dir = clone_dir

    def update(self):
        threading.Thread(target=self._perform_update, daemon=True).start()

    def _perform_update(self):
        try:
            self._clone_or_pull_repo()
            self._install_dependencies()
        except Exception as e:
            print(f"Update failed: {e}")

    def _clone_or_pull_repo(self):
        if os.path.exists(self.clone_dir):
            print("Repository exists, pulling latest changes...")
            repo = git.Repo(self.clone_dir)
            repo.git.checkout(self.branch)
            repo.remotes.origin.pull()
        else:
            print("Cloning repository...")
            git.Repo.clone_from(self.repo_url, self.clone_dir, branch=self.branch)

    def _install_dependencies(self):
        requirements_path = os.path.join(self.clone_dir, "requirements.txt")
        if os.path.exists(requirements_path):
            print("Installing dependencies...")
            subprocess.run(["pip", "install", "-r", requirements_path], check=True)
        else:
            print("No requirements.txt found, skipping dependency installation.")

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

    def update(self):
        # Define your repository details
        REPO_URL = "https://github.com/mmzarein/SIET1010.git"
        BRANCH = "refactor"
        USERNAME = "mmzarein"
        TOKEN = os.getenv('GITHUB_TOKEN')
        CLONE_DIR = "/home/pipi/SIET1010/"

        updater = Updater(REPO_URL, BRANCH, USERNAME, TOKEN, CLONE_DIR)

        updater.update()

