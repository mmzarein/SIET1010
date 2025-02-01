import os
import re
import mimetypes
import shutil
import math
from types import MethodType

from kivy.metrics import dp
from kivy.core.window import Window
from kivymd.uix.screen import MDScreen
from kivymd.uix.textfield import MDTextField
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton
from kivymd.uix.datatables import MDDataTable
from kivymd.uix.datatables.datatables import (
    TableData,
    TableRecycleGridLayout
)


TableRecycleGridLayout.select_row = MethodType(
    lambda self, _: None,
    TableRecycleGridLayout
)


def _get_checked_rows(self):
    checked_rows = []
    total_cols = self.total_col_headings
    for page in self.current_selection_check:
        # Skip if the page no longer exists (e.g., after data change)
        if page >= len(self._row_data_parts):
            continue
        page_data = self._row_data_parts[page]
        for index in self.current_selection_check[page]:
            row_in_page = index // total_cols
            # Ensure the row index is within the current page's data
            if row_in_page < len(page_data):
                checked_rows.append(page_data[row_in_page])
    return checked_rows


class CustomDataTable(MDDataTable):
    def __init__(self, screen, **kwargs):
        super().__init__(**kwargs)
        self.screen = screen
        self.stop_propagation = False
        self.table_data._get_row_checks = MethodType(
            _get_checked_rows,
            self.table_data
        )

    def on_row_press(self, cell):
        Window.release_all_keyboards()
        try:
            if not self.stop_propagation:
                path_candidate = os.path.join(
                    self.screen.current_path, cell.text)
                if os.path.isdir(path_candidate):
                    self.screen.change_directory(path_candidate)
        finally:
            # Always reset the flag after handling the row press!
            self.stop_propagation = False

    def on_check_press(self, row):
        self.stop_propagation = True
        self.screen.update_buttons_state()


class ArchiveScreen(MDScreen):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.table = None
        self.archive_path = 'Archive/'
        self.current_path = self.archive_path

    def on_enter(self):
        if not self.table:
            self.setup_table()
            self.update_breadcrumbs()
        self.update_buttons_state()
        self.update_new_folder_button_state()

    def setup_table(self):
        self.table = CustomDataTable(
            screen=self,
            elevation=0,
            check=True,
            use_pagination=True,
            background_color_header=(.23, .29, .36, .5),
            row_data=self.list_contents(),
            column_data=[
                ('Name', dp(100)),
                ('Type', dp(30)),
                ('Size', dp(20))
            ],
        )
        self.ids.data_table.add_widget(self.table)

    def update_breadcrumbs(self):
        self.ids.breadcrumb.text = self.current_path

    def change_directory(self, target):
        self.current_path = target
        self.update_breadcrumbs()
        self.refresh()

    def refresh(self):
        self.table.update_row_data(
            self.table,
            self.list_contents()
        )
        self.update_buttons_state()
        self.update_new_folder_button_state()

    def back(self):
        norm_current = os.path.normpath(self.current_path)
        norm_archive = os.path.normpath(self.archive_path)
        if norm_current != norm_archive:
            self.current_path = os.path.dirname(self.current_path)
            self.update_breadcrumbs()
            self.refresh()
        else:
            self.manager.current = 'home'
            self.manager.transition.direction = 'right'

    def create_directory(self):
        os.makedirs(
            os.path.join(
                self.current_path,
                self.create_directory_dialog.content_cls.text
            ),
            exist_ok=True
        )
        self.create_directory_dialog.dismiss()
        self.refresh()

    def rename_item(self, old_path, new_name):
        new_path = os.path.join(self.current_path, new_name)
        os.rename(old_path, new_path)
        self.rename_dialog.dismiss()
        self.refresh()

    def delete_items(self):
        for path in self.files_to_delete:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        self.delete_dialog.dismiss()
        self.refresh()

    def set_default(self, new_path):
        self.manager.default = new_path
        self.set_default_dialog.dismiss()
        self.refresh()
        self.update_buttons_state()

    def show_create_directory_dialog(self):
        self.create_directory_dialog = MDDialog(
            title='New Folder',
            type='custom',
            pos_hint={'center_x': .5, 'center_y': .8},
            content_cls=MDTextField(mode='round'),
            buttons=[
                MDFlatButton(
                    text='CANCEL',
                    # Only available KivyMD 1.2.0!
                    text_color=self.theme_cls.error_color,
                    theme_text_color='Custom',
                    on_release=lambda _: self.create_directory_dialog.dismiss()
                ),
                MDFlatButton(
                    text='OK',
                    # Only available in KivyMD 1.2.0!
                    text_color=self.theme_cls.primary_color,
                    theme_text_color='Custom',
                    on_release=lambda _: self.create_directory()
                )
            ]
        )
        self.create_directory_dialog.open()

    def show_rename_dialog(self):
        selected_rows = self.table.get_row_checks()
        if not selected_rows or len(selected_rows) != 1: return
        old_name = selected_rows[0][0]
        old_path = os.path.join(self.current_path, old_name)
        if old_path == self.manager.default: return
        self.rename_dialog = MDDialog(
            title='Rename',
            type='custom',
            pos_hint={'center_x': .5, 'center_y': 0.8},
            content_cls=MDTextField(
                mode='round',
                text=old_name
            ),
            buttons=[
                MDFlatButton(
                    text='CANCEL',
                    text_color=self.theme_cls.primary_color,
                    theme_text_color='Custom',
                    on_release=lambda _: self.rename_dialog.dismiss()
                ),
                MDFlatButton(
                    text='OK',
                    text_color=self.theme_cls.primary_color,
                    theme_text_color='Custom',
                    on_release=lambda _: self.rename_item(
                        old_path,
                        self.rename_dialog.content_cls.text
                    ),
                )
            ]
        )
        self.rename_dialog.open()

    def show_delete_dialog(self):
        # TODO: This function may need some improvments!
        selected_rows = self.table.get_row_checks()
        if not selected_rows: return
        self.files_to_delete = [
            os.path.join(self.current_path, row[0])
            for row in selected_rows
            if os.path.normpath(row[0]) != self.manager.default
            and re.sub(r'\[.*?\]', '', row[1]).strip() != 'Default Folder'
        ]
        if len(self.files_to_delete) > 20:
            message = 'More than 20 files is going to deleted!'
        else:
            message = '\n'.join(self.files_to_delete)
        self.delete_dialog = MDDialog(
            title='Delete',
            text=message,
            buttons=[
                MDFlatButton(
                    text='CANCEL',
                    # Only available KivyMD 1.2.0!
                    text_color=self.theme_cls.error_color,
                    theme_text_color='Custom',
                    on_release=lambda _: self.delete_dialog.dismiss()
                ),
                MDFlatButton(
                    text='OK',
                    # Only available in KivyMD 1.2.0!
                    text_color=self.theme_cls.primary_color,
                    theme_text_color='Custom',
                    on_release=lambda _: self.delete_items()
                )
            ]
        )
        self.delete_dialog.open()

    def show_set_default_dialog(self):
        selected_rows = self.table.get_row_checks()
        if not selected_rows or len(selected_rows) != 1: return
        default_candidate = os.path.join(self.current_path, selected_rows[0][0])
        if default_candidate == self.manager.default: return
        if os.path.isfile(default_candidate): return
        self.set_default_dialog = MDDialog(
            title='Set as Default Folder?',
            text=default_candidate,
            buttons=[
                MDFlatButton(
                    text='NO',
                    text_color=self.theme_cls.error_color,
                    theme_text_color='Custom',
                    on_release=lambda _: self.set_default_dialog.dismiss()
                ),
                MDFlatButton(
                    text='YES',
                    text_color=self.theme_cls.primary_color,
                    theme_text_color='Custom',
                    on_release=lambda _: self.set_default(default_candidate)
                )
            ]
        )
        self.set_default_dialog.open()

    def list_contents(self):
        try:
            self.contents = []
            with os.scandir(self.current_path) as entries:
                for entry in entries:
                    if entry.is_dir():
                        file_type = (
                            f'[color=#FF0000]Default Folder[/color]'
                            if entry.path == self.manager.default
                            else 'Folder'
                        )
                        size = ArchiveScreen.get_directory_size(entry.path)
                    else:
                        file_type, _ = mimetypes.guess_type(entry.name)
                        file_type = file_type or 'Unknown'
                        size = entry.stat().st_size
                    self.contents.append((
                        entry.name,
                        file_type,
                        ArchiveScreen.human_readable_size(size)
                    ))
            return self.contents
        # TODO: Notify the user about `PermissionError` and `FileNotFoundErrors`.
        except FileNotFoundError: return []
        except PermissionError: return []

    def can_create_new_folder(self):
        try:
            count = sum(1 for entry in os.scandir(self.current_path) if entry.is_dir())
            return count < 5
        except (PermissionError, FileNotFoundError): return False

    def update_new_folder_button_state(self):
        self.ids.new_folder_button.disabled = not self.can_create_new_folder()

    def update_buttons_state(self):
        selected = self.table.get_row_checks()
        num_selected = len(selected)

        # Rename button: Enable if exactly one row selected and not Default Folder!
        if num_selected == 1:
            selected_path = os.path.join(self.current_path, selected[0][0])
            if selected_path != self.manager.default:
                self.ids.rename_button.disabled = False
        else:
            self.ids.rename_button.disabled = True

        # Delete button: Enable if any selected, but not the Default Folder
        delete_disabled = True
        if num_selected > 0:
            default_path = self.manager.default
            has_default = any(os.path.join(self.current_path, row[0]) == default_path for row in selected)
            delete_disabled = has_default
        self.ids.delete_button.disabled = delete_disabled

        # Set Default button: Enable if one folder selected, not the current default
        set_default_disabled = True
        if num_selected == 1:
            name = selected[0][0]
            item_path = os.path.join(self.current_path, name)
            if os.path.isdir(item_path) and item_path != self.manager.default:
                set_default_disabled = False
        self.ids.set_default_button.disabled = set_default_disabled

    @staticmethod
    def get_directory_size(directory):
        total_size = 0
        for entry in os.scandir(directory):
            if entry.is_dir():
                total_size += get_directory_size(entry.path)
            else:
                total_size += entry.stat().st_size
        return total_size

    @staticmethod
    def human_readable_size(size_in_bytes):
        if size_in_bytes == 0: return '0 B'
        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        power = int(math.log(size_in_bytes, 1024))
        power = min(power, len(units) - 1)
        size = size_in_bytes / (1024 ** power)
        return f'{size:.2f} {units[power]}'

