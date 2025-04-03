import json
import os
from datetime import datetime
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.tab.tab import MDTabsBase, MDTabs
from kivymd.uix.floatlayout import MDFloatLayout
from kivy.properties import StringProperty, ObjectProperty
from kivymd.uix.button import MDFlatButton
from kivymd.uix.textfield import MDTextField
from kivymd.uix.dialog import MDDialog


class Tab(MDFloatLayout, MDTabsBase):
    '''Class implementing content for a tab.'''
    pass


class TabContainer(MDTabs):
    def __ini__(self, **kwargs):
        super().__init__(**kwargs)

    def on_tab_switch(self, *args):
        self.screen.update_calculation_button_status()


class ModulusScreen(MDScreen):
    bar_choice = StringProperty('')
    rod_choice = StringProperty('')

    def __ini__(self, **kwargs):
        super().__init__(**kwargs)

    def show_main_peak(self):
        self.ids.main_peak.text = f'Main Peak: {self.manager.app.main_peak:.5f} kHz'

    def on_enter(self):
        self.tab = self.ids.tab_container.get_current_tab().title
        self.show_main_peak()
        self.update_calculation_button_status()

    def change_bar_choice(self):
        bar_choices = {
            'poisson_bar_choice': 'poisson',
            'flexural_bar_choice': 'flexural',
            'torsional_bar_choice': 'torsional'
        }

        self.bar_choice = next(
            (value for key, value in bar_choices.items()
             if getattr(self.ids, key).state == 'down'),
            ''
        )

        self.update_calculation_button_status()

    def change_rod_choice(self):
        rod_choices = {
            'poisson_rod_choice': 'poisson',
            'flexural_rod_choice': 'flexural',
            'torsional_rod_choice': 'torsional'
        }

        self.rod_choice = next(
            (value for key, value in rod_choices.items()
             if getattr(self.ids, key).state == 'down'),
            ''
        )

        self.update_calculation_button_status()

    def open_save_dialog(self):
        current_time = datetime.now()

        if self.tab == 'BAR':
            file_name = current_time.strftime(f"SIET1010_%Y-%m-%d_%H-%M-%S_BAR.json")
        elif self.tab == 'ROD':
            file_name = current_time.strftime(f"SIET1010_%Y-%m-%d_%H-%M-%S_ROD.json")
        elif self.tab == 'DISC':
            file_name = current_time.strftime(f"SIET1010_%Y-%m-%d_%H-%M-%S_DISC.json")

        self.save_dialog = MDDialog(
            title='File Name:',
            type='custom',
            pos_hint={'center_x': .5, 'center_y': 0.8},
            content_cls=MDTextField(
                mode='round',
                text=file_name,
                icon_left='rename-box-outline'
            ),
            buttons=[
                MDFlatButton(
                    text='CANCEL',
                    text_color=self.theme_cls.error_color,
                    theme_text_color='Custom',
                    on_release=lambda _: self.save_dialog.dismiss()
                ),
                MDFlatButton(
                    text='OK',
                    text_color=self.theme_cls.primary_color,
                    theme_text_color='Custom',
                    on_release=lambda _: self.save(
                        self.save_dialog.content_cls.text
                    )
                )
            ]
        )
        self.save_dialog.open()

    def save(self, name):
        if self.tab == 'BAR':
            inputs = {
                'length': self.ids.bar_length.ids.label_field.text,
                'width': self.ids.bar_width.ids.label_field.text,
                'thickness': self.ids.bar_thickness.ids.label_field.text,
                'mass': self.ids.bar_mass.ids.label_field.text,
                'flexural_frequency': self.ids.bar_flexural_frequency.ids.label_field.text,
                'torsional_frequency': self.ids.bar_torsional_frequency.ids.label_field.text,
                'initial_poisson_ratio': self.ids.bar_initial_poisson_ratio.ids.label_field.text,
                'measurement_type': self.bar_choice
            }
            outputs = {
                'young_modulus': self.ids.bar_young_modulus_output.ids.label_field.text,
                'shear_modulus': self.ids.bar_shear_modulus_output.ids.label_field.text,
                'poisson_ratio': self.ids.bar_poisson_ratio_output.ids.label_field.text
            }

            with open(os.path.join(self.manager.default, name), 'w') as f:
                json.dump({'Inputs': inputs, 'Outputs': outputs}, f, indent=4)

        elif self.tab == 'ROD':
            inputs = {
                'length': self.ids.rod_length.ids.label_field.text,
                'diameter': self.ids.rod_diameter.ids.label_field.text,
                'mass': self.ids.rod_mass.ids.label_field.text,
                'flexural_frequency': self.ids.rod_flexural_frequency.ids.label_field.text,
                'torsional_frequency': self.ids.rod_torsional_frequency.ids.label_field.text,
                'initial_poisson_ratio': self.ids.rod_initial_poisson_ratio.ids.label_field.text,
                'measurement_type': self.rod_choice
            }
            outputs = {
                'young_modulus': self.ids.rod_young_modulus_output.ids.label_field.text,
                'shear_modulus': self.ids.rod_shear_modulus_output.ids.label_field.text,
                'poisson_ratio': self.ids.rod_poisson_ratio_output.ids.label_field.text
            }

            with open(os.path.join(self.manager.default, name), 'w') as f:
                json.dump({'Inputs': inputs, 'Outputs': outputs}, f, indent=4)

        elif self.tab == 'DISC':
            inputs = {
                'diameter': self.ids.disc_diameter.ids.label_field.text,
                'thickness': self.ids.disc_thickness.ids.label_field.text,
                'mass': self.ids.disc_mass.ids.label_field.text,
                'first_frequency': self.ids.disc_first_frequency.ids.label_field.text,
                'second_frequency': self.ids.disc_second_frequency.ids.label_field.text,
            }
            outputs = {
                'young_modulus': self.ids.disc_young_modulus_output.ids.label_field.text,
                'shear_modulus': self.ids.disc_shear_modulus_output.ids.label_field.text,
                'poisson_ratio': self.ids.disc_poisson_ratio_output.ids.label_field.text
            }

            with open(os.path.join(self.manager.default, name), 'w') as f:
                json.dump({'Inputs': inputs, 'Outputs': outputs}, f, indent=4)

        self.save_dialog.dismiss()

    def reset(self):
        if self.tab == 'BAR':
            for btn in [
                self.ids.poisson_bar_choice,
                self.ids.flexural_bar_choice,
                self.ids.torsional_bar_choice
            ]:
                btn.state = 'normal'
            self.bar_choice = ''

            for field in [
                self.ids.bar_length.ids.label_field,
                self.ids.bar_width.ids.label_field,
                self.ids.bar_thickness.ids.label_field,
                self.ids.bar_mass.ids.label_field,
                self.ids.bar_initial_poisson_ratio.ids.label_field,
                self.ids.bar_flexural_frequency.ids.label_field,
                self.ids.bar_torsional_frequency.ids.label_field
            ]:
                field.text = ''

            for field in [
                self.ids.bar_young_modulus_output.ids.label_field,
                self.ids.bar_shear_modulus_output.ids.label_field,
                self.ids.bar_poisson_ratio_output.ids.label_field
            ]:
                field.text = ''
        elif self.tab == 'ROD':
            for btn in [
                self.ids.poisson_rod_choice,
                self.ids.flexural_rod_choice,
                self.ids.torsional_rod_choice
            ]:
                btn.state = 'normal'
            self.rod_choice = ''

            for field in [
                self.ids.rod_length.ids.label_field,
                self.ids.rod_diameter.ids.label_field,
                self.ids.rod_mass.ids.label_field,
                self.ids.rod_initial_poisson_ratio.ids.label_field,
                self.ids.rod_flexural_frequency.ids.label_field,
                self.ids.rod_torsional_frequency.ids.label_field
            ]:
                field.text = ''

            for field in [
                self.ids.rod_young_modulus_output.ids.label_field,
                self.ids.rod_shear_modulus_output.ids.label_field,
                self.ids.rod_poisson_ratio_output.ids.label_field
            ]:
                field.text = ''

        elif self.tab == 'DISC':
            for field in [
                self.ids.disc_diameter.ids.label_field,
                self.ids.disc_thickness.ids.label_field,
                self.ids.disc_mass.ids.label_field,
                self.ids.disc_first_frequency.ids.label_field,
                self.ids.disc_second_frequency.ids.label_field
            ]:
                field.text = ''

            for field in [
                self.ids.disc_young_modulus_output.ids.label_field,
                self.ids.disc_shear_modulus_output.ids.label_field,
                self.ids.disc_poisson_ratio_output.ids.label_field
            ]:
                field.text = ''


        self.update_calculation_button_status()

    def update_calculation_button_status(self):
        self.tab = self.ids.tab_container.get_current_tab().title

        if self.tab == 'BAR':
            calculate_button = self.ids.bar_calculation_btn
            reset_button = self.ids.bar_reset_btn

            fields = [
                bool(self.ids.bar_length.ids.label_field.text),
                bool(self.ids.bar_width.ids.label_field.text),
                bool(self.ids.bar_thickness.ids.label_field.text),
                bool(self.ids.bar_mass.ids.label_field.text),
                bool(self.ids.bar_initial_poisson_ratio.ids.label_field.text),
                bool(self.bar_choice)
            ]

            output_fields = [
                bool(self.ids.bar_young_modulus_output.ids.label_field.text),
                bool(self.ids.bar_shear_modulus_output.ids.label_field.text),
                bool(self.ids.bar_poisson_ratio_output.ids.label_field.text)
            ]

            if self.bar_choice == 'flexural':
                fields.append(bool(self.ids.bar_flexural_frequency.ids.label_field.text))
            elif self.bar_choice == 'torsional':
                fields.append(bool(self.ids.bar_torsional_frequency.ids.label_field.text))
            else:
                fields.extend([
                    bool(self.ids.bar_flexural_frequency.ids.label_field.text),
                    bool(self.ids.bar_torsional_frequency.ids.label_field.text)
                ])

            calculate_button.disabled = not all(fields)
            reset_button.disabled = not any(fields)

            self.ids.modulus_panel.ids.right_button.disabled = not any(
                output_fields)

        elif self.tab == 'ROD':
            calculate_button = self.ids.rod_calculation_btn
            reset_button = self.ids.rod_reset_btn

            fields = [
                bool(self.ids.rod_length.ids.label_field.text),
                bool(self.ids.rod_diameter.ids.label_field.text),
                bool(self.ids.rod_mass.ids.label_field.text),
                bool(self.ids.rod_initial_poisson_ratio.ids.label_field.text),
                bool(self.rod_choice)
            ]

            output_fields = [
                bool(self.ids.rod_young_modulus_output.ids.label_field.text),
                bool(self.ids.rod_shear_modulus_output.ids.label_field.text),
                bool(self.ids.rod_poisson_ratio_output.ids.label_field.text)
            ]

            if self.rod_choice == 'flexural':
                fields.append(bool(self.ids.rod_flexural_frequency.ids.label_field.text))
            elif self.rod_choice == 'torsional':
                fields.append(bool(self.ids.rod_torsional_frequency.ids.label_field.text))
            else:
                fields.extend([
                    bool(self.ids.rod_flexural_frequency.ids.label_field.text),
                    bool(self.ids.rod_torsional_frequency.ids.label_field.text)
                ])

            calculate_button.disabled = not all(fields)
            reset_button.disabled = not any(fields)

            self.ids.modulus_panel.ids.right_button.disabled = not any(
                output_fields)

        elif self.tab == 'DISC':
            calculate_button = self.ids.disc_calculation_btn
            reset_button = self.ids.disc_reset_btn

            fields = [
                bool(self.ids.disc_thickness.ids.label_field.text),
                bool(self.ids.disc_diameter.ids.label_field.text),
                bool(self.ids.disc_mass.ids.label_field.text),
                bool(self.ids.disc_first_frequency.ids.label_field.text),
                bool(self.ids.disc_second_frequency.ids.label_field.text),
            ]

            output_fields = [
                bool(self.ids.disc_young_modulus_output.ids.label_field.text),
                bool(self.ids.disc_shear_modulus_output.ids.label_field.text),
                bool(self.ids.disc_poisson_ratio_output.ids.label_field.text)
            ]

            calculate_button.disabled = not all(fields)
            reset_button.disabled = not any(fields)

            self.ids.modulus_panel.ids.right_button.disabled = not any(
                output_fields)

    def bar_calculation(self):
        inputs = {
            'length': self.ids.bar_length.ids.label_field.text,
            'width': self.ids.bar_width.ids.label_field.text,
            'thickness': self.ids.bar_thickness.ids.label_field.text,
            'mass': self.ids.bar_mass.ids.label_field.text,
            'flexural_frequency': self.ids.bar_flexural_frequency.ids.label_field.text,
            'torsional_frequency': self.ids.bar_torsional_frequency.ids.label_field.text,
            'initial_poisson_ratio': self.ids.bar_initial_poisson_ratio.ids.label_field.text,
            'measurement_type': self.bar_choice
        }

        result = self.manager.calculator.bar(**inputs)
        self.ids.bar_young_modulus_output.ids.label_field.text = result['dynamic_young_modulus_output']
        self.ids.bar_shear_modulus_output.ids.label_field.text = result['dynamic_shear_modulus_output']
        self.ids.bar_poisson_ratio_output.ids.label_field.text = result['poisson_ratio_output']

        self.update_calculation_button_status()

    def rod_calculation(self):
        inputs = {
            'length': self.ids.rod_length.ids.label_field.text,
            'diameter': self.ids.rod_diameter.ids.label_field.text,
            'mass': self.ids.rod_mass.ids.label_field.text,
            'flexural_frequency': self.ids.rod_flexural_frequency.ids.label_field.text,
            'torsional_frequency': self.ids.rod_torsional_frequency.ids.label_field.text,
            'initial_poisson_ratio': self.ids.rod_initial_poisson_ratio.ids.label_field.text,
            'measurement_type': self.rod_choice
        }

        result = self.manager.calculator.rod(**inputs)
        self.ids.rod_young_modulus_output.ids.label_field.text = result['dynamic_young_modulus_output']
        self.ids.rod_shear_modulus_output.ids.label_field.text = result['dynamic_shear_modulus_output']
        self.ids.rod_poisson_ratio_output.ids.label_field.text = result['poisson_ratio_output']

        self.update_calculation_button_status()

    def disc_calculation(self):
        inputs = {
            'diameter': self.ids.disc_diameter.ids.label_field.text,
            'mass': self.ids.disc_mass.ids.label_field.text,
            'thickness': self.ids.disc_thickness.ids.label_field.text,
            'first_frequency': self.ids.disc_first_frequency.ids.label_field.text,
            'second_frequency': self.ids.disc_second_frequency.ids.label_field.text,
        }

        result = self.manager.calculator.disc(**inputs)
        self.ids.disc_young_modulus_output.ids.label_field.text = result['dynamic_young_modulus_output']
        self.ids.disc_shear_modulus_output.ids.label_field.text = result['dynamic_shear_modulus_output']
        self.ids.disc_poisson_ratio_output.ids.label_field.text = result['poisson_ratio_output']

        self.update_calculation_button_status()

