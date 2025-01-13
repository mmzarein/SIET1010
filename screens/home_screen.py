from kivy.lang import Builder
from kivymd.uix.screen import MDScreen
from kivymd.uix.label import MDLabel


class HomeScreen(MDScreen):

    Builder.load_file('designs/custom.kv')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_kv_post(self, base):
        self.initialize_peaks()

    def initialize_peaks(self):
        self.ids.first_peak.text = f'{"First Peak:":<15}---{"":<2}kHz'
        self.ids.second_peak.text = f'{"Second Peak:":<15}---{"":<2}kHz'
        self.ids.third_peak.text = f'{"Third Peak:":<15}---{"":<2}kHz'

