from kivy.core.window import Window
from kivy.properties import ColorProperty
from kivymd.app import MDApp
from kivymd.uix.screenmanager import MDScreenManager


class Navigator(MDScreenManager):
    snow = ColorProperty((1, 1, 1, 1))
    stale_blue = ColorProperty((.23, .29, .36, 1))
    stale_gray = ColorProperty([.85, .85, .90, 1])

    def __init__(self):
        super().__init__()


class Main(MDApp):
    def build(self):
        return Navigator()


if __name__ == '__main__':
    Window.borderless = True
    Window.size = 1024, 600
    Main().run()

