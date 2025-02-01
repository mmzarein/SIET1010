import sys
import shutil
from pathlib import Path
from kivy.core.window import Window
from kivy.properties import ColorProperty
from kivymd.app import MDApp
from kivymd.uix.screenmanager import MDScreenManager
from kivy.uix.vkeyboard import VKeyboard


def setup_keyboard_layouts():
    # Get the Python version dynamically
    python_version = sys.version_info
    python_version_str = f"python{python_version.major}.{python_version.minor}"

    # Break the path construction into multiple lines for readability
    keyboard_path = (
        Path(sys.prefix) / 'lib' / python_version_str / 'site-packages' /
        'kivy' / 'data' / 'keyboards' / 'minimal.json'
    )

    # Check if the file exists and print a message
    if keyboard_path.exists():
        VKeyboard.layout = 'minimal'
    else:
        source = 'keyboards/minimal/minimal.json'
        destination = keyboard_path
        try: shutil.copy(source, destination)
        except Exception as e: print(f"Error: {e}")
        finally: VKeyboard.layout = 'minimal'


DEVELOPMENT = False

if DEVELOPMENT:
    from screeninfo import get_monitors
    monitors = get_monitors()

    if len(monitors) >= 1:
        Window.left = monitors[1].x + 100
        Window.top = monitors[1].y + 100


class Navigator(MDScreenManager):
    snow = ColorProperty((1, 1, 1, 1))
    stale_blue = ColorProperty((.23, .29, .36, 1))
    stale_gray = ColorProperty([.85, .85, .90, 1])

    def __init__(self):
        super().__init__()

        self.current = 'archive' # TMP

        # TODO: Ensure the path exists; create the directory if it doesn't.
        # TODO: Store default path and settings in persistent storage!
        self.default = 'Archive/SIET1010'


class Main(MDApp):
    def build(self):
        return Navigator()


if __name__ == '__main__':
    Window.borderless = True
    Window.size = 1024, 600
    setup_keyboard_layouts()
    Main().run()

