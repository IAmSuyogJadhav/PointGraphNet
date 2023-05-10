import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import platform
import random
import threading
import time
from inference import *

isMacOS = (platform.system() == "Darwin")

# Create a simple GUI for the inference.py script

class DemoGUI(gui.Application):
    def __init__(
