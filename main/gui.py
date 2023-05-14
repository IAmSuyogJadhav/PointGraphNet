# PointNormalNet Demo, adapted from https://github.com/isl-org/Open3D/blob/master/examples/python/visualization/vis_gui.py
# Modified by: Suyog Jadhav, 10-05-2023

# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import glob
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import platform
import sys
from inference import *
import copy
import sys

if len(sys.argv) > 1 and sys.argv[1] == 'debug':
    DEBUG = True
else:
    DEBUG = False

isMacOS = platform.system() == "Darwin"
MAX_N_LIMITS = (1000, 200_000)
DEPTH_LIMITS = (1, 24)
CLEAN_MESH_AREA_THRESH = 0.1

class Settings:
    UNLIT = "defaultUnlit"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"

    DEFAULT_PROFILE_NAME = "Bright day with sun at +Y [default]"
    POINT_CLOUD_PROFILE_NAME = "Cloudy day (no direct sun)"
    CUSTOM_PROFILE_NAME = "Custom"
    LIGHTING_PROFILES = {
        DEFAULT_PROFILE_NAME: {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at -Y": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at +Z": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at -Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Z": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        POINT_CLOUD_PROFILE_NAME: {
            "ibl_intensity": 60000,
            "sun_intensity": 50000,
            "use_ibl": True,
            "use_sun": False,
            # "ibl_rotation":
        },
    }

    DEFAULT_MATERIAL_NAME = "Polished ceramic [default]"
    PREFAB = {
        DEFAULT_MATERIAL_NAME: {
            "metallic": 0.0,
            "roughness": 0.7,
            "reflectance": 0.5,
            "clearcoat": 0.2,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0,
        },
        "Metal (rougher)": {
            "metallic": 1.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0,
        },
        "Metal (smoother)": {
            "metallic": 1.0,
            "roughness": 0.3,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0,
        },
        "Plastic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.5,
            "clearcoat": 0.5,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0,
        },
        "Glazed ceramic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 1.0,
            "clearcoat_roughness": 0.1,
            "anisotropy": 0.0,
        },
        "Clay": {
            "metallic": 0.0,
            "roughness": 1.0,
            "reflectance": 0.5,
            "clearcoat": 0.1,
            "clearcoat_roughness": 0.287,
            "anisotropy": 0.0,
        },
    }

    def __init__(self):
        self.mouse_model = gui.SceneWidget.Controls.ROTATE_CAMERA
        self.bg_color = gui.Color(1, 1, 1)
        self.show_skybox = False
        self.show_axes = False
        self.use_ibl = True
        self.use_sun = True
        self.new_ibl_name = None  # clear to None after loading
        self.ibl_intensity = 45000
        self.sun_intensity = 45000
        self.sun_dir = [0.577, -0.577, -0.577]
        self.sun_color = gui.Color(1, 1, 1)

        self.apply_material = True  # clear to False after processing
        self._materials = {
            Settings.LIT: rendering.MaterialRecord(),
            Settings.UNLIT: rendering.MaterialRecord(),
            Settings.NORMALS: rendering.MaterialRecord(),
            Settings.DEPTH: rendering.MaterialRecord(),
        }

        self._materials[Settings.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.LIT].has_alpha = True  #DEBUG
        self._materials[Settings.LIT].shader = Settings.LIT
        self._materials[Settings.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.UNLIT].shader = Settings.UNLIT
        self._materials[Settings.NORMALS].shader = Settings.NORMALS
        self._materials[Settings.DEPTH].shader = Settings.DEPTH

        # Conveniently, assigning from self._materials[...] assigns a reference,
        # not a copy, so if we change the property of a material, then switch
        # to another one, then come back, the old setting will still be there.
        self.material = self._materials[Settings.LIT]

    def set_material(self, name):
        self.material = self._materials[name]
        self.apply_material = True

    def apply_material_prefab(self, name):
        assert self.material.shader == Settings.LIT
        prefab = Settings.PREFAB[name]
        for key, val in prefab.items():
            setattr(self.material, "base_" + key, val)

    def apply_lighting_profile(self, name):
        profile = Settings.LIGHTING_PROFILES[name]
        for key, val in profile.items():
            setattr(self, key, val)


class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21
    MENU_LOAD_MODEL = 31
    MENU_LOAD_POINTS = 32
    MENU_EXPORT_POINTS = 33
    MENU_EXPORT_POINTS_CSV = 34
    MENU_EXPORT_POINTS_NOMESH = 35
    MENU_CONFIGURE = 41

    DEFAULT_IBL = "default"

    MATERIAL_NAMES = ["Lit", "Unlit", "Normals", "Depth"]
    MATERIAL_SHADERS = [Settings.LIT, Settings.UNLIT, Settings.NORMALS, Settings.DEPTH]

    def __init__(self, width, height):
        self.settings = Settings()
        resource_path = gui.Application.instance.resource_path
        self.settings.new_ibl_name = resource_path + "/" + AppWindow.DEFAULT_IBL

        self.window = gui.Application.instance.create_window(
            "PointNormalNet - Open3D" if not DEBUG \
                else f"PointNormalNet - Open3D [{os.path.basename(DEFAULT_CKPT_DIR)}]", 
            width, height
        )
        w = self.window  # to make the code more concise

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.set_on_sun_direction_changed(self._on_sun_dir)

        # ---- Settings panel ----
        # Rather than specifying sizes in pixels, which may vary in size based
        # on the monitor, especially on macOS which has 220 dpi monitors, use
        # the em-size. This way sizings will be proportional to the font size,
        # which will create a more visually consistent size across platforms.
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        # Widgets are laid out in layouts: gui.Horiz, gui.Vert,
        # gui.CollapsableVert, and gui.VGrid. By nesting the layouts we can
        # achieve complex designs. Usually we use a vertical layout as the
        # topmost widget, since widgets tend to be organized from top to bottom.
        # Within that, we usually have a series of horizontal layouts for each
        # row. All layouts take a spacing parameter, which is the spacing
        # between items in the widget, and a margins parameter, which specifies
        # the spacing of the left, top, right, bottom margins. (This acts like
        # the 'padding' property in CSS.)
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em)
        )

        # Create a collapsible vertical widget, which takes up enough vertical
        # space for all its children when open, but only enough for text when
        # closed. This is useful for property pages, so the user can hide sets
        # of properties they rarely use.
        view_ctrls = gui.CollapsableVert(
            "View controls", 0.25 * em, gui.Margins(em, 0, 0, 0)
        )

        self._arcball_button = gui.Button("Arcball")
        self._arcball_button.horizontal_padding_em = 0.5
        self._arcball_button.vertical_padding_em = 0
        self._arcball_button.set_on_clicked(self._set_mouse_mode_rotate)
        self._fly_button = gui.Button("Fly")
        self._fly_button.horizontal_padding_em = 0.5
        self._fly_button.vertical_padding_em = 0
        self._fly_button.set_on_clicked(self._set_mouse_mode_fly)
        self._model_button = gui.Button("Model")
        self._model_button.horizontal_padding_em = 0.5
        self._model_button.vertical_padding_em = 0
        self._model_button.set_on_clicked(self._set_mouse_mode_model)
        self._sun_button = gui.Button("Sun")
        self._sun_button.horizontal_padding_em = 0.5
        self._sun_button.vertical_padding_em = 0
        self._sun_button.set_on_clicked(self._set_mouse_mode_sun)
        self._ibl_button = gui.Button("Environment")
        self._ibl_button.horizontal_padding_em = 0.5
        self._ibl_button.vertical_padding_em = 0
        self._ibl_button.set_on_clicked(self._set_mouse_mode_ibl)
        view_ctrls.add_child(gui.Label("Mouse controls"))
        # We want two rows of buttons, so make two horizontal layouts. We also
        # want the buttons centered, which we can do be putting a stretch item
        # as the first and last item. Stretch items take up as much space as
        # possible, and since there are two, they will each take half the extra
        # space, thus centering the buttons.
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._arcball_button)
        h.add_child(self._fly_button)
        h.add_child(self._model_button)
        h.add_stretch()
        view_ctrls.add_child(h)
        h = gui.Horiz(0.25 * em)  # row 2
        h.add_stretch()
        h.add_child(self._sun_button)
        h.add_child(self._ibl_button)
        h.add_stretch()
        view_ctrls.add_child(h)

        self._show_skybox = gui.Checkbox("Show skymap")
        self._show_skybox.set_on_checked(self._on_show_skybox)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_skybox)

        self._bg_color = gui.ColorEdit()
        self._bg_color.set_on_value_changed(self._on_bg_color)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("BG Color"))
        grid.add_child(self._bg_color)
        view_ctrls.add_child(grid)

        self._show_axes = gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_axes)

        self._profiles = gui.Combobox()
        for name in sorted(Settings.LIGHTING_PROFILES.keys()):
            self._profiles.add_item(name)
        self._profiles.add_item(Settings.CUSTOM_PROFILE_NAME)
        self._profiles.set_on_selection_changed(self._on_lighting_profile)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(gui.Label("Lighting profiles"))
        view_ctrls.add_child(self._profiles)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(view_ctrls)

        advanced = gui.CollapsableVert("Advanced lighting", 0, gui.Margins(em, 0, 0, 0))
        advanced.set_is_open(False)

        self._use_ibl = gui.Checkbox("HDR map")
        self._use_ibl.set_on_checked(self._on_use_ibl)
        self._use_sun = gui.Checkbox("Sun")
        self._use_sun.set_on_checked(self._on_use_sun)
        advanced.add_child(gui.Label("Light sources"))
        h = gui.Horiz(em)
        h.add_child(self._use_ibl)
        h.add_child(self._use_sun)
        advanced.add_child(h)

        self._ibl_map = gui.Combobox()
        for ibl in glob.glob(gui.Application.instance.resource_path + "/*_ibl.ktx"):

            self._ibl_map.add_item(os.path.basename(ibl[:-8]))
        self._ibl_map.selected_text = AppWindow.DEFAULT_IBL
        self._ibl_map.set_on_selection_changed(self._on_new_ibl)
        self._ibl_intensity = gui.Slider(gui.Slider.INT)
        self._ibl_intensity.set_limits(0, 200000)
        self._ibl_intensity.set_on_value_changed(self._on_ibl_intensity)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("HDR map"))
        grid.add_child(self._ibl_map)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._ibl_intensity)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label("Environment"))
        advanced.add_child(grid)

        self._sun_intensity = gui.Slider(gui.Slider.INT)
        self._sun_intensity.set_limits(0, 200000)
        self._sun_intensity.set_on_value_changed(self._on_sun_intensity)
        self._sun_dir = gui.VectorEdit()
        self._sun_dir.set_on_value_changed(self._on_sun_dir)
        self._sun_color = gui.ColorEdit()
        self._sun_color.set_on_value_changed(self._on_sun_color)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._sun_intensity)
        grid.add_child(gui.Label("Direction"))
        grid.add_child(self._sun_dir)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._sun_color)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label("Sun (Directional light)"))
        advanced.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(advanced)

        material_settings = gui.CollapsableVert(
            "Material settings", 0, gui.Margins(em, 0, 0, 0)
        )

        self._shader = gui.Combobox()
        self._shader.add_item(AppWindow.MATERIAL_NAMES[0])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[1])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[2])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[3])
        self._shader.set_on_selection_changed(self._on_shader)
        self._material_prefab = gui.Combobox()
        for prefab_name in sorted(Settings.PREFAB.keys()):
            self._material_prefab.add_item(prefab_name)
        self._material_prefab.selected_text = Settings.DEFAULT_MATERIAL_NAME
        self._material_prefab.set_on_selection_changed(self._on_material_prefab)
        self._material_color = gui.ColorEdit()
        self._material_color.set_on_value_changed(self._on_material_color)
        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 10)
        self._point_size.set_on_value_changed(self._on_point_size)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Type"))
        grid.add_child(self._shader)
        grid.add_child(gui.Label("Material"))
        grid.add_child(self._material_prefab)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._material_color)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(self._point_size)

        # Slider to adjust remove_large_triangles
        grid.add_child(gui.Label("Remove low density vertices"))
        self.clean_mesh_slider = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.clean_mesh_slider.set_limits(0.0, 0.9)
        self.clean_mesh_slider.set_on_value_changed(self._on_clean_mesh_slider_changed)
        grid.add_child(self.clean_mesh_slider)

        material_settings.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(material_settings)
        # ----

        # Normally our user interface can be children of all one layout (usually
        # a vertical layout), which is then the only child of the window. In our
        # case we want the scene to take up all the space and the settings panel
        # to go above it. We can do this custom layout by providing an on_layout
        # callback. The on_layout callback should set the frame
        # (position + size) of every child correctly. After the callback is
        # done the window will layout the grandchildren.
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)

        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("About", AppWindow.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit", AppWindow.MENU_QUIT)
            file_menu = gui.Menu()
            file_menu.add_item(
                "Load Point Cloud...", AppWindow.MENU_LOAD_POINTS
            )
            file_menu.add_item(
                "Export 3D object (.ply)...", AppWindow.MENU_EXPORT_POINTS
            )
            file_menu.add_item(
                "Export 3D point cloud (no surface information) (.ply)...", AppWindow.MENU_EXPORT_POINTS_NOMESH
            )
            file_menu.add_item(
                "Export point cloud as csv (x,y,z,nx,ny,nz)...", AppWindow.MENU_EXPORT_POINTS_CSV
            )
            file_menu.add_item("Export Current Scene as PNG...", AppWindow.MENU_EXPORT)
            
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Open existing 3D Model (visualization only)...", AppWindow.MENU_OPEN)
                file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            else:
                file_menu.add_separator()
                file_menu.add_item("Open existing 3D Model (visualization only)...", AppWindow.MENU_OPEN)
            settings_menu = gui.Menu()
            settings_menu.add_item("Lighting & Materials", AppWindow.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)
            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)

            PointNormalNet_menu = gui.Menu()
            PointNormalNet_menu.add_item("Load Model", AppWindow.MENU_LOAD_MODEL)
            PointNormalNet_menu.add_separator()
            PointNormalNet_menu.add_item(
                "Configure PointNormalNet...", AppWindow.MENU_CONFIGURE
            )


            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("PointNormalNet", PointNormalNet_menu)
                menu.add_menu("View", settings_menu)
                menu.add_menu("Help", help_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("PointNormalNet", PointNormalNet_menu)
                menu.add_menu("View", settings_menu)
                menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        w.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT, self._on_menu_export)
        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(
            AppWindow.MENU_SHOW_SETTINGS, self._on_menu_toggle_settings_panel
        )

        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        w.set_on_menu_item_activated(AppWindow.MENU_LOAD_MODEL, self._on_load_model)
        w.set_on_menu_item_activated(AppWindow.MENU_LOAD_POINTS, self._on_load_points)
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT_POINTS, self._on_export_points)
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT_POINTS_NOMESH, self._on_export_points_nomesh)
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT_POINTS_CSV, self._on_export_points_csv)
        w.set_on_menu_item_activated(AppWindow.MENU_CONFIGURE, self._on_configure)
        # ----

        # PointNormalNet Stuff
        self.ckpt_dir = DEFAULT_CKPT_DIR
        self.input_file = None
        self.noise_thresh = NOISE_THRESH
        self.max_n = MAX_N
        self.device = DEVICE

        self.model = None
        self.params = None
        self.depth = 8
        self.failure_message = ""

        self.curr_mesh = None
        self.curr_clean_mesh = None
        self.curr_densities = None
        self.curr_pcd = None
        self.curr_df = None
        self.remove_low_quantile = 0.00  # 0 = None

        self.clean_mesh_slider.set_value(self.remove_low_quantile)

        self._apply_settings()

        # Make sure export_points menu item is disabled until points are loaded
        gui.Application.instance.menubar.set_enabled(AppWindow.MENU_EXPORT_POINTS, False)
        gui.Application.instance.menubar.set_enabled(AppWindow.MENU_EXPORT_POINTS_NOMESH, False)
        gui.Application.instance.menubar.set_enabled(AppWindow.MENU_EXPORT_POINTS_CSV, False)

        # Pre-load model
        if os.path.isdir(self.ckpt_dir):
            print("Loading model...")
            ret = self.load_model(self.ckpt_dir, self.device)
            print("Done!")

            # Show a message box stating that the model was loaded
            if ret == 0:
                self._show_message_box("Default model loaded successfully. Go to File > Load Point Cloud... to load a point cloud.")
            else:
                self._show_message_box("Could not load default model. Load a model before loading a point cloud. Error: " + self.failure_message)
        else:
            print(
                "No model found at specified checkpoint directory. Please load a model before performing inference."
            )

            # Show a message box stating that the model was not loaded
            self._show_message_box(
                "No model found at the default checkpoint directory. Please load a model before performing inference."
            )

    ## PointNormalNet Stuff
    def _on_clean_mesh_slider_changed(self, value):
        self.remove_low_quantile = value
        if self.curr_mesh is not None:
            print("Cleaning mesh...")
            mesh_copy = copy.deepcopy(self.curr_mesh)
            clean_3d_mesh(self.curr_mesh, remove_low_quantile=self.remove_low_quantile, densities=self.curr_densities)
            
            # Update the scene
            print("Updating scene...")
            
            # Clear the scene
            self._scene.scene.clear_geometry()
            
            # self._scene.scene.add_model("__model__", mesh)
            self._scene.scene.add_geometry("__model__", self.curr_mesh, self.settings.material)

            # Add the pcd to the scene
            self._scene.scene.add_geometry("pcd", self.curr_pcd, self.settings.material)

            # Do not change camera
            # bounds = self._scene.scene.bounding_box
            # self._scene.setup_camera(60, bounds, bounds.get_center())

            # Save the clean mesh
            self.curr_clean_mesh = self.curr_mesh

            # Replace the original mesh (we need original mesh for reverting back to original)
            self.curr_mesh = mesh_copy

    def _on_configure(self):
        # Create a new window to adjust parameters
        # self.noise_thresh = slider from 0 to 1
        # self.max_n = any integer > 1000, max 1_000_000
        # self.device = dropdown with options "cpu" and "cuda"
        
        noise_thresh_slider = gui.Slider(gui.Slider.DOUBLE)
        noise_thresh_slider.set_limits(0, 1)
        noise_thresh_slider.double_value = self.noise_thresh
        noise_thresh_slider.set_on_value_changed(self._on_noise_thresh_changed)

        max_n_slider = gui.Slider(gui.Slider.INT)
        max_n_slider.set_limits(*MAX_N_LIMITS)
        max_n_slider.int_value = int(self.max_n)
        max_n_slider.set_on_value_changed(self._on_max_n_changed)

        depth_slider = gui.Slider(gui.Slider.INT)
        depth_slider.set_limits(*DEPTH_LIMITS)
        depth_slider.int_value = int(self.depth)
        depth_slider.set_on_value_changed(self._on_depth_changed)

        # device_dropdown = gui.Combobox()
        # device_dropdown.add_item("cpu")
        # device_dropdown.add_item("cuda")
        # device_dropdown.set_on_selection_changed(self._on_device_changed)

        
        # Show a message box
        em = self.window.theme.font_size
        dlg = gui.Dialog('Configure')

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        
        dlg_layout.add_child(gui.Label('Noise threshold (lower == more noise in output)'))
        dlg_layout.add_child(noise_thresh_slider)

        dlg_layout.add_child(gui.Label('Max points per batch (lower for less memory usage) (Caution: Higher values will cause CUDA out of memory errors)'))
        dlg_layout.add_child(max_n_slider)

        dlg_layout.add_child(gui.Label('Depth (lower values are faster, and produce more "rounded" surface, higher values introduce sharper kinks)'))
        dlg_layout.add_child(depth_slider)

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        re_run = gui.Button("Update Reconstruction")
        re_run.set_on_clicked(self._on_re_run)

        ok = gui.Button("Save and Close")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        if self.model is not None and self.input_file is not None:
            h.add_child(re_run)
            h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)
        return
    
    def _on_noise_thresh_changed(self, val):
        self.noise_thresh = val
        return
    
    def _on_max_n_changed(self, val):
        self.max_n = int(val)
        return
    
    def _on_depth_changed(self, val):
        self.depth = int(val)
        return
    
    def _on_re_run(self):
        self.window.close_dialog()
        self._show_message_box("Updating Reconstruction...")
        ret = self.do_inference(self.input_file)
        self.window.close_dialog()
        if ret == 0:
            self._show_message_box("Updated succesfully!")
        else:
            self._show_message_box("Failed to update!")

    def load_model(self, ckpt_dir, device):
        try:
            print(f"Loading model from {ckpt_dir}...")
            self.model, self.params = load_model(ckpt_dir=ckpt_dir, device=device)
            print("Done!")
            return 0
        except Exception as e:
            print(e)
            self.failure_message = str(e)
            return 1

    def do_inference(self, input_file):
        # Load points
        print(f"Loading points from {input_file}...")
        graphs = load_points(input_file, self.max_n)
        if graphs is None:
            print("Failed to load points. Exiting...")
            self.failure_message = (
                "Failed to load points. Make sure the file is in the correct format."
            )
            return 1
        print("Done!")

        # Perform inference
        try:
            print("Performing inference...")
            dfs = []
            for i, g in enumerate(graphs):
                df = infer(
                    self.model,
                    g,
                    self.params["model"]["strategy"],
                    self.noise_thresh,
                    self.device,
                )
                dfs.append(df)
                print(f"{i+1}/{len(graphs)}", end="\r")

            df = pd.concat(dfs)

            # Get rid of nans
            df.dropna(inplace=True)

            # Drop duplicates (if any)
            df = df.drop_duplicates(subset=["x", "y", "z"], keep="first")
            print("Done!")

            # Get the 3D mesh and pcd
            print("Generating mesh...")
            mesh, pcd, densities = get_3d_mesh(df, depth=self.depth, noise_label=NOISE_LABEL, return_densities=True)
            print("Done!")
        except Exception as e:
            print(e)
            self.failure_message = str(e)
            return 1

        # Add the mesh to the scene
        try:
            # Clear the scene
            self._scene.scene.clear_geometry()
            
            # self._scene.scene.add_model("__model__", mesh)
            self._scene.scene.add_geometry("__model__", mesh, self.settings.material)

            # Add the pcd to the scene
            self._scene.scene.add_geometry("pcd", pcd, self.settings.material)

            bounds = self._scene.scene.bounding_box
            self._scene.setup_camera(60, bounds, bounds.get_center())

            # Save mesh and the df
            self.curr_mesh = mesh
            self.curr_clean_mesh = mesh
            self.curr_df = df
            self.curr_densities = densities
            self.curr_pcd = pcd
        except Exception as e:
            print(e)
            self.failure_message = "Mesh visualization failed."
            return 1

        # Add pcd to the scene, with transparency
        # pcd.material = material.Material(
        #    metallic=0.0,
        #   roughness=0.5,
        #   base_color=[0.0, 0.0, 1.0, 0.5],
        #  alpha_mode=material.AlphaMode.BLEND,
        # double_sided=True,
        # blend=material.BlendMode.TRANSLUCENT,
        # )
        # self._scene.scene.add_geometry(pcd)

        # Enable export_points menu item
        return 0

    def _on_load_model(self):
        dlg = gui.FileDialog(
            gui.FileDialog.OPEN, "Choose file to load", self.window.theme
        )
        dlg.add_filter(".pth", "PyTorch model files (.pth)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_model_dialog_done)
        self.window.show_dialog(dlg)

    def _on_load_model_dialog_done(self, filename):
        self.window.close_dialog()
        self.ckpt_dir = os.path.dirname(filename)
        ret = self.load_model(self.ckpt_dir, self.device)

        # Show a message box
        em = self.window.theme.font_size
        dlg = gui.Dialog("Model")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(
            gui.Label(
                "Model loaded successfully!"
                if ret == 0
                else "Failed to load model! Error Details:\n" + self.failure_message
            )
        )
        self.failure_message = ""

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_load_points(self):
        if self.model is None:
            # Show a message box
            em = self.window.theme.font_size
            dlg = gui.Dialog("Model")

            # Add the text
            dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
            dlg_layout.add_child(
                gui.Label(
                    "No model loaded. Please load a model first! "
                    "(PointNormalNet -> Load Model)\n"
                )
            )
            self.failure_message = ""

            # Add the Ok button. We need to define a callback function to handle
            # the click.
            ok = gui.Button("OK")
            ok.set_on_clicked(self._on_about_ok)

            # We want the Ok button to be an the right side, so we need to add
            # a stretch item to the layout, otherwise the button will be the size
            # of the entire row. A stretch item takes up as much space as it can,
            # which forces the button to be its minimum size.
            h = gui.Horiz()
            h.add_stretch()
            h.add_child(ok)
            h.add_stretch()
            dlg_layout.add_child(h)

            dlg.add_child(dlg_layout)
            self.window.show_dialog(dlg)
            return

        dlg = gui.FileDialog(
            gui.FileDialog.OPEN, "Choose file to load", self.window.theme
        )
        dlg.add_filter(
            ".csv .tsv .parquet .xyz",
            "Supported files (.csv .tsv .parquet .xyz)",
        )
        dlg.add_filter(".csv", "Comma-separated values (.csv)")
        dlg.add_filter(".tsv", "Tab-separated values (.tsv)")
        dlg.add_filter(".parquet", "Apache Parquet (.parquet)")
        dlg.add_filter(".xyz", "XYZ (space-separated values) (.xyz)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_points_dialog_done)
        self.window.show_dialog(dlg)

    def _on_load_points_dialog_done(self, filename):
        self.window.close_dialog()
        self.input_file = filename

        # Run Inference
        ret = self.do_inference(self.input_file)

        # Enable export_points menu item if inference is successful
        if ret == 0:
            gui.Application.instance.menubar.set_enabled(AppWindow.MENU_EXPORT_POINTS, True)
            gui.Application.instance.menubar.set_enabled(AppWindow.MENU_EXPORT_POINTS_NOMESH, True)
            gui.Application.instance.menubar.set_enabled(AppWindow.MENU_EXPORT_POINTS_CSV, True)

        # Show a message box
        em = self.window.theme.font_size
        dlg = gui.Dialog("Infer")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(
            gui.Label(
                "Inference Done Successfully!"
                if ret == 0
                else "Inference failed! Error Details:\n" + self.failure_message
            )
        )
        self.failure_message = ""

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_export_points(self):
        dlg = gui.FileDialog(
            gui.FileDialog.SAVE, "Choose where to save", self.window.theme
        )
        dlg.add_filter(".ply", "ASCII PLY files (.ply)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_points_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_points_dialog_done(self, filename):
        self.window.close_dialog()
        # current_mesh = self.curr_mesh
        current_mesh = self.curr_clean_mesh
        if current_mesh is None:
            self._show_message_box("No mesh loaded in the scene. Please load a point cloud first!")
            return
        o3d.io.write_triangle_mesh(filename, current_mesh, write_vertex_normals=True, write_ascii=True)

    def _on_export_points_nomesh(self):
        dlg = gui.FileDialog(
            gui.FileDialog.SAVE, "Choose where to save", self.window.theme
        )
        dlg.add_filter(".ply", "ASCII PLY files (.ply)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_points_nomesh_dialog_done)
        self.window.show_dialog(dlg)
    
    def _on_export_points_nomesh_dialog_done(self, filename):
        self.window.close_dialog()
        current_df = self.curr_df
        if current_df is None:
            self._show_message_box("No mesh loaded in the scene. Please load a point cloud first!")
            return
        
        write_ply(
            current_df,
            filename,
            noise_label=NOISE_LABEL,
        )

    def _on_export_points_csv(self):
        dlg = gui.FileDialog(
            gui.FileDialog.SAVE, "Choose where to save", self.window.theme
        )
        dlg.add_filter(".csv", "Comma-separated values (.csv)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_points_csv_dialog_done)
        self.window.show_dialog(dlg)
    
    def _on_export_points_csv_dialog_done(self, filename):
        self.window.close_dialog()
        current_df = self.curr_df
        if current_df is None:
            self._show_message_box("No mesh loaded in the scene. Please load a point cloud first!")
            return
        
        save_csv(current_df, filename)

    def _show_message_box(self, message, title="Message", add_ok_button=True):
        # Show a message box
        em = self.window.theme.font_size
        dlg = gui.Dialog(title)

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label(message))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        if add_ok_button:
            ok = gui.Button("OK")
            ok.set_on_clicked(self._on_about_ok)

            # We want the Ok button to be an the right side, so we need to add
            # a stretch item to the layout, otherwise the button will be the size
            # of the entire row. A stretch item takes up as much space as it can,
            # which forces the button to be its minimum size.
            h = gui.Horiz()
            h.add_stretch()
            h.add_child(ok)
            h.add_stretch()
            dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)
        return

    # Rest of the Open3D GUI code

    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red,
            self.settings.bg_color.green,
            self.settings.bg_color.blue,
            self.settings.bg_color.alpha,
        ]
        self._scene.scene.set_background(bg_color)
        self._scene.scene.show_skybox(self.settings.show_skybox)
        self._scene.scene.show_axes(self.settings.show_axes)
        if self.settings.new_ibl_name is not None:
            self._scene.scene.scene.set_indirect_light(self.settings.new_ibl_name)
            # Clear new_ibl_name, so we don't keep reloading this image every
            # time the settings are applied.
            self.settings.new_ibl_name = None
        self._scene.scene.scene.enable_indirect_light(self.settings.use_ibl)
        self._scene.scene.scene.set_indirect_light_intensity(
            self.settings.ibl_intensity
        )
        sun_color = [
            self.settings.sun_color.red,
            self.settings.sun_color.green,
            self.settings.sun_color.blue,
        ]
        self._scene.scene.scene.set_sun_light(
            self.settings.sun_dir, sun_color, self.settings.sun_intensity
        )
        self._scene.scene.scene.enable_sun_light(self.settings.use_sun)

        if self.settings.apply_material:
            self._scene.scene.update_material(self.settings.material)
            self.settings.apply_material = False

        self._bg_color.color_value = self.settings.bg_color
        self._show_skybox.checked = self.settings.show_skybox
        self._show_axes.checked = self.settings.show_axes
        self._use_ibl.checked = self.settings.use_ibl
        self._use_sun.checked = self.settings.use_sun
        self._ibl_intensity.int_value = self.settings.ibl_intensity
        self._sun_intensity.int_value = self.settings.sun_intensity
        self._sun_dir.vector_value = self.settings.sun_dir
        self._sun_color.color_value = self.settings.sun_color
        self._material_prefab.enabled = self.settings.material.shader == Settings.LIT
        c = gui.Color(
            self.settings.material.base_color[0],
            self.settings.material.base_color[1],
            self.settings.material.base_color[2],
            self.settings.material.base_color[3],
        )
        self._material_color.color_value = c
        self._point_size.double_value = self.settings.material.point_size

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()
            ).height,
        )
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)

    def _set_mouse_mode_rotate(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _set_mouse_mode_fly(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.FLY)

    def _set_mouse_mode_sun(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_SUN)

    def _set_mouse_mode_ibl(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_IBL)

    def _set_mouse_mode_model(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)

    def _on_bg_color(self, new_color):
        self.settings.bg_color = new_color
        self._apply_settings()

    def _on_show_skybox(self, show):
        self.settings.show_skybox = show
        self._apply_settings()

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

    def _on_use_ibl(self, use):
        self.settings.use_ibl = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_use_sun(self, use):
        self.settings.use_sun = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_lighting_profile(self, name, index):
        if name != Settings.CUSTOM_PROFILE_NAME:
            self.settings.apply_lighting_profile(name)
            self._apply_settings()

    def _on_new_ibl(self, name, index):
        self.settings.new_ibl_name = gui.Application.instance.resource_path + "/" + name
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_ibl_intensity(self, intensity):
        self.settings.ibl_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_intensity(self, intensity):
        self.settings.sun_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_dir(self, sun_dir):
        self.settings.sun_dir = sun_dir
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_color(self, color):
        self.settings.sun_color = color
        self._apply_settings()

    def _on_shader(self, name, index):
        self.settings.set_material(AppWindow.MATERIAL_SHADERS[index])
        self._apply_settings()

    def _on_material_prefab(self, name, index):
        self.settings.apply_material_prefab(name)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_material_color(self, color):
        self.settings.material.base_color = [
            color.red,
            color.green,
            color.blue,
            color.alpha,
        ]
        self.settings.apply_material = True
        self._apply_settings()

    def _on_point_size(self, size):
        self.settings.material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_menu_open(self):
        dlg = gui.FileDialog(
            gui.FileDialog.OPEN, "Choose file to load", self.window.theme
        )
        dlg.add_filter(
            ".ply .stl .fbx .obj .off .gltf .glb",
            "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, " ".gltf, .glb)",
        )
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, " ".pcd, .pts)",
        )
        dlg.add_filter(".ply", "Polygon files (.ply)")
        dlg.add_filter(".stl", "Stereolithography files (.stl)")
        dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
        dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        dlg.add_filter(".off", "Object file format (.off)")
        dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
        dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
        dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
        dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
        dlg.add_filter(".xyzrgb", "ASCII point cloud files with colors (.xyzrgb)")
        dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        dlg.add_filter(".pts", "3D Points files (.pts)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load(filename)

    def _on_menu_export(self):
        dlg = gui.FileDialog(
            gui.FileDialog.SAVE, "Choose file to save", self.window.theme
        )
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        frame = self._scene.frame
        self.export_image(filename, frame.width, frame.height)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible
        )

    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))

        # info
        info = "".join([
        f"Currently loaded model: {self.ckpt_dir}/model.pth\n" if self.ckpt_dir is not None else "",
        f"Currently loaded point cloud: {self.input_file}\n"  if self.input_file is not None else ""
        ])

        dlg_layout.add_child(
            gui.Label(
                "PointNormalNet Demo\n"
                "By: Suyog S. Jadhav, May 2023\n"
                "************************************\n\n"
                "How to use:\n"
                "1. Load one of the pretrained models by clicking on PointNormalNet->Load Model."
                "You can find the pretrained models under core/static/weights folder.\n"
                "2. Run the model on a point cloud file by clicking on File->Load Point Cloud... . These formats are supported:"
                ".csv, .tsv, and .parquet. The file must have at least 3 columns for x, y, and z coordinates."
                "If additional columns are there, make sure that the coordinates are marked with one of these common column headers:"
                "[x, y, z], [X, Y, Z], ['x [nm]', 'y [nm]', 'z [nm]'], ['X [nm]', 'Y [nm]', 'Z [nm]'\n"
                "You can some sample files under examples/\n"
                "3. The output is shown on the demo screen, overlaid on the input point cloud.\n"
                "4. From the file menu, you can now export the reconstruction, either as a 3D object (has surface data) or as a point cloud (has points and their normals, but not the surface data). "
                "If you wish, you can also export the point cloud with normals, as a CSV file. "
                "You can also export just the current view as a PNG image.\n\n"
                "Additionally, you can change the settings of the model by clicking on PointNormalNet->Configure PointNormalNet...\n"
                "You can also adjust visualization parameters by using the on-screen controls.\n\n"
                "************************************\n\n"
                "Credits: This demo GUI is modified from Open3D's demo GUI example. Original source:"
                "https://github.com/isl-org/Open3D/blob/master/examples/python/visualization/vis_gui.py\n"
                "************************************\n\n"
                f"{info}" + \
                "************************************\n\n" if info else ""
            )
        )

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()

    def load(self, path):
        self._scene.scene.clear_geometry()

        geometry = None
        geometry_type = o3d.io.read_file_geometry_type(path)

        mesh = None
        if geometry_type & o3d.io.CONTAINS_TRIANGLES:
            mesh = o3d.io.read_triangle_model(path)
        if mesh is None:
            print("[Info]", path, "appears to be a point cloud")
            cloud = None
            try:
                cloud = o3d.io.read_point_cloud(path)
            except Exception:
                pass
            if cloud is not None:
                print("[Info] Successfully read", path)
                if not cloud.has_normals():
                    cloud.estimate_normals()
                cloud.normalize_normals()
                geometry = cloud
            else:
                print("[WARNING] Failed to read points", path)

        if geometry is not None or mesh is not None:
            try:
                if mesh is not None:
                    # Triangle model
                    self._scene.scene.add_model("__model__", mesh)
                else:
                    # Point cloud
                    self._scene.scene.add_geometry(
                        "__model__", geometry, self.settings.material
                    )
                bounds = self._scene.scene.bounding_box
                self._scene.setup_camera(60, bounds, bounds.get_center())
            except Exception as e:
                print(e)

    def export_image(self, path, width, height):
        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)


def main():
    # We need to initialize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()

    w = AppWindow(1024, 768)

    # if len(sys.argv) > 1:
    #     path = sys.argv[1]
    #     if os.path.exists(path):
    #         w.load(path)
    #     else:
    #         w.window.show_message_box("Error", "Could not open file '" + path + "'")

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
