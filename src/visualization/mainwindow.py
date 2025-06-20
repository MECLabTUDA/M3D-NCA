import sys
from qtpy import QtWidgets, QtCore # No longer need QtGui specifically for this simple case

from typing import List, Dict, Optional


import os
import sys
import numpy as np
from qtpy import QtWidgets, QtCore # No longer need QtGui specifically for this simple case

from qtpy.QtGui import *
from qtpy.QtCore import *
from qtpy.QtWidgets import *
import PySide6QtAds as QtAds
from torchio.transforms.augmentation.random_transform import RandomTransform
from torchio.transforms import RandomAffine

from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.utils.DataFlowHandler import DataFlowHandler, ResultGetters

from .DatasetVis import DatapointLoaderWidget
from src.visualization.ComboChannelVisualizer import ComboChannelVisualization
from src.visualization.OverlayOutputVis import OverlayedAnimationWidget
from src.visualization.OutputVisualizer import OutputVisualizer
from src.visualization.OutputVisualizerMultiHeatMap import OutputVisualizerMulitHeatMap
from src.visualization.VideoRenderer import Output2VideoDialog, OutputVideoRenderer, Output2VideoWindow
from src.utils.helper import merge_img_label_gt, overlay_sdf_field_nicer

from src.visualization.ConfigEditor import ConfigEditorDialog, ConfigOption, default_config_options_to_dict

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, title: str, dataflow_handler: DataFlowHandler, config_options_editor: List[ConfigOption], default_config: dict, icon_path="", parent=None, ompc_group: str = "", ompc_mapping_process: str = "", ompc_surface_process: str = "", 
                 transforms: Dict[str, RandomTransform] = {}):
        super().__init__(parent)
        self.transforms: Dict[str, RandomTransform] = transforms
        self.dataflow_handler = dataflow_handler

        self.config_options_editor = config_options_editor

        # this will be used for all initializations
        self.default_config = default_config

        QtAds.CDockManager.setConfigFlag(QtAds.CDockManager.OpaqueSplitterResize, True)
        QtAds.CDockManager.setConfigFlag(QtAds.CDockManager.FocusHighlighting, True)

        self.setWindowTitle(title)
        dock_container = QWidget()
        layout = QVBoxLayout(dock_container)
        layout.setContentsMargins(QMargins(0, 0, 0, 0))

        # create menubar
        menu_bar = self.menuBar()
        exit_action = QAction(u"Exit", self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(lambda f: sys.exit(0))

        file_menu = menu_bar.addMenu("File")
        file_menu.addAction(exit_action)

        self.view_menu = menu_bar.addMenu("View")

        # create config editor action
        config_action: QAction = QAction("Config Settings", self)
        config_action.setCheckable(False)
        config_action.setStatusTip("Open ConfigEditor dialog")
        config_action.triggered.connect(self.edit_config)
        file_menu.addAction(config_action)

        # create video renderer action
        video_render_action: QAction = QAction("Video Renderer", self)
        video_render_action.setCheckable(False)
        video_render_action.setStatusTip("Open Video Renderer dialog")
        video_render_action.triggered.connect(self.render_video)
        video_render_action.setIcon(QPixmap(os.path.join(icon_path, "video_icon.png")))
        file_menu.addAction(video_render_action)

        # create toolbar
        self.tool_bar = QToolBar()
        self.addToolBar(self.tool_bar)

        # create central docking widget, which will stay open even if empty
        label = QLabel()
        label.setText("Central Docking Widget. Nothing to show :(")
        label.setAlignment(Qt.AlignCenter)
        central_dock_widget = QtAds.CDockWidget("Central")
        central_dock_widget.setWidget(label)
        central_dock_widget.setFeature(QtAds.CDockWidget.NoTab, True)

        # create docking manager and set as the only top level widget 
        self.dock_manager = QtAds.CDockManager(dock_container)
        self.central_dock_area = self.dock_manager.setCentralWidget(central_dock_widget)

        # create dataset viewer widget, wrap as dockwidget and add to top level dock widget
        
        def merge_func(img, lbl, gt):
            if self.default_config["label_select"] == "seg":
                return merge_img_label_gt(img, lbl, gt)
            else:
                return overlay_sdf_field_nicer(img, gt)

        dataset_widget = DatapointLoaderWidget(dataflow_handler, transforms=transforms, merge_func=merge_func)    
        dataset_dock_widget = QtAds.CDockWidget("Dataset")
        dataset_dock_widget.setWidget(dataset_widget)

        # add toggle view action to toolbar
        dataloader_toggle_act: QAction = dataset_dock_widget.toggleViewAction()
        dataloader_toggle_act.setText("Dataset Loader")
        dataloader_toggle_act.setIcon(QPixmap(os.path.join(icon_path, "dataset_icon.png")))
        dataloader_toggle_act.setCheckable(False)
        self.tool_bar.addAction(dataloader_toggle_act)
        self.tool_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        # add toggle view to view menu as well
        self.view_menu.addAction(dataloader_toggle_act)

        # create animation widget
        ret = {}
        for i in range(21):
            ret[i] = np.random.uniform(low=0, high=1, size=[64, 64, 52, 20])
        
        self.dataflow_handler.current_data = ret
        animation_widget = ComboChannelVisualization(stepsDict=ret, net_config=self.default_config, step_axis=2, start_step=0, 
                                                     axis_step=13, channel=1, dataflowhandler=dataflow_handler, ompc_groupd=ompc_group, 
                                                     ompc_mapping_process=ompc_mapping_process, ompc_surface_process=ompc_surface_process)
        animation_widget.setup_remote_output_getter(dataflow_handler)
        animation_dock_widget = QtAds.CDockWidget("Animation")
        animation_dock_widget.setWidget(animation_widget)

        animation_toggle_act: QAction = animation_dock_widget.toggleViewAction()
        animation_toggle_act.setText("Animation Plot")
        animation_toggle_act.setIcon(QPixmap(os.path.join(icon_path, "plot_icon.png")))
        animation_toggle_act.setCheckable(False)
        self.tool_bar.addAction(animation_toggle_act)
        # add toggle view to view menu as well
        self.view_menu.addAction(animation_toggle_act)

        # create multi heatmap widget
        
        animation_multi_widget = OutputVisualizerMulitHeatMap(stepsDict=ret, net_config=self.default_config, step_axis=2, start_step=20, axis_step=13, channel=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) #, channel=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
        animation_multi_widget.setup_remote_output_getter(dataflow_handler)
        animation_multi_dock_widget = QtAds.CDockWidget("Animation Multi")
        animation_multi_dock_widget.setWidget(animation_multi_widget)
        animation_multi_toggle_act: QAction = animation_multi_dock_widget.toggleViewAction()
        animation_multi_toggle_act.setText("MultiAnimation Plot")
        animation_multi_toggle_act.setIcon(QPixmap(os.path.join(icon_path, "plot_multi_icon.png")))
        animation_multi_toggle_act.setCheckable(False)
        self.tool_bar.addAction(animation_multi_toggle_act)
        self.view_menu.addAction(animation_multi_toggle_act)

        self.tool_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)

        self.dock_manager.addDockWidget(QtAds.LeftDockWidgetArea, dataset_dock_widget, self.central_dock_area)
        self.dock_manager.addDockWidget(QtAds.CenterDockWidgetArea, animation_dock_widget, self.central_dock_area)
        self.dock_manager.addDockWidget(QtAds.RightDockWidgetArea, animation_multi_dock_widget, self.central_dock_area)


        self.tool_bar.addAction(video_render_action)

        layout.addWidget(self.dock_manager)
        self.setCentralWidget(dock_container)

        self.resize(1920, 1280)

    def addWidgetAsDock(self, w: QWidget, name: str):
        
        dock_widget = QtAds.CDockWidget(name)
        dock_widget.setWidget(w)
        self.view_menu.addAction(dock_widget.toggleViewAction())

        self.dock_manager.addDockWidget(QtAds.CenterDockWidgetArea, dock_widget, self.central_dock_area)

    def edit_config(self):
       raise Exception("Not implemented in the current setup")

    def render_video(self):
        video_renderer = Output2VideoWindow(self.dataflow_handler, self)
        video_renderer.show()

