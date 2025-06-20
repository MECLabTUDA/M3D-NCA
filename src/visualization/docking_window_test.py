import os
import sys

from qtpy.QtGui import *
from qtpy.QtCore import *
from qtpy.QtWidgets import *

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

import PySide6QtAds as QtAds

from .DatasetVis import DatapointLoaderWidget

plot_functions = { x.__name__ : x for x in [np.tanh, np.sinh, np.cosh, np.sin, np.cos, np.tan, np.exp, np.exp2, np.log]}

class SinglePlot():

    def __init__(self, axes, x, func, style):
        self.plot, = axes.plot(x, func(x), style)
        self.x = x
        self.style = style
        self.axes = axes

    def update_f(self, func):
        print("[PLOT]: update to " + func.__name__)
        ys = func(self.x)
        self.plot.set_ydata(ys)
        max = np.max(ys)
        min = np.min(ys)
        self.axes.set_ylim(1.1*min, 1.1*max)


class SimplePlotCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=8, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(1, 1, 1)
        super(FigureCanvasQTAgg, self).__init__(fig)

class MainWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)

        QtAds.CDockManager.setConfigFlag(QtAds.CDockManager.OpaqueSplitterResize, True)
        QtAds.CDockManager.setConfigFlag(QtAds.CDockManager.FocusHighlighting, True)

        self.setWindowTitle("docking test")
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

        view_menu = menu_bar.addMenu("View")

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
        layout.addWidget(self.dock_manager)

        # create dataset viewer widget, wrap as dockwidget and add to top level dock widget
        dataset_widget = DatapointLoaderWidget()    
        dataset_dock_widget = QtAds.CDockWidget("Dataset")
        dataset_dock_widget.setWidget(dataset_widget)

        act : QAction = dataset_dock_widget.toggleViewAction()
        act.setText("Dataset Loader")
        act.setIcon(QPixmap("icons/dataset.png"))
        act.setCheckable(False)
        self.tool_bar.addAction(act)
        self.tool_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)

        # create plot container widget as its own docking widget
        plot_container = QWidget()
        layout = QVBoxLayout(plot_container)
        layout.setContentsMargins(QMargins(0, 0, 0, 0))
        self.plot_dock_manager = QtAds.CDockManager(plot_container)
        layout.addWidget(self.plot_dock_manager)

        self.plot_dock_widget = QtAds.CDockWidget("Plots")
        self.plot_dock_widget.setWidget(plot_container)
        view_menu.addAction(self.plot_dock_widget.toggleViewAction())

        
        create_plot_action = QAction(QPixmap("icons/plot.png"), "Create Plot", self)
        create_plot_action.triggered.connect(self.add_plot)
        self.tool_bar.addAction(create_plot_action)

        # toggle-able filetree widget 
        filetree_widget = QWidget()
        file_tree_layout = QVBoxLayout(filetree_widget)
        treeview = QTreeView()
        file_tree_layout.addWidget(treeview)
        model = QFileSystemModel()
        model.setRootPath(QDir.rootPath())
        model.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs)
        treeview.setModel(model)
        treeview.hideColumn(1)
        treeview.hideColumn(2)
        treeview.hideColumn(3)
        filetree_dock = QtAds.CDockWidget("Files")
        filetree_dock.setWidget(filetree_widget)
        view_menu.addAction(filetree_dock.toggleViewAction())

        self.central_dock_area = self.dock_manager.setCentralWidget(central_dock_widget)

        # dataset viewer, when opened, is part of central widget
        self.dock_manager.addDockWidget(QtAds.CenterDockWidgetArea, dataset_dock_widget, self.central_dock_area)
        self.dock_manager.addDockWidget(QtAds.CenterDockWidgetArea, self.plot_dock_widget, self.central_dock_area)

        # filetree is not part of center widget
        self.dock_manager.addDockWidget(QtAds.LeftDockWidgetArea, filetree_dock)

        self.setCentralWidget(dock_container)

    def add_plot(self):
        print("Add plot")

        function_combobox = QComboBox()
        function_combobox.addItems(plot_functions.keys())

        plot_widget = SimplePlotCanvas()
        x = np.linspace(0, 10, 501)
        new_plot = SinglePlot(plot_widget.axes, x, plot_functions[function_combobox.currentText()], ".")

        plot_dock_widget = QtAds.CDockWidget("plt")
        plot_dock_widget.setWidget(plot_widget)

        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.addWidget(function_combobox)

        def f_upd(s):
            new_plot.update_f(plot_functions[s])
            plot_widget.draw()

        function_combobox.currentTextChanged.connect(f_upd)

        settings_dock_widget = QtAds.CDockWidget("settings")
        settings_dock_widget.setWidget(settings_widget)

        self.plot_dock_manager.addDockWidget(QtAds.CenterDockWidgetArea, plot_dock_widget)
        self.plot_dock_manager.addDockWidget(QtAds.BottomDockWidgetArea, settings_dock_widget)

