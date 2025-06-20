from qtpy import QtWidgets
import qtpy.QtWebEngineWidgets as webw
from typing import Callable, Tuple, Any, List, Dict
import numpy as np
import plotly.graph_objects as go
import plotly
from src.utils.QThreadUtils import GuiComputationDispatcher
"""
Plot that is able to vizualize the overlay of several 3D surfaces. 
This is so far only used for illustration purposes. 
"""

class Surface3Dplot(QtWidgets.QWidget):
    get_plot_data: List[Callable[[Any], Tuple[np.ndarray, np.ndarray, np.ndarray]]] 
    do_on_start_extern: Callable[[], None] = None 
    do_on_complete_extern: Callable[[], None]=None
    do_on_start: Callable[[], None]
    doRender: Callable[[], str]
    do_on_stop: Callable[[str], None]
    main_layout: QtWidgets.QVBoxLayout
    webengine_view = None
    colorScales = ["plasma","blackbody", "blues", "solar", "fall", "oxy", "spectral", "bugn", "bupu"]
    displayed_data: Dict[int,  Any] = {}
    render_func: Callable[[], str] = None
    displayed_surfaces: int = 0
    gutless: bool = False
    def __init__(self, get_plot_data: List[Callable[[Any], Tuple[np.ndarray, np.ndarray, np.ndarray]]], plot_data: List[Dict[str, Any]], do_on_start: Callable[[], None] = None, do_on_complete: Callable[[], None]=None, width=8, height=4, dispatch_computation: bool = True, gutless: bool = False):
        """
        Initiates the Surface3DPlot.
        Args:
            get_plot_data: Functions that are used to generate the Data for the 3D Surface Plot. Has to return Tuple of arrays (x, y, z_data)::: x: (n_x, ), y: (n_y, ) z:(n_y, n_x)
            Each function is used to generate a Surface
            plot_data: List of parameter that are passed to the get_plot_data_functions to generate the initial plot.
            do_on_start: Additional actions that are executed when the Computation for the plot begins. This should not include the updating 
            of the SurfacePlot but rather update other GUI elements(Disabling of buttons, ...)
            do_on_complete: Additional actions that are executed when the Computation for the plot Terminates. This ...
            ---> Most of these pre/post computations should not be used. 
            This plot is currently only used in the guttless mode. Guts typically live in a seperate process 
            to archieve better performane (see ompc_process_func as an example)
        
        """
        """"""
        self.gutless = gutless
        super(Surface3Dplot, self).__init__()
        self.main_layout = QtWidgets.QVBoxLayout()
        self.webengine_view = webw.QWebEngineView()
        self.main_layout.addWidget(self.webengine_view)
        self.setLayout(self.main_layout)
        if not self.gutless:
            self.get_plot_data = get_plot_data
            self.displayed_surfaces = len(plot_data)
            for i in range(0, len(plot_data)):
                self.displayed_data[i] = plot_data[i]
            def doRender() -> str:
                surfaces = list()
                for i in range(0, len(self.get_plot_data)):
                    if (not self.get_plot_data[i] is None) and  (not self.displayed_data[i] is None):
                        x, y, z = self.get_plot_data[i](**self.displayed_data[i])
                        surf = go.Surface(z=z, x=x, y=y, opacity=0.9, colorscale=self.colorScales[i%len(self.colorScales)])
                        surfaces.append(surf)
                self.fig = go.Figure(data=surfaces)
                self.fig.update_layout(title="PDF Surface", autosize=True)
                h_out = plotly.offline.plot(self.fig, output_type='div', include_plotlyjs='cdn')
                html = '<html><body>'
                html += h_out
                html += '</body></html>'
                return html
            self.render_func = doRender
            def do_on_start():
                if self.do_on_start_extern is None:
                    return
                
            def do_on_stop(arg: str):
                self.webengine_view.setHtml(arg)
                if not self.do_on_complete_extern is None:
                    self.do_on_complete_extern()
            self.do_on_start = do_on_start
            self.render_func = doRender
            self.do_on_stop = do_on_stop
            
            if dispatch_computation:
                GuiComputationDispatcher.dispatch_gui_computation(doRender, self.do_on_stop, self.do_on_start)

    def set_html(self, html: str): 
        self.webengine_view.setHtml(html)
    def update_contents(self, data: Dict[int, Any], do_on_start: Callable[[], None] = None, do_on_complete: Callable[[], None]=None):
        """
        Updates the Contents that are displayed . Dict contains arguments for the individual given Computation Functions. 
        All surface plots will be rerendered, even if only certain plots are updated. This might be to inefficient for larger number of displayed Surfaces.

        
        """
        if self.gutless:
            return
        funcs: int = 0
        for k in data.keys():
            if data[k] is not None:
                funcs += 1
        self.displayed_surfaces = funcs
        for k in data.keys():
            self.displayed_data[k] = data[k]
        do_start = None
        do_stop = None
        if do_on_start is not None:
            def temp():
                do_on_start()
        
            do_start = temp
        else:
            do_start = self.do_on_start

        if do_on_complete is not None:
            def temp(arg: str):
                self.webengine_view.setHtml(arg)
                do_on_complete()
            do_stop = temp
        else:
            do_stop = self.do_on_stop
        GuiComputationDispatcher.dispatch_gui_computation(self.render_func, do_stop, do_start)


class Surface3DplotGuts():
    """
    Guts of the surface 3D plot. This Version does not do multithreading or multiprocessing. 
    It handles all the computation for the Surface3D plot but is single threaded/ single process. 
    It has everything except the actual plot and is able to hold an internal state. 
    This class is intended to be used with the DataflowHandler and multiprocessing. If this is the case, the actual 
    surface plot has to be started in gutless mode.
    -----------

    """
    get_plot_data: List[Callable[[Any], Tuple[np.ndarray, np.ndarray, np.ndarray]]] 
    doRender: Callable[[], str]
    colorScales = ["plasma","blackbody", "blues", "solar", "fall", "oxy", "spectral", "bugn", "bupu"]
    displayed_data: Dict[int,  Any] = {}
    render_func: Callable[[], str] = None
    displayed_surfaces: int = 0
    def __init__(self, get_plot_data: List[Callable[[Any], Tuple[np.ndarray, np.ndarray, np.ndarray]]], plot_data: List[Dict[str, Any]], width=8, height=4):
        """
        Initiates the Surface3DPlot.
        Args:
            get_plot_data: Functions that are used to generate the Data for the 3D Surface Plot. Has to return Tuple of arrays (x, y, z_data)::: x: (n_x, ), y: (n_y, ) z:(n_y, n_x)
            Each tuple is used to generate a Surface
            plot_data: List of parameter that are passed to the get_plot_data_functions to generate the initial plot.
            

        
        """
        self.fig: go.Figure = None
        self.get_plot_data = get_plot_data
        self.displayed_surfaces = len(plot_data)
        for i in range(0, len(plot_data)):
            self.displayed_data[i] = plot_data[i]
        def doRender() -> str:
            surfaces = list()
            for i in range(0, len(self.get_plot_data)):
                if (not self.get_plot_data[i] is None) and  (not self.displayed_data[i] is None):
                    x, y, z = self.get_plot_data[i](**self.displayed_data[i])
                    surf = go.Surface(z=z, x=x, y=y, opacity=0.9, colorscale=self.colorScales[i%len(self.colorScales)])
                    surfaces.append(surf)
            fig = go.Figure(data=surfaces)
            fig.update_layout(title="PDF Surface", autosize=True)
            self.fig = fig
            h_out = plotly.offline.plot(fig, output_type='div', include_plotlyjs='cdn')
            html = '<html><body>'
            html += h_out
            html += '</body></html>'
            return html
        self.render_func = doRender

    def display_figure(self):
        if not self.fig is None:
            self.fig.show()
    def update_contents(self, data: Dict[int, Any]):
        """
        Updates the Contents that are displayed . Dict contains arguments for the individual given Computation Functions. 
        All surface plots will be rerendered, even if only certain plots are updated. This might be to inefficient for larger number of displayed Surfaces.

        
        """
        funcs: int = 0
        for k in data.keys():
            if data[k] is not None:
                funcs += 1
        self.displayed_surfaces = funcs
        for k in data.keys():
            self.displayed_data[k] = data[k]

        html = self.render_func()
        return html
        

        


