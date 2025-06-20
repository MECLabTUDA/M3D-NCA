import matplotlib 
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.collections import PathCollection
import numpy as np
from matplotlib.lines import Line2D
import os
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib import animation
from src.utils.MathHelper import MathHelper
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"




class SegmentedPseudo1DScatterPlot(FigureCanvas):
    """
    This class represents a segmented 1D scatter plot. 
    It takes 1D data as well as segment points and plots 
    the 1D Data with semi randomized y coordinates into a diagram: 
    Y coordinates are randomized to vizualize data density better.
    """
    x_data: np.ndarray # for scatter plot
    y_data: np.ndarray # for scatter plot
    segemnts: np.ndarray
    horizontal_delimiters: np.ndarray
    vlines: None
    pathcol: PathCollection
    line: Line2D = None
    pdf_line: Line2D = None
    fig = None
    def __init__(self, data:np.ndarray, segments_num: int = 0, parent=None, width=8, height=4, dpi=100):
        """
        Initializes the object

        Args:
            data (np.ndarray): ndarray containing the points that are to be scattered
            segments_num (int): Contains the Number of segments into whicht the data should be devided
            parent (_type_, optional): _description_. Defaults to None.
            width (int, optional): _description_. Defaults to 8.
            height (int, optional): _description_. Defaults to 4.
            dpi (int, optional): _description_. Defaults to 100.
        """
        fig = plt.figure(figsize=(width, height), dpi=dpi)

        super(SegmentedPseudo1DScatterPlot, self).__init__(fig)
        self.axes = self.figure.add_subplot(111)
        self.x_data = data
        self.vlines = list()
        norm_distrib = np.random.normal(size=data.shape)
        self.y_data = (norm_distrib-np.min(norm_distrib))/(np.max(norm_distrib)-np.min(norm_distrib))
        self.pathcol =  self.axes.scatter(self.x_data, self.y_data, s=0.05)
        if segments_num > 1:
            self.horizontal_delimiters = self._get_default_intervals(segments_num)
            for delim in self.horizontal_delimiters:
                self.vlines.append(self.axes.axvline(x=delim, color="r"))
        x_data, y_data = MathHelper.get_pdf_for_data(data)
        self.draw_pdf(x_data, y_data)
    
    def redraw_scatter_points(self, points_x: np.ndarray):
        """Redraws the scatter plot with a new set of x coordinates.
        Y coordinates are chosen at random through a normal distribution.
        Vlines and interpolation function are not redrawn.

        Args:
            points_x (np.ndarray): 1D array containing x Coordinates
        """
        self.x_data = points_x
        self.y_data = np.random.normal(size=self.x_data.shape)
        data_array = np.zeros((self.x_data.shape[0], 2))
        data_array[:, 0] = self.x_data
        data_array[:, 1] = self.y_data
        self.pathcol.set_offsets(np.c_[self.x_data,self.y_data])
        x_min = np.min(points_x)
        x_max = np.max(points_x)
        self.axes.set_xlim(left=x_min, right=x_max)
        self.figure.canvas.draw_idle()

    def draw_pdf(self, x_data: np.ndarray, y_data: np.ndarray):
        """Intended to draw an PDF for the currently displayed Data. 
        
        args:
            x_data (np.ndarray): X-Data for PDF
            y_data: (np.ndarray): Y-Data for PDF
        """
        if self.pdf_line is None:
            self.pdf_line= self.axes.plot(x_data, y_data, color="green")
        else:
            self.pdf_line[0].set_xdata(x_data)
            self.pdf_line[0].set_ydata(y_data)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def kill_pdf(self, x_data: np.ndarray=None, y_data: np.ndarray=None):
        """Method that removes the currently displayed PDF function from the canvas. 
        Does nothing if no PDF is currently displayed.
        Scattered points and segments are not removed.
        
        """
        if self.pdf_line is not None:
            self.pdf_line.pop(0).remove()
            self.pdf_line = None
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            
        


    def draw_interpolation_function(self, x_data: np.ndarray, y_data: np.ndarray):
        """If there already exists a drawn interpolation function, the function object 
        is updated with new data.

        Args:
            x_data (np.ndarray): X-Data for points in line
            y_data (np.ndarray): Y-Data
        """
        if self.line is None:
            self.line = self.axes.plot(x_data, y_data, color="red")
        else:
            self.line[0].set_xdata(x_data)
            self.line[0].set_ydata(y_data)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        
    def kill_interpolation_function(self):
        """
        Removes the drawn interpolation function. Leaves everything else untouched. 
        """
        if self.line is not None:
            self.line.pop(0).remove()
            self.line = None
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            
        
    def _get_default_intervals(self, interval_nr: int):
        """
        Internal function, draws horizontal lines equally spaced over the displayed range of the x axis
        """
        if interval_nr == 0:
            return np.zeros(0)
        x_limits = self.axes.get_xlim()
        x_range = x_limits[1] - x_limits[0]
        width = x_range/interval_nr
        offset = x_limits[0] + width
        delimiters = np.zeros(interval_nr-1)
        for i in range(0, interval_nr -1):
            delimiters[i] = offset
            offset += width
        return delimiters
    
    def update_vlines(self, vline_x_pos: np.ndarray):
        """Sets new values for vertical delimiter lines. 
        Does not redraw the vlines on the plot.

        Args:
            vline_x_pos (np.ndarray): 1D Array containing X Positions for vertical lines.
        """
        self.horizontal_delimiters = vline_x_pos
        
            
    def _redrar_vlines(self):
        """
        Does an efficient redraw of only the vertical lines. 
        Uses the internal array of horizontal delimiters for redraw
        """
        num_lines = len(self.vlines)
        for i in range(0, self.horizontal_delimiters.size):
            if i >= num_lines:
                self.vlines.append(self.axes.axvline(x=self.horizontal_delimiters[i], color="r"))
            else:
                self.vlines[i].set_data([self.horizontal_delimiters[i], self.horizontal_delimiters[i]], [0, 1])
                
        diff = len(self.vlines) - self.horizontal_delimiters.size
        if diff > 0:
            for i in range(0, diff):
                vl = self.vlines.pop()
                vl.remove()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        
    def _redraw_complete(self):
        """
        Does a complete redraw of the diagram. 
        Deletes all internal vline objects
        """
        self.axes.cla()
        self.axes.scatter(self.x_data, self.y_data, s=0.05)
        self.vlines = list()
        if not self.horizontal_delimiters is None:
            for delim in self.horizontal_delimiters:
                self.vlines.append(self.axes.axvline(x=delim, color="r"))

        
