import sys
import matplotlib
matplotlib.use('QtAgg')
import torch
from typing import Dict, Any, Callable
from matplotlib.figure import Figure
from typing import List
import numpy as np
import matplotlib.animation as animation
from src.utils.VisUtils import OutputSliceRetreiver
import os
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib import animation
import matplotlib.pyplot as plt
from math import ceil
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class AnimatedHeatMap_multi(FigureCanvas):
    """
    Class for plotting multiple channels in a grid. Created via the OutputVisualizerMulitHeatMap Widget
    """
    tepsDict: Dict[int, np.ndarray] = None
    # Dict containing individual steps of neural network
    # step nr -> (x, y, z, channel) array
    # with NCA3D this is (64, 64, 52, 16)
    retreiver: OutputSliceRetreiver
    step_axis: int # axis along which can be stepped
    ax_len: int # number of steps along axis
    current_step: int # current step along time line
    axis_step: int # current step along the step axis
    running: bool = True
    pending_step: int = 0 # can be set to do individual steps at next render interval. 
        # is set to 0 & ignored when not currently running 
    executed_axis_step: bool = False
    channel: [int] = None
    config: Dict[str, Any]
    general_change: bool = False
    transformation_func: Callable[[np.ndarray], np.ndarray] = None
    input_range: np.ndarray
    optional_event_loop_augentation: Callable = None
    def __init__(self, stepsDict: Dict[int, np.ndarray], net_config: Dict[str, Any], step_axis: int,  parent=None, width=8, height=4, dpi=100, start_step: int = 0, axis_step: int = 0, channnel: int = None, map_color: str = "plasma",
                 input_range: np.ndarray = np.array([0, 1])):
        """
        Initializes the multi HeatMap Animation
        stepsDict: Dict containing the result from a monitored run
        net_config: Contains the config file of the network
        step_axis: axis along which the animation should step through the data
        start_step: time step at which visualization starts
        axis_step: point along steppable axis at which visualization is done
        """
        self.dict = stepsDict
        self.channel = channnel
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.retreiver = OutputSliceRetreiver(stepsDict, net_config)
        self.config = net_config
        self.step_axis = step_axis

        # max value of the z-axis
        self.ax_len = self.retreiver.get_num_steps_along_axis(self.step_axis)
        self.current_step = start_step
        self.axis_step = axis_step
        self.map_color = map_color
        self.input_range = input_range
        self.fwidth = width
        self.fheight = height
        self.fdpi = dpi
        self.fig = Figure(figsize=(width, height), dpi=dpi)

        # arrange channels in n by n grid if possible
        self.dim = len(self.channel)**(1/2)
        if self.dim < ceil(self.dim) - 0.5:
            self.dim = (ceil(self.dim) - 1, ceil(self.dim))
            fig, self.axes = plt.subplots(self.dim[0], self.dim[1])
        else:
            self.dim = (ceil(self.dim), ceil(self.dim))
            fig, self.axes = plt.subplots(self.dim[0], self.dim[1])

        super(AnimatedHeatMap_multi, self).__init__(fig)
        conv_tensor = torch.FloatTensor(*(1, 64, 64)).uniform_(0, 1)
        # list of all images
        self.image = []
        x = 0
        y = 0

        if self.dim[0] <= 1: #covering 1,2 plot
            for i in self.channel:
                # get slice data
                data = self.retreiver.get_slice(channel=i, step=self.current_step, slice_axis=self.step_axis, slice_index=self.axis_step, normalize_for_output=False)
                t = torch.from_numpy(data)
                t = torch.unsqueeze(t, -1)
                # display data in subplot
                self.axes[x].imshow(t, cmap=self.map_color, vmin=input_range[0], vmax=input_range[1])
                self.axes[x].set_title('Channel: ' + str(i), x=0.5, y=0.975, fontsize = 8)
                # add data to the image list
                self.image.append(self.axes[x].get_images()[0])
                x += 1
        else:
            for i in self.channel:
                # get slice data
                data = self.retreiver.get_slice(channel=i, step=self.current_step, slice_axis=self.step_axis, slice_index=self.axis_step, normalize_for_output=False)
                t = torch.from_numpy(data)
                t = torch.unsqueeze(t, -1)
                # display data in subplot
                self.axes[x][y].imshow(t, cmap=self.map_color, vmin=input_range[0], vmax=input_range[1])
                self.axes[x][y].set_title('Channel: ' + str(i), x=0.5, y=0.975, fontsize = 8)
                # image has to be 1D list
                self.image.append(self.axes[x][y].get_images()[0])
                y += 1
                if y%self.dim[1] == 0:
                    y = 0
                    x += 1

        # removes empty plots at the back 
        if len(self.image) < self.dim[0]*self.dim[1]:
            for i in range(self.dim[0]*self.dim[1]-len(self.image)):
                self.axes[-1][-(i+1)].set_visible(False)

        # remove ticks from images
        for axes in fig.get_axes():
            axes.set_xticks([])
            axes.set_yticks([])

        self.anim = animation.FuncAnimation(
            self.figure,
            self.callback_animation,
            frames=20,
            interval=200,
            blit=True,
        )

    def set_eventloop_augmentation(self, augmentation: Callable):
        self.optional_event_loop_augentation = augmentation
    def set_new_data(self, data: Dict[int, np.ndarray]):
        # update stepsDict
        self.dict = data
        self.retreiver = OutputSliceRetreiver(data, self.config)
        self.time_steps = len(data.keys())
        self.general_change = True
    def get_new_data(self):
        """
        Returns the currently used stepsDict. Mainly used for switching between ABSOLUTE and DIFFERENCE mode
        """
        return self.dict
    def callback_animation(self, i):
        """
        Function Animation of all Channels.
        i: the current timestep
        returns: sequence of new images
        """
        if not self.optional_event_loop_augentation is None:
            self.optional_event_loop_augentation()
        # check if animation is running
        if self.running or self.pending_step != 0 or self.executed_axis_step or self.general_change:
            if self.running:
                self.pending_step = 1
            conv_tensor = torch.FloatTensor(*(1, 64, 64)).uniform_(0, 0.15) 
            im = conv_tensor.permute(1,2,0)
            self.current_step += self.pending_step
            # get new timestpes
            if self.current_step >= self.retreiver.get_time_steps():
                self.current_step = 0
            if self.current_step < 0:
                self.current_step = self.retreiver.get_time_steps() -1

            # update the displayed date in every subplot
            x = 0
            y = 0
            k = 0
            for i in self.channel:
                data = self.retreiver.get_slice(channel=i, step=self.current_step, slice_axis=self.step_axis, slice_index=self.axis_step, normalize_for_output=False)
                t = torch.from_numpy(data)
                t = torch.unsqueeze(t, -1)
                self.image[k].set_data(t)
                k += 1
                y += 1
                if y%self.dim[1] == 0:
                    y = 0
                    x += 1
            self.pending_step = 0
            self.executed_axis_step = False
            self.general_change = False
        # needs to return a sequence (this is why self.image has to be 1D)
        return (*self.image, )
    
    def do_render(self, filepath: str, animation_interval: int = 225):
        """
        Save the current animation as a gif
        filepath: path were the gif will be stored
        animation_interval: determins the interval of the function animation
        """
        animated_data = list()
        # gets all slices data 
        for i in range(0, self.retreiver.get_time_steps()):
            subs = []
            for j in self.channel:
                im_show = self.retreiver.get_slice(channel=j, step=i, slice_axis=self.step_axis, slice_index=self.axis_step, normalize_for_output=False)
                subs.append(im_show)
            animated_data.append(subs)
        
        # creates the figure equivalent to init
        fwidth = self.fwidth
        fheight = self.fheight
        fdpi = self.fdpi
        fig = Figure(figsize=(fwidth, fheight), dpi=fdpi)
        dim = len(self.channel)**(1/2)
        if dim < ceil(dim) - 0.5:
            dim = (ceil(dim) - 1, ceil(dim))
            fig, axes = plt.subplots(dim[0], dim[1])
        else:
            dim = (ceil(dim), ceil(dim))
            fig, axes = plt.subplots(dim[0], dim[1])
        render_im = []
        x = 0
        y = 0
        k = 0

        # assign slice date equivalent to init
        if dim[0] <= 1: #covering 1,2 plot
            for i in self.channel:
                axes[x].imshow(animated_data[0][x], cmap=self.map_color, vmin=self.input_range[0], vmax=self.input_range[1])
                axes[x].set_title('Channel: ' + str(i), x=0.5, y=0.975, fontsize = 8)
                render_im.append(axes[x].get_images()[0])
                x += 1
        else:
            for i in self.channel:
                axes[x][y].imshow(animated_data[0][k], cmap=self.map_color, vmin=self.input_range[0], vmax=self.input_range[1])
                axes[x][y].set_title('Channel: ' + str(i), x=0.5, y=0.975, fontsize = 8)
                # image has to be 1D list
                render_im.append(axes[x][y].get_images()[0])
                # plt.show()
                y += 1
                k += 1
                if y%dim[1] == 0:
                    y = 0
                    x += 1
        
        # removes empty plots at the back 
        if len(render_im) < dim[0]*dim[1]:
            for i in range(dim[0]*dim[1]-len(render_im)):
                axes[-1][-(i+1)].set_visible(False)

        # remove ticks from images
        for axes in fig.get_axes():
            axes.set_xticks([])
            axes.set_yticks([])
        # animation for the render
        def dummy_animate(i: int):
            x = 0
            y = 0
            k = 0
            for c in self.channel:
                render_im[k].set_data(animated_data[i][k])
                k += 1
                y += 1
                if y%dim[1] == 0:
                    y = 0
                    x += 1
            return (*render_im,)
        ani = animation.FuncAnimation(fig, dummy_animate, repeat=True,
                                    frames=len(animated_data), interval=animation_interval)
        # save animation as a gif
        ani.save(filename=filepath)

    def set_transformation_function(self, transform: Callable[[np.ndarray], np.ndarray]):
        self.transformation_func = transform
        self.general_change = True
    
    def switch_running(self):
        self.running = not self.running
        if self.running:
            return "Stop"
        else:
            return "Start"
    
    def set_pending_step(self, step: int):
        self.pending_step = step

    def set_axis_step(self, step: int):
        """
        This method receives either 1 or -1 and sets the current axis step to the appropriate value
        """
        if step == 1:
            self.axis_step += 1
            self.executed_axis_step = True
            if self.axis_step >= self.retreiver.get_num_steps_along_axis(self.step_axis):
                self.axis_step = 0
            
        elif step == -1:
            self.axis_step -= 1
            self.executed_axis_step = True
            if self.axis_step < 0:
                self.axis_step = self.retreiver.get_num_steps_along_axis(self.step_axis) - 1
        
        return self.axis_step
