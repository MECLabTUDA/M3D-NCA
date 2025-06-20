import sys
import matplotlib
matplotlib.use('QtAgg')
import torch
from typing import Dict, Any, Callable, List
from matplotlib.figure import Figure
import numpy as np
import matplotlib.animation as animation
from src.utils.VisUtils import OutputSliceRetreiver
from qtpy import QtWidgets
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib import animation, gridspec
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


"""
This class represents the main method used to vizualize channel activations.

"""
class AnimatedHeatMap(FigureCanvas):
    color_map: List[str] = ["gist_earth", "terrain", "gnuplot", "brg", "rainbow", "PiYG", "cool", "plasma", "inferno", "magma", "cividis"]
    current_color_map = "plasma"
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
    channel: int = None
    config: Dict[str, Any]
    general_change: bool = False
    transformation_func: Callable[[np.ndarray], np.ndarray] | List[Callable[[np.ndarray], np.ndarray]] = None
    time_steps: int
    input_range: np.ndarray
    start_stop_label: QtWidgets.QLabel = None
    complete_redraw: bool = False
    optional_event_loop_augentation: Callable = None
    cached_data: Dict[int, Any] = {}
    apply_transformation_func: bool = True
    im_show = None
    def __init__(self, stepsDict: Dict[int, np.ndarray], net_config: Dict[str, Any], step_axis: int,  parent=None, width=8, height=4, dpi=100, start_step: int = 0, axis_step: int = 0, channnel: int = None, 
                 input_range: np.ndarray = np.array([0.0, 1.0])):
        """
        Initializes the Animated HeatMap
        stepsDict: Dict containing the result from a monitored run
        net_config: Contains the config file of the network
        step_axis: axis along which the animation should step through the data
        start_step: time step at which visualization starts
        axis_step: point along steppable axis at which visualization is done"""
        self.channel = channnel
        self.time_steps = len(stepsDict.keys())
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.retreiver = OutputSliceRetreiver(stepsDict, net_config)
        self.config = net_config
        self.step_axis = step_axis
        self.ax_len = self.retreiver.get_num_steps_along_axis(self.step_axis)
        self.current_step = start_step
        self.axis_step = axis_step
        self.input_range = input_range
        fig = Figure(figsize=(width, height), dpi=dpi)
        super(AnimatedHeatMap, self).__init__(fig)
        conv_tensor = torch.FloatTensor(*(1, 64, 64)).uniform_(0, 1) 
        self.axes = self.figure.add_subplot(1, 11, (1, 10))
        self.cbar_ax = self.figure.add_subplot(1, 11, 11)
        if self.channel is None or self.channel in self.retreiver.get_output_channels():
            data = self.retreiver.get_output_slice(self.current_step, self.step_axis, self.axis_step, True)
        else:
            data = self.retreiver.get_slice(channel=self.channel, step=self.current_step, slice_axis=self.step_axis, slice_index=self.axis_step, normalize_for_output=False)
        t = torch.from_numpy(data)
        t = torch.unsqueeze(t, -1)
        self.im_show = t
        self.image = self.axes.imshow(t, cmap=self.current_color_map, vmin=0, vmax=1)
        self._cbar = self.figure.colorbar(mappable=self.image, cax=self.cbar_ax)
        self.anim = animation.FuncAnimation(
            self.figure,
            self.callback_animation,
            frames=20,
            interval=200,
            blit=True,
        )
        
    def render_image_series(self, folderpath: str, interval: int):
        """Renders out frames at given interval into the selected folder.
           includes first and last step in all cases
        Args:
            folderpath (str): Path to folder into which should be rendered
            interval (int): interval into which should be rendered
        """
        render_steps: List[int] = [0]
        index: int = interval
        while index < self.retreiver.get_time_steps():
            render_steps.append(index)
            index += interval
        if not self.retreiver.get_time_steps()-1 in render_steps:
            render_steps.append(self.retreiver.get_time_steps()-1)
        for r_step in render_steps:
            render_data = self._get_image_at_step(r_step, self.apply_transformation_func)
            fig = Figure()
            ax = fig.add_subplot(1, 11, (1, 10))
            crbr = fig.add_subplot(1, 11, 11)
            render_im = ax.imshow(render_data, cmap=self.current_color_map, vmin=0, vmax=1)
            fig.colorbar(render_im, cax=crbr)
            file_path: str = os.path.join(folderpath, 'channel_' + str(self.channel) + "_step_" + str(r_step) + '.PNG')
            fig.savefig(file_path)

    def render_current_img(self, filepath: str): 
        """
        Renders only currently shown timestep.
        """
        img = self._get_current_image(self.apply_transformation_func)
        fig = Figure()
        ax = fig.add_subplot(1, 11, (1, 10))
        crbr = fig.add_subplot(1, 11, 11)
        render_im = ax.imshow(img, cmap=self.current_color_map, vmin=0, vmax=1)
        fig.colorbar(render_im, cax=crbr)
        fig.savefig(filepath)


    def do_render(self, filepath: str, animation_interval: int = 225):
        """
        Renders all timesteps as gif.
        Generates it's own animation object. 
        This should not cause problems, if no local references to the animation are kept. Otherwise -> high memory usage.
        """
        animated_data = list()
        apply_transform_func = self.apply_transformation_func
        aaa = self.retreiver.get_time_steps()
        for i in range(0, self.retreiver.get_time_steps()):
            if self.channel is None:
                im_show = self.retreiver.get_output_slice(i, self.step_axis, self.axis_step, True)
            elif self.channel in self.retreiver.get_output_channels():
                im_show = self.retreiver.get_slice(self.channel, i, self.step_axis, self.axis_step, True)
            else:
                im_show = self.retreiver.get_slice(self.channel, i, self.step_axis, self.axis_step, False)
            if (not self.transformation_func is None) and apply_transform_func:
                
                im_show = self.transformation_func(im_show, i % self.time_steps)
                
            else:
                im_show[im_show < self.input_range[0]] = self.input_range[0]
                im_show[im_show > self.input_range[1]] = self.input_range[1]
                im_show = im_show - self.input_range[0]
                im_show = im_show/(self.input_range[1] - self.input_range[0])
            animated_data.append(im_show)

        fig = Figure()
        ax = fig.add_subplot(1, 11, (1, 10))
        crbr = fig.add_subplot(1, 11, 11)
        render_im = ax.imshow(animated_data[i], cmap=self.current_color_map, vmin=0, vmax=1)
        fig.colorbar(render_im, cax=crbr)
        def dummy_animate(i: int):
            render_im.set_data(animated_data[i])
            return (render_im,)
        
        ani = animation.FuncAnimation(fig, dummy_animate, repeat=True,
                                    frames=len(animated_data), interval=animation_interval)
        ani.save(filename=filepath)
        
    def set_eventloop_augmentation(self, augmentation: Callable):
        """
        Deprecated
        """
        self.optional_event_loop_augentation = augmentation
    def set_new_data(self, data: Dict[int, np.ndarray]):
        
        self.retreiver = OutputSliceRetreiver(data, self.config)
        self.time_steps = len(data.keys())
        self.general_change = True
        self.cached_data = {}
    def callback_animation(self, i):
        """
        This is called by the main animation. 
        caches image data for later use. While this increases performance, 
        it also increses memory usage. 
        """
        if not self.optional_event_loop_augentation is None:
            self.optional_event_loop_augentation()
        if self.running or self.pending_step != 0 or self.executed_axis_step or self.general_change:
            if self.running:
                self.pending_step = 1
            self.current_step += self.pending_step
            if self.current_step >= self.retreiver.get_time_steps():
                self.current_step = 0
            if self.current_step < 0:
                self.current_step = self.retreiver.get_time_steps() -1
            t = self._get_current_image(self.apply_transformation_func)
            if self.current_step not in self.cached_data:
                self.cached_data[self.current_step] = t
            self.image.set_data(t)
            self.pending_step = 0
            self.executed_axis_step = False
            self.general_change = False
            if self.start_stop_label is not None:
                if self.running:
                    self.start_stop_label.setText("Stop " + str(self.current_step))
                else:
                    self.start_stop_label.setText("Start " + str(self.current_step))
        return (self.image,)
    
    def _get_current_image(self, apply_transformation_func: bool = True):
        """
        Returns the current image. Handles applying the transformaiton function and retreiving from cache.
        """
        if self.current_step in self.cached_data:
                im_show = self.cached_data[self.current_step]
        else:
            if self.channel is None:
                im_show = self.retreiver.get_output_slice(self.current_step, self.step_axis, self.axis_step, True)
            elif self.channel in self.retreiver.get_output_channels():
                im_show = self.retreiver.get_slice(self.channel, self.current_step, self.step_axis, self.axis_step, True)
            else:
                im_show = self.retreiver.get_slice(self.channel, self.current_step, self.step_axis, self.axis_step, False)
            if (not self.transformation_func is None)  and apply_transformation_func:
                
                im_show = self.transformation_func(im_show, self.current_step % self.time_steps)
                
            else:
                im_show[im_show < self.input_range[0]] = self.input_range[0]
                im_show[im_show > self.input_range[1]] = self.input_range[1]
                im_show = im_show - self.input_range[0]
                im_show = im_show/(self.input_range[1] - self.input_range[0])
        return im_show

    def _get_image_at_step(self, step: int, apply_transformation_func: bool = True):
            """
            Returns the image at given step. Handles applying the transformaiton function and retreiving from cache.
            """
            if step >= self.retreiver.get_time_steps():
                step = step % self.retreiver.get_time_steps()
                
            if step in self.cached_data:
                im_show = self.cached_data[step]
            else:
                if self.channel is None:
                    im_show = self.retreiver.get_output_slice(step, self.step_axis, self.axis_step, True)
                elif self.channel in self.retreiver.get_output_channels():
                    im_show = self.retreiver.get_slice(self.channel, step, self.step_axis, self.axis_step, True)
                else:
                    im_show = self.retreiver.get_slice(self.channel, step, self.step_axis, self.axis_step, False)
                if (not self.transformation_func is None)  and apply_transformation_func:

                    im_show = self.transformation_func(im_show, step % self.time_steps)

                else:
                    im_show[im_show < self.input_range[0]] = self.input_range[0]
                    im_show[im_show > self.input_range[1]] = self.input_range[1]
                    im_show = im_show - self.input_range[0]
                    im_show = im_show/(self.input_range[1] - self.input_range[0])
            return im_show



    def set_channel(self, val: int):
        self.channel = val
        self.general_change = True
        self.cached_data = {}
    
    def set_cropping(self, lower_crop: float, higher_crop: float):
        self.input_range = np.array([lower_crop, higher_crop])
        self.general_change = True
        self.cached_data = {}
    def set_transformation_function(self, transform: Callable[[np.ndarray], np.ndarray] | List[Callable[[np.ndarray], np.ndarray]]):
        self.transformation_func = transform
        self.general_change = True
        self.cached_data = {}
    
    def set_apply_transformation_func(self, val: bool):
        if self.apply_transformation_func == val:
            return
        self.apply_transformation_func = val
        self.general_change = True
        self.cached_data = {}

    def switch_running(self):
        self.running = not self.running
        if self.running:
            return "Stop"
        else:
            return "Start"
    
    def set_pending_step(self, step: int):
        self.pending_step = step

    def set_color_map(self, color_map: str):
        """
        Contains workarounds. Necessary for properly switching colorbar.
        """
        self.anim._stop()
        self.current_color_map = color_map
        self.complete_redraw = False
        self.figure.delaxes(self.cbar_ax)
        self.cbar_ax = self.figure.add_subplot(1, 11, 11)
        self.image = self.axes.imshow(self._get_current_image(), cmap=self.current_color_map, vmin=0, vmax=1)
        self._cbar = self.figure.colorbar(mappable=self.image, cax=self.cbar_ax)
        self.general_change = True
        self.figure.tight_layout()
        self.anim = animation.FuncAnimation(
            self.figure,
            self.callback_animation,
                frames=20,
                interval=200,
                blit=True,
        )
        

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
        self.cached_data = {}
        return self.axis_step
    