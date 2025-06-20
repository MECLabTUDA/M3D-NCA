from typing import Dict, Any
import numpy as np
import torch
import builtins

class StepsDictOperations:
    """
    Contains utility functions that are used to manipulate the steps dict that is obtained after a network run. 
    Ideally, data should be retreived from the steps dict only using this object. This should reduce error probability.
    """
    
    @classmethod
    def channel_values(cls, steps_dict: Dict[int, np.ndarray], channel: int,  time_steps: np.ndarray = None, step_axis: int = None, axis_steps: np.ndarray = None, 
                       subsampling_interval: int = None) -> np.ndarray:
        """Retreives all values from specified location in the steps_dict as a flat array
        This method expects all individual steps to have the same dimensions.

        Args:
            steps_dict (Dict[int, np.ndarray]): Steps Dict containing all information over a network run
            channel (int): Channel for which values should be collected. 
                Must be specified. Values can only be collected for one channel at once
            time_steps (np.ndarray, optional): _description_. Defaults to None.
                Specifies at which time step values should be collected. If left empty values are collected 
                from all time steps
            step_axis (int, optional): _description_. Defaults to None.
                Specifies axis along which steps should be selected. Can be left empty if 
                steps should be collected from the complete volume
            axis_steps (np.ndarray, optional): _description_. Defaults to None.
                Specifies steps at which the values should be collected. Can be left blank 
                if values are to be colected from the whole volume
            subsampling_interval: Sets an interval in which the array of all points should be subsampled.
            If set, a subsampled view of the original array will be returned

        Returns:
            np.ndarray: Returns flat array containing all colected values
        """
        if time_steps is None:
            time_steps = np.array([*steps_dict.keys()])
            
        if step_axis is None or axis_steps is None:
            if (step_axis is not None) or (axis_steps is not None):
                raise Exception("step_axis and axis_step both need to either be None or have reasonable values")
            arr = steps_dict[time_steps[0]]
            step_axis = 0
            axis_steps = np.array([*range(0, arr.shape[0])]) # unpacks range into ndarray
        D2 = False
        if len(arr.shape) == 3:
            D2 = True
        ret = None
        for t in time_steps:
            tmp = None
            if step_axis==0:
                if D2:
                   tmp =  steps_dict[t][:, :, channel][axis_steps]
                else: 
                    tmp = steps_dict[t][:, :, :, channel][axis_steps]
            elif step_axis == 1:
                if D2:
                   tmp =  steps_dict[t][:, :, channel][:, axis_steps]
                else: 
                    tmp = steps_dict[t][:, :, :, channel][:, axis_steps]
            elif step_axis == 2:
                if D2:
                   tmp =  steps_dict[t][:, :, channel][:, :, axis_steps]
                else: 
                    tmp = steps_dict[t][:, :, :, channel][:, :, axis_steps]
                
            if ret is None:
                ret = np.ndarray.flatten(tmp)
            else:
                ret = np.concatenate((ret, np.ndarray.flatten(tmp)))
        if subsampling_interval is None:
            return ret
        else:
            return (ret, ret[::subsampling_interval])
                
        



class OutputSliceRetreiver:
    """
    Utility class intended for convenient retreival of output slices
    from a given run of the network. Should be used to reduce error probability. 
    
    """
    output_dict: Dict[str, np.ndarray] = None
    net_config: Dict[str, Any] = None
    def __init__(self, output_dict: Dict[int, np.ndarray], net_config: Dict[str, Any]):
        """
        Initializes the class. 

        Parameters: 
        output_dict: Dict that is generated during a monitored run of the network
        net_config: Config with which the network was initialized
        """
        self.output_dict = output_dict
        self.net_config = net_config

    def get_num_steps_along_axis(self, axis: int): 
        """
        Retreives dimensionality along a given axis. 
        Assumes constant shapes across time. This assumption does not always hold, 
        but this did not seem to cause any problems so far.
        """
        keys = self.output_dict.keys()
        arr = self.output_dict[list(keys)[0]]
        return arr.shape[axis]

    def get_time_steps(self): 
        return len(list(self.output_dict.keys()))
    
    
    def get_output_slice(self,  step: int,  slice_axis: int, slice_index: int, normalize_for_output: bool = False) -> np.ndarray:
        """
        Returns specified slice of the channel that is specified in the Config as the output channel of the network. 
        
        Parameters: 
        step: step from which the slice is to be taken
        slice_axis: axis along which the slice is to be taken
        slice_index: index of the slice to be taken
        normalize_for_output: determines whether all values are normalized to be within the range of [0, 1]
        """
        step_out: np.ndarray = self.output_dict[step]
        single_channel_out = step_out[..., self.net_config["input_channels"]:self.net_config["input_channels"]+self.net_config["output_channels"]]
        single_channel_out = single_channel_out.squeeze()
        single_channel_out = np.copy(single_channel_out)
        ret = None
        if len(single_channel_out.shape) == 2:
            ret = single_channel_out
            if normalize_for_output:
                temp = torch.from_numpy(ret)
                temp = torch.sigmoid(temp)
                ret = temp.numpy()
            return ret
        if slice_axis==0:
            ret = single_channel_out[slice_index:slice_index+1, ...]
        elif slice_axis == 1:
            if len(single_channel_out.size) == 2:
                ret = single_channel_out[:, slice_index:slice_index+1]
            else:
                ret = single_channel_out[:, slice_index:slice_index+1, :]
        elif slice_axis == 2:
            ret = single_channel_out[:, :, slice_index:slice_index+1]
        ret = ret.squeeze()
        if normalize_for_output:
            temp = torch.from_numpy(ret)
            temp = torch.sigmoid(temp)
            ret = temp.numpy()
        return ret
    
    def get_output_channels(self) -> np.ndarray:
        chan_num = self.net_config["output_channels"]
        ret = np.zeros(chan_num)
        for i in range(0, chan_num):
            ret[i] = self.net_config["input_channels"] + i
        return ret
    
    def get_slice(self, channel: int,   step: int,  slice_axis: int, slice_index: int, normalize_for_output: bool = False) -> np.ndarray:
        """
        Returns specified slice. 
        
        Parameters: 
        step: step from which the slice is to be taken
        slice_axis: axis along which the slice is to be taken
        slice_index: index of the slice to be taken
        normalize_for_output: determines whether all values are normalized to be within the range of [0, 1]
        channel: Channel from which output is retreived
        """
        step_out: np.ndarray = self.output_dict[step]
        single_channel_out = step_out[..., channel]
        single_channel_out = single_channel_out.squeeze()
        single_channel_out = np.copy(single_channel_out)
        ret = None
        if len(single_channel_out.shape) == 2:
            ret = single_channel_out
            if normalize_for_output:
                temp = torch.from_numpy(ret)
                temp = torch.sigmoid(temp)
                ret = temp.numpy()
            return ret
        if slice_axis==0:
            ret = single_channel_out[slice_index:slice_index+1, ...]
        elif slice_axis == 1:
            if len(single_channel_out.size) == 2:
                ret = single_channel_out[:, slice_index:slice_index+1]
            else:
                ret = single_channel_out[:, slice_index:slice_index+1, :]
        elif slice_axis == 2:
            ret = single_channel_out[:, :, slice_index:slice_index+1]
        ret = ret.squeeze()
        if normalize_for_output:
            temp = torch.from_numpy(ret)
            temp = torch.sigmoid(temp)
            ret = temp.numpy()
        return ret
