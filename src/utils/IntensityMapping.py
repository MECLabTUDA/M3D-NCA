import numpy as np
from enum import Enum
from typing import Dict, Tuple, Literal, List, Callable, Optional
from src.utils.VisUtils import StepsDictOperations
from src.utils.util_enumy import *
from src.utils.MathHelper import MathHelper
import copy

class IntensityMapping():
    """
    This class represents an intensity mapping for output values. Should be used to translate scalar values found in network activations to 
    displayable values in fixed range for a heatmap. 

    Why has state? depending on the changes to the mapping, much already computed data can be reused. a stateful implementation 
    is computationally more efficient, especially for 3D data.
    
    This class is only ever used in a computation process. Usage example can be found in ompc_process_func.py
    """
    points: Dict[int, np.ndarray] = None
    flat_points: List[np.ndarray]|np.ndarray = None
    percentile_values: List[Tuple[float, float]]|Tuple[float, float] = None
    windowed_data: List[np.ndarray]|np.ndarray = None
    percentage: float = None
    subsampling: float
    mapping_mode: MappingMode = None
    output_range: Tuple[float, float] = (0, 1)
    interpolation_style: Literal["multilinear"] = "multilinear"
    binning_mode: BinningMode = BinningMode.EQUI_WIDTH
    channel: int
    bin_nums: int = 1
    interpolation_mode: InterpolationMode = InterpolationMode.LINEAR
    current_q_value: float = np.pi/4
    bins: List[np.ndarray]|List[List[np.ndarray]] = None
    mappings: List[Callable[[np.ndarray], np.ndarray]]|Callable[[np.ndarray], np.ndarray] = None
    lower_crop: float = None
    higher_crop: float = None
    def __init__(self, points: Dict[int, np.ndarray], output_range: Tuple[float, float], channel: int = 0, mapping_mode: MappingMode = MappingMode.SINGLE, subsampling: int = None, percentile_values: float = 0.985, 
                 bin_number: int = 1, initial_binning_mode: BinningMode = BinningMode.EQUI_WIDTH, initial_interpolation_mode: InterpolationMode = InterpolationMode.LINEAR, initial_q_value: float = np.pi/4, 
                 lower_crop: float = None, higher_crop: float = None):
        """Initializes the intensity mapping. 
        underlying points as well as output range, channel and percentiles are fixed and cannot be changed in the lifetime of the mapping. 
        If those need to be changed -> Create new mapping class.
        
        args:
            points: points on which the mapping is computed. Underlying data can have flexible size/ shape.
            output_range: output range for the mapping. A.k.a. what is the input range of my heatmap?
            channel: channel of the data on which the intensity mapping is computed
            mapping_mode: one mapping for all timesteps vs one maping for each. Latter is much more comutationally expensive.
            subsampling: Interval in which data is subsampled. DEPRECATED
            percentile_values: percentage of data which is considered. Depending on the model/ input used this 
            can remove all interesting activations if set to high. DEPRECATED
            bin_number: #bin for histogram on which mapping is computed.
            initial_binning_mode: Equi width or multi_otsu. Multi_otsu maximizes inter-bin variance
            interpolation_mode: Linear or tangens. Does not make much of a difference in praxis.
            initial_q_value: controls linearity of tangens interpolation mode. Has even less of an influence in praxis.
            lower_crop/higher_crop: Lower/higher boundary, outside of which data points are ignored for generating of mapping. Can be used to specify regions of interest or 
            remove outliers. Replacement for percentiles.
            subsampling: sets the maximum array size with which the class deals with. All larger inputs will be subsampled to this size.Probably deprecated.
        """
        self.mapping_mode = mapping_mode
        self.channel = channel
        self.points = points
        self.subsampling =subsampling
        self.output_range = output_range
        self.percentage = percentile_values
        self.interpolation_mode = initial_interpolation_mode
        self.bin_nums = bin_number
        self.binning_mode = initial_binning_mode
        self.current_q_value = initial_q_value
        
        
        if self.mapping_mode == MappingMode.SINGLE:
            self.flat_points = StepsDictOperations.channel_values(self.points, self.channel)
        elif self.mapping_mode == MappingMode.STEPPED:
            self.flat_points = list()
            for k in list(self.points.keys()):
                self.flat_points.append(self.points[k].flatten())
        else:
            self.flat_points = None
        self._set_crop(lower_crop=lower_crop, higher_crop=higher_crop)
        self._set_bin_nr(self.bin_nums)


    def set_crop(self, lower_crop: float, higher_crop: float):
        self._set_crop(lower_crop=lower_crop, higher_crop=higher_crop)
        self._set_bin_nr(self.bin_nums)

    def _set_crop(self, lower_crop: float = None, higher_crop: float = None):
        """Sets percentile values of the data to be used for the mapping. It is often usefull 
        to use this to exclude outliers. Disregards subsamplling settings. 
        Also generates subsampled data arrays that can be used for all further computations. 

        args:
            percentage: percentage of the data that is to be regarded
        """
        self.lower_crop = lower_crop
        self.higher_crop = higher_crop
        if not isinstance(self.flat_points, List):
            if (not self.higher_crop is None) and (not self.lower_crop is None):
                
                self.windowed_data = self.flat_points[np.where((self.flat_points >= self.lower_crop) & (self.flat_points <= self.higher_crop))]
            elif self.higher_crop is None and self.lower_crop is None:
                self.windowed_data = np.copy(self.flat_points)
                a=2
            elif self.lower_crop is None:
                self.windowed_data = self.flat_points[np.where(self.flat_points <= self.higher_crop)]
            elif self.higher_crop is None:
                self.windowed_data = self.flat_points[np.where(self.flat_points >= self.lower_crop)]
            if self.windowed_data.shape[0] <2:
                self.windowed_data = np.zeros(shape=(2,))

        else:
            self.windowed_data = list()
            for i in range(len(self.flat_points)):
                if (not self.higher_crop is None) and (not self.lower_crop is None):
                    
                    self.windowed_data.append(self.flat_points[i][np.where((self.flat_points[i] >= self.lower_crop) & (self.flat_points[i] <= self.higher_crop))])
                elif self.higher_crop is None and self.lower_crop is None:
                    self.windowed_data.append(self.flat_points[i][np.where(True)])
                elif self.lower_crop is None:
                    self.windowed_data.append(self.flat_points[i][np.where(self.flat_points[i] <= self.higher_crop)])
                elif self.higher_crop is None:
                    self.windowed_data.append(self.flat_points[i][np.where(self.flat_points[i] >= self.lower_crop)])
                if self.windowed_data[-1].shape[0] < 2:
                    self.windowed_data[-1] = np.zeros(shape=(2,))

            a = 2
    def get_percentiles(self) -> float:
        return self.percentage
    

    def set_mapping_mode(self, mode: MappingMode):
        """Sets mapping mode. 
        Changes will not be immediatly reflected in the mapping but have to be
        manually added through initiating a mapping refresh.
        """
        if self.mapping_mode == mode:
            return
        else:
            self.mapping_mode = mode

        if self.mapping_mode == MappingMode.SINGLE:
            self.flat_points = StepsDictOperations.channel_values(self.points, self.channel)
        elif self.mapping_mode == MappingMode.STEPPED:
            self.flat_points = list()
            for k in list(self.points.keys()):
                self.flat_points.append(self.points[k].flatten())
        else:
            self.flat_points = None
        self._set_crop(lower_crop=self.lower_crop, higher_crop=self.higher_crop)
        self._set_bin_nr(self.bin_nums)
    
    def get_mapping_mode(self) -> MappingMode:
        return self.mapping_mode

    def _set_bin_nr(self, bin_number: int):
        """
        Internal use only. sets bin number and recomputes bins used for mappig.
        """
        self.bin_nums = bin_number
        if self.bin_nums == 0:
            return
        if self.mapping_mode == MappingMode.SINGLE:
            self.bins = self._get_bins_for_data(self.windowed_data)
        else:
            bins = list()
            for d in self.windowed_data:
                bins.append(self._get_bins_for_data(d))
            self.bins = bins

    def set_binning_mode(self, binning_mode: BinningMode):
        """
        Sets the binning mode and recomputes the bin Indexes. 
        This does not recompute the mapping function.
        """
        if binning_mode == self.binning_mode:
            return copy.deepcopy(self.bins)
        self.binning_mode = binning_mode
        if self.bin_nums == 0:
            return
        if self.mapping_mode == MappingMode.SINGLE:
            self.bins = self._get_bins_for_data(self.windowed_data)
        else:
            bins = list()
            for d in self.windowed_data:
                bins.append(self._get_bins_for_data(d))
            self.bins = bins

        return copy.deepcopy(self.bins)
    

    def get_bin_nr(self) -> int:
        return self.bin_nums
            
    def _get_bins_for_data(self, data: np.ndarray):
        """
        Internal only. Generates bins.
        """
        if self.binning_mode == BinningMode.EQUI_WIDTH:
            srtd, bin_indexes =  MathHelper.get_evenly_spaced_bins(data, self.bin_nums)
            return MathHelper.get_mapping_parameters_for_binned_array(srtd, bin_indexes, self.output_range)
        elif self.binning_mode == BinningMode.MULTI_OTSU:
            srtd, mapping_indexes = MathHelper.get_mapping_parameters_multi_otsu(data, self.bin_nums, self.output_range)
            return mapping_indexes
        
    def get_bins(self) -> Optional[List[np.ndarray]|List[List[np.ndarray]]]:
        return self.bins

    def set_bin_number(self, bin_number: int):
        """Sets the number of bins used to generate the Interpolation
        During this operation bin borders are recomputed if necessary. This is relatively computationally expensive.
        """
        if bin_number == self.bin_nums:
            return
        else:
            self._set_bin_nr(bin_number)


        
    def generate_mapping(self) -> Callable[[np.ndarray, int], np.ndarray]:
        """Does internal Mapping generation. Also generates callable funciton to access the mapping. 
        Generates an efficient implementation of the mapping function which accepts numpy arrays and is shape-agnostic.
        
        returns:
            Callable[[np.ndarray, int], np.ndarray]. 
            Function takes an input array as well as the current time step and returns
            the result .
        """
        if self.bins is None:
            self._set_bin_nr(self.bin_nums)
        if self.bin_nums == 0:
            return lambda x : x
        if self.mapping_mode == MappingMode.SINGLE:
            mappings = MathHelper.generate_mapping(self.bins, self.interpolation_mode, self.current_q_value)
        elif self.mapping_mode == MappingMode.STEPPED:
            mappings = list()

            for bns in self.bins:
                mappings.append(MathHelper.generate_mapping(bns, self.interpolation_mode, q_value=self.current_q_value))
        def anon(mode: MappingMode, mapings: List[Callable[[np.ndarray], np.ndarray]]|Callable[[np.ndarray], np.ndarray], 
                 data: np.ndarray, time_step: int = 0):
            if mode == MappingMode.SINGLE:
                return mapings(data)

            elif mode == MappingMode.STEPPED:  
                if time_step == 0:
                    return data
                else:  
                    return mapings[time_step](data)
        if self.mapping_mode == MappingMode.SINGLE:
            return lambda data, time_step=0, mode=self.mapping_mode, mapings=mappings : anon(mode, mapings, data, time_step)
        else:
            return lambda data, time_step, mode=self.mapping_mode, mapings=mappings : anon(mode, mapings, data, time_step)
    def set_interpolation_mode(self, interpolation_mode: InterpolationMode):
        self.interpolation_mode = interpolation_mode

    def set_q_value(self, q_value: float):
        if q_value <= 0 or q_value >= np.pi/2:
            raise Exception("Invalid Q value for tangens mapping")
        self.current_q_value = q_value


    def get_mapping_func(self) -> Callable[[np.ndarray, int], np.ndarray]:
        """
        Fixes bug occuring when bin # is changed without applying the generated mapping.
        """
        if self.bin_nums == 0:
            return lambda x : x
        def anon(mode: MappingMode, mapings: List[Callable[[np.ndarray], np.ndarray]]|Callable[[np.ndarray], np.ndarray], 
                 data: np.ndarray, time_step: int = 0):
            if mode == MappingMode.SINGLE:
                return mapings(data)

            elif mode == MappingMode.STEPPED:  
                if time_step == 0:
                    return data
                else:  
                    return mapings[time_step](data)
        return lambda data, time_step, mode=self.mapping_mode, mapings=self.mappings : anon(mode, mapings, data, time_step)
