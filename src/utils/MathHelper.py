import numpy as np
from typing import Tuple, Dict, List, Callable
from scipy.stats import gaussian_kde
import scipy.ndimage as ndim
from src.utils.util_enumy import InterpolationMode
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture 
from skimage.filters import threshold_multiotsu
from scipy.interpolate import RegularGridInterpolator
class MathHelper():
    """
    Helper class for math. Contains functions that are mainly used for visualization and intensity mappin.
    
    """
    

    @classmethod
    def _normalize_state_dict_spatial_dimensions(cls, state_dict: Dict[int, np.ndarray], norm_method: str = 'nearest'):
        """Normalizes state dictionary to uniform spatial dimensionsions. Upscales individual steps 
           to match the spatial dimensions of the largest step available. Performs upscaling across all dimensions.

        Args:
            state_dict (Dict[int, np.ndarray]): Dictionary containing the state. Has dimension (H, W, [Z], channels)
            norm_method (str, optional): Method used for upscaling. Defaults to 'nearest'.
        """
        state_keys: List[int] = list(state_dict.keys())
        is_3d: bool = len(state_dict[state_keys[0]].shape) == 4
        max_height: int = -1
        max_width: int = -1
        max_depth: int = -1

        if is_3d:
            for k in state_keys:
                h, w, z, _ = state_dict[k].shape
                if h > max_height:
                    max_height = h
                if w > max_width:
                    max_width = w 
                if z > max_depth:
                    max_depth = z
            for k in state_keys:

                state: np.ndarray = state_dict[k]
                if (not state.shape[0] < max_height) and (not state.shape[1] < max_width) and (not state.shape[2] < max_depth):
                    continue
                new_state: np.ndarray = np.zeros((max_height, max_width, max_depth, state.shape[3]))
                for channel in range(state.shape[3]):
                    channel_data: np.ndarray = state[:, :, :, channel]
                    #shape (H, W, Z)
                    H, W, Z = channel_data.shape
                    
                    H_sample = np.linspace(0, H-1, H)
                    W_sample = np.linspace(0, W-1, W)
                    Z_sample = np.linspace(0, Z-1, Z)
                    grid_interpolator = RegularGridInterpolator((H_sample, W_sample, Z_sample), channel_data, 
                                                                method=norm_method, bounds_error=False, fill_value=0.0)
                    

                    
                    
                    
                    row_sampling: np.ndarray = np.linspace(0, H-1, max_height)
                    col_sampling: np.ndarray = np.linspace(0, W-1, max_width)
                    z_sampling: np.ndarray = np.linspace(0, Z-1, max_depth)
                    r_index, c_index, z_index = np.meshgrid(row_sampling, col_sampling, z_sampling, indexing='ij', sparse=False)
                    
                    r_index = r_index.flatten()
                    c_index = c_index.flatten()
                    z_index = z_index.flatten()
                    
                    r_index = r_index.reshape((1, r_index.shape[0]))
                    c_index = c_index.reshape((1, c_index.shape[0]))
                    z_index = z_index.reshape((1, z_index.shape[0]))
                    
                    sampling_indexes =  np.transpose(np.concatenate((r_index, c_index, z_index), axis=0), (1, 0))
                    sampled = grid_interpolator(sampling_indexes)
                    sampled = sampled.reshape((max_height, max_width, max_depth))
                    new_state[:, :, :, channel] = sampled
                state_dict[k] = new_state
                
            
            
            
        else:
            for k in state_keys:
                h, w, _ = state_dict[k].shape
                if h > max_height:
                    max_height = h
                if w > max_width:
                    max_width = w 
                
            for k in state_keys:
                state: np.ndarray = state_dict[k]
                if (not state.shape[0] < max_height) and (not state.shape[1] < max_width):
                    continue
                for channel in range(state.shape[2]):
                    channel_data: np.ndarray = state[:, :, :, channel]
                    #shape (H, W, Z)
                    H, W = channel_data.shape
                    
                    H_sample = np.linspace(0, H-1, H)
                    W_sample = np.linspace(0, W-1, W)
                    grid_interpolator = RegularGridInterpolator((H_sample, W_sample), channel_data, 
                                                                method=norm_method, bounds_error=False, fill_value=0.0)
                    

                    
                    
                    
                    row_sampling: np.ndarray = np.linspace(0, H-1, max_height)
                    col_sampling: np.ndarray = np.linspace(0, W-1, max_width)
                    r_index, c_index = np.meshgrid(row_sampling, col_sampling, indexing='ij', sparse=False)
                    
                    r_index = r_index.flatten()
                    c_index = c_index.flatten()
                    
                    r_index = r_index.reshape((1, r_index.shape[0]))
                    c_index = c_index.reshape((1, c_index.shape[0]))
                    
                    sampling_indexes =  np.transpose(np.concatenate((r_index, c_index), axis=0), (1, 0))
                    sampled = grid_interpolator(sampling_indexes)
                    sampled = sampled.reshape((max_height, max_width))
                    state[:, :, channel] = sampled
                state_dict[k] = state
        return state_dict


    @classmethod
    def get_percentile_values(cls, data: List[np.ndarray]| np.ndarray, data_perentage: float, subsampling: float = None) -> Tuple[float, float]:
        """Returns Range that captures data_percentage% of the input data. 
        Args:
            data: List of Data Arrays
            data_percentage: Percentage of the underlying Data that is to be captured in the range
            subsampling: Sets whether the array should be subsampled before determining the percentile ranges. 
            Gives Percentage of data to which it should be subsampled.
            While this increases performance, it may reduce the accuracy of the returned range.
        Returns: lower and upper bound for the range
        """
        if isinstance(data, List):
            whole_data = np.concatenate(data, axis=0)
        else:
            whole_data = data
        if not subsampling is None:
            target_number = int(np.floor(whole_data.shape[0]*subsampling))
            whole_data = cls.get_subsampled_array_of_desired_size(whole_data, target_number)
        temp = (1-data_perentage)/2
        lower, upper = np.percentile(whole_data, [temp*100, (1-temp)*100])
        if lower == upper:
            lower = np.min(data)
            upper = np.max(data)
            lower = lower - 0.01
            upper = upper - 0.01
        return (lower, upper)
    

    @classmethod
    def get_gradient_map(cls, data: np.ndarray) -> np.ndarray:
        """Uses convolution to map all pixels to their gradient magnitude.

        args: 
            data: np.ndarray: 3Dimensional array

        returns: 
            reurns 3D array with same dimensions containing gradient magnitudes
        
        """
        x_zob = cls.zobelX(data)
        y_zob = cls.zobelY(data)
        z_zob = cls.zobelZ(data)
        x_zob = x_zob**2
        y_zob = y_zob**2
        z_zob = z_zob**2
        ret = x_zob + y_zob + z_zob
        return ret**(0.5)

    @classmethod
    def zobelX(cls, data: np.ndarray) -> np.ndarray:
        """Takes an np.ndarray and applies a 3D Sobel filter along the first axis.
        args:
            data: np.ndarray, 3D
        returns:
            np.ndarray filtered through a zobel filter
        
        """
        zobelx = np.array([1, 0, -1])
        zobely = np.array([1, 2, 1])
        zobelz = np.array([1, 2, 1])
        temp = ndim.convolve1d(data, zobelx, axis=0, mode="mirror")
        temp = ndim.convolve1d(temp, zobely, axis=0, mode="mirror")
        temp = ndim.convolve1d(temp, zobelz, axis=0, mode="mirror")
        return temp          

    @classmethod
    def zobelY(cls, data: np.ndarray) -> np.ndarray:
        """Takes an np.ndarray and applies a 3D Sobel filter along the second axis.
        args:
            data: np.ndarray, 3D
        returns:
            np.ndarray filtered through a zobel filter
        
        """
        zobelx = np.array([1, 2, 1])
        zobely = np.array([1, 0, -1])
        zobelz = np.array([1, 2, 1])
        temp = ndim.convolve1d(data, zobelx, axis=0, mode="mirror")
        temp = ndim.convolve1d(temp, zobely, axis=0, mode="mirror")
        temp = ndim.convolve1d(temp, zobelz, axis=0, mode="mirror")
        return temp
    

    @classmethod
    def zobelZ(cls, data: np.ndarray) -> np.ndarray:
        """Takes an np.ndarray and applies a 3D Sobel filter along the third axis.
        args:
            data: np.ndarray, 3D
        returns:
            np.ndarray filtered through a zobel filter
        
        """
        zobelx = np.array([1, 2, 1])
        zobely = np.array([1, 2, 1])
        zobelz = np.array([1, 0, -1])
        temp = ndim.convolve1d(data, zobelx, axis=0, mode="mirror")
        temp = ndim.convolve1d(temp, zobely, axis=0, mode="mirror")
        temp = ndim.convolve1d(temp, zobelz, axis=0, mode="mirror")
        return temp
    

    @classmethod
    def fit_gaussianMM_Gradient(cls, data: np.ndarray, n_components: int,  max_iter: int = 100, n_init: int = 3, init_method="kmeans", subsampling: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Fits a gaussian mixture model on the relationship between Intensity 
        and Gradient magnitude. Returns the parameters of the fitted Gaussian Mixture model.
        Does not yield any useful results
        args:
            data: np.ndarray, containing data points: Should be 3D array containing intesities, alternatively 
            a 4D array with dimensions [x, y, z, n]. Only the first 3 array dimensions will be treated as spatial dimensions, 
            the last one can be used to enumerate arrays
            n_components: number of components for the Gaussian Mixture model
            max_iter: MAximum number of Iterations performed during EM Maximization
            n_init: Number of attempted initializations: Best result is chosen
            init_method: Method used for initialization: kmeans is basically the only reasonable choice
            subsampling: Size of subsampled array on which the computations are performed. has no 
            effect if none or greater than original number of data points.
        returns:
            (means, covariances, data):
            means have shape [n_components, 2]
            covariances have shape [n_components, 2, 2]
            data: [n_points, 2]: data on which optimizaiton was performed
        """
        if len(data.shape) == 3:
            gradients = cls.get_gradient_map(data)
            data = np.ndarray.flatten(data)
            grads = np.ndarray.flatten(gradients)
        elif len(data.shape)==4:
            n = data.shape[3]
            grads = np.zeros(data.shape)
            for i in range(0, n):
                grads[..., i] = cls.get_gradient_map(grads[..., i])
            data = np.ndarray.flatten(data)
            grads = np.ndarray.flatten(data)
        else:
            raise Exception("Unsupported dimensions for Input array")
        if not subsampling is None:
            data = cls.get_subsampled_array_of_desired_size(data, subsampling)
            grads = cls.get_subsampled_array_of_desired_size(grads, subsampling)
        arr = np.array([data, grads])
        arr = np.transpose(arr)
        gm = GaussianMixture(n_components=n_components, covariance_type='full', max_iter=max_iter, n_init=n_init, init_params=init_method)
        gm = gm.fit(arr)
        means = gm.means_
        cov = gm.covariances_
        return (means, cov, arr)

    @classmethod
    def get_pdf_hist_method(cls, data: np.ndarray, subsampling: float = None, borders: Tuple[float, float] = None, bins: int = 200, normalized_max_values: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Computes PDF on data. Data is not approximated using a fitted distribution but rather the shape of an equi Width Histogram is plotted essentially.
        Only works with 1D np arrays.
        Args:
            data: Array containing the data
            subsampling: Percentage of the data that is to be used to compute the PDF. Aggressive subsampling might lead to lack of detail
            but increases performance
            borders: Tuple containing left and right borders for the Data. All data outside of this range is ignored when creating the PDF
            bins: Number of bins of the resulting Histogram
        Returns: 
            x_values, f(x)_value sof Histogram
        """
        if not subsampling is None:
            l = data.shape[0]
            num_elements = int(np.floor(l*subsampling))
            data = cls.get_subsampled_array_of_desired_size(data, num_elements)
        if data.min() == data.max() and borders is None:
            raise Exception("Can Determine bin dimensions from Zero array")
        if borders is None:
            borders = (data.min(), data.max())

        bin_densities, x_s = np.histogram(data, bins=bins, range=(borders[0], borders[1]), density=False)
        width = x_s[1]-x_s[0]
        x_s = x_s[:-1] + width/2.0
        bin_densities = bin_densities*(normalized_max_values/float(bin_densities.max()))
            
        return x_s, bin_densities
    

    @classmethod 
    def get_pdf_surface_plot_binned(cls, data: List[np.ndarray], subsamplling: float = None, bins: int = 200, consider_percentile: float = 0.985) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Creates PDF Surface for Data. 
        data is list of 1D Arrays. List===z-Dimensions.
        Args:
            data: List of arrays containing the data. PDFs are computed from each array and staggered along the y axis. 
            subsampling: Whether the data will be subsampled
            bins: Resolution of the PDF Surface plot along the X-Axis
            consider_percentile: Sets how much of the data is plotted. Elimining outliers often leads to a Plot with better scale and more Detail in relevant areas
        returns:
            np.ndarray: X-Data (x_d, )
            np.ndarray: Y-Data (y_d, )
            np.ndarray: Z-Data (x_d, y_d)
            float: lower_limit that is used
            float: higher Limit that is used
        """
        low, high = cls.get_percentile_values(data, data_perentage=consider_percentile, subsampling=subsamplling)
        x_dim = bins
        y_dim = len(data)
        z_values = np.zeros((x_dim, y_dim))
        x_values = None
        y_values = np.linspace(0, len(data) - 1, len(data))
        for i in range(0, y_dim):
            x, z_values[:, i] = cls.get_pdf_hist_method(data[i], subsampling=subsamplling, borders=(low, high), bins=bins)
            if x_values is None:
                x_values = x
        return (x_values, y_values, np.transpose(z_values), low, high)


    @classmethod
    def get_surface_plot_from_mappings(cls, bin_mappings: List[Callable[[np.ndarray], np.ndarray]], low_limit:float, high_limit:float, resolution: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Takes a list of mappings and generates a Surface plot from those mappings. 
        list <===> z-Axis.
        Args:
            bin_mappings: list of Mapping functions
            low_limit: lower x limit of the surface plot
            high_limit: higher x limit of the surface plot
            resolution: Resolution of the Surfac plot for the X axis        
        
        """
        yres = len(bin_mappings)

        X = np.linspace(low_limit, high_limit, resolution)
        Y = np.linspace(0, len(bin_mappings), len(bin_mappings))
        Z = np.zeros((resolution, yres))
        for i in range(0, yres):
            x_d = np.linspace(low_limit, high_limit, resolution)
            func = bin_mappings[i]
            Z[:, i] = func(x_d)
        return (X, Y, np.transpose(Z))
            

    
    @classmethod
    def get_mapping_parameters_multi_otsu(cls, arr: np.ndarray, bins: int, target_range: np.ndarray, is_sorted: bool = False) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Takes an array sorts it and returns indexes that seperate the array into #bins bins with maximal cross class variance.
        Only works for less than 10 bins in reasonable time.


        Args:
            arr (np.ndarray): Array which is to be devided into evenly sized bins
            bins (int): number of bins into which the array should be devided
            target_range: array containing two values: Lower bound of mapping range and higher bound of mapping range.

        Returns:
            Tuple[np.ndarray, List[np.ndarray]]: element1: The sorted array. The original array should be left untouched. 
            List[np.ndarray]: Returns a list that maps bin borders to their corresponding matches in the target range. 
            bin borders include lowest element and highest element 

        Much room for further performance tweaks, see
        https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_multiotsu
        """
        min_bin = 5
        max_bin = 10
        bin_numbers = {5: 128, 6: 68, 7: 43, 8: 34, 9: 28, 10: 24}
        util_bins = 2
        if bins not in bin_numbers:
            if bins < min_bin:
                util_bins = 256
            elif bins > max_bin:
                util_bins = bins
        else:
            util_bins = bin_numbers[bins]
        if not is_sorted:
            srtd = np.sort(arr)
        else:
            srtd = np.copy(arr)
        len = arr.size
        min_val = srtd[0]
        max_val = srtd[-1]
        offset_arr = arr - min_val
        mappings = list()
        unique = np.unique(srtd)
        if bins == 1:
            if max_val <= min_val:
                max_val = min_val + 0.0001
            mappings.append(np.array([min_val, target_range[0]]))
            mappings.append(np.array([max_val, target_range[1]]))
            return (srtd, mappings)
        if unique.shape[0] < util_bins:
            util_bins = int(unique.shape[0]/2.0)
            if util_bins <= bins:
                if max_val <= min_val:
                    max_val = min_val + 0.0001
                mappings.append(np.array([min_val, target_range[0]]))
                mappings.append(np.array([max_val, target_range[1]]))
                return (srtd, mappings)

        bin_edges = threshold_multiotsu(offset_arr, classes=bins, nbins=util_bins)
        bin_edges = bin_edges + min_val
        step_width = (target_range[1] - target_range[0])/bins
        oof = bin_edges.shape
        
        mappings.append(np.array([min_val, target_range[0]]))
        accumulated = target_range[0]
        for i in range(bins-1):
            accumulated += step_width
            mappings.append(np.array([bin_edges[i], accumulated]))
        # have zeroth and last
        mappings.append(np.array([max_val, target_range[1]]))
        return (srtd, mappings) 


    @classmethod
    def get_evenly_spaced_bins(cls, arr: np.ndarray, bins: int, is_sorted: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Takes an array sorts it and returns indexes that seperate the array into #bins evenly sized segments.

        Args:
            arr (np.ndarray): Array which is to be devided into evenly sized bins
            bins (int): number of bins into which the array should be devided

        Returns:
            Tuple[np.ndarray, np.ndarray]: element1: The sorted array. The original array should be left untouched. 
            element2: An array containing indexes of the bins in the sorted array. contains the right bin indexes. 
            bins can be adressed as arr[bin_(n-1):bin_(n)]
        """
        if not is_sorted:
            srtd = np.sort(arr)
        else:
            srtd = arr
        len = arr.size
        if len < bins:
            return (srtd, np.arange(len))
        index_interval = int(np.floor(len/bins))
        bin_indexes = np.zeros(bins, dtype=np.int32)
        running_count = index_interval
        for i in range(bins-1):
            bin_indexes[i] = int(running_count)
            running_count += index_interval
        bin_indexes[bins-1] = len
        return (srtd, bin_indexes) 

    @classmethod
    def get_subsampled_array_of_desired_size(cls, arr: np.ndarray, size: int):
        if size > arr.size:
            return arr
        interval: int = int(np.floor(arr.size/size))
        return arr[::interval].copy()
    
    @classmethod
    def get_pdf_surface_plot(cls, data: List[np.ndarray], limit_low: float=None, limit_high: float = None, pdf_resolution: int = 200, covariance: float=0.002, max_density: float = 1.0, desired_sample_size: int = 15000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates pdf surface plot for given data. Intended to visualize changes to Intensity PDF over Time for NCA Inference. 
        Args:
            data: List(np.ndarray): List of arrays. PDF Function will be computed from each entry of the list. List entries will be staggered along the y-Axis in the surface Plot: 
            limit_low: Lower Limit to which PDF should be plotted. If left empty, lowest data point in the List will be chosen as lower limit.
            limit_heigh: Heigher Limit to which PDF should be plotted. If left empty, highest Data point in the List will be chosen as higher Limit. 
            Not explicitly specifying higher and lower limits has negative impact on performance
            pdf_resolution: Resolution of the Surface plot along the X-Axis: Higher Resolution: Smoother Details can be captured. 
            covariance: Covariance for gaussian kde. Higher Covariance -> Smoother PDF. From experience a Value greater then 0.05 leads to unhelpful results.
            max_density: Maximum Density to which Values are to be normalized. Surface plot will have Z-Values between [0, max_density]       
        Returns:
            np.ndarray: (xs, ) -> 1D Array containing all x Coordinates
            np.ndarray: (len(data), ) -> 1D Array containing all y coordinates
            np.ncarray: (len(data), xs) -> 2D Array containing the corresponding Z values 
        """
        if limit_low is None or limit_high is None:
            no_high = False
            no_low = False
            if limit_high is None: 
                no_high = True
            if limit_low is None:
                no_low = True

            for data_arr in data:
                arr_use = cls.get_subsampled_array_of_desired_size(data_arr, desired_sample_size)
                if no_low:
                    min_v = np.min(arr_use)
                    if limit_low is None:
                        limit_low = min_v
                    elif min_v < limit_low:
                        limit_low = min_v
                if no_high:
                    max_v = np.max(arr_use)
                    if limit_high is None:
                        limit_high = max_v
                    elif max_v > limit_high:
                        limit_high = max_v

        xlen = pdf_resolution
        ylen = len(data)
        z_data = np.zeros((ylen, xlen))
        x_data = np.linspace(limit_low, limit_high, pdf_resolution)
        y_data = np.linspace(0, len(data), ylen)
        for i in range(0, ylen):
            if data[i].min() == data[i].max()==0.0:
                ys = np.zeros((pdf_resolution, ))
            else:
                arr_use = cls.get_subsampled_array_of_desired_size(data[i], desired_sample_size)
                xs, ys = cls.get_pdf_for_data(arr_use, limit_low=limit_low, limit_high=limit_high, pdf_resolution=pdf_resolution, covariance=covariance, max_density=max_density)
            z_data[i]=ys
        return (x_data, y_data, np.transpose(z_data))
            


    @classmethod
    def get_pdf_for_data(cls, data: np.ndarray, limit_low: float = None, limit_high: float = None, pdf_resolution: int = 200, covariance: float= 0.05, max_density: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Function that computes plot for PDF for given Data. 

        Args: 
            data: 1D (n, ) array containing Data Points
            limit_low: Lower Limit to which PDF should be plotted. If left empty, lowest data point will be chosen as lower limit
            limit_heigh: Heigher Limit to which PDF should be plotted. If left empty, highest Data point will be chosen as limit
            pdf_resolution: Number of points to be computed in the range. 
            covariance: Covariance for gaussian kde. Higher Covariance -> Smoother PDF
            max_density: Maximum Density to which Values are to be normalized
        """
        if limit_low is None:
            limit_low = np.min(data)
        if limit_high is None:
            limit_high = np.max(data)
        density = gaussian_kde(data)
        xs = np.linspace(limit_low,limit_high,pdf_resolution)
        density.covariance_factor = lambda : covariance
        density._compute_covariance()
        ys = density(xs)
        max_val = np.max(ys)
        ys = ys*(max_density/max_val)
        return (xs, ys)

    @classmethod
    def get_mapping_parameters_for_binned_array(cls, arr: np.ndarray, bins: np.ndarray, target_range: np.ndarray) -> List[np.ndarray]:
        """Takes a sorted array as well as bin indexes on that array and performs an equi width mapping of the 
        bins onto the traget range. Returns resulted mapping of bin edges to the target range.

        Args:
            arr (np.ndarray): A sorted array of floats.
            bins (np.ndarray): An array definining bin indexes on arr. 
                Always contains the tight bin edges. Bin edges are given exactly one over the index of the last element of the 
                corresponding bin. The last index in bins is always equal to arr.size
            target_range (np.ndarray): np array containing two values. Upper and lower bound of target range

        Returns:
            List[np.ndarray]: Returns a list that maps bin borders to their corresponding matches in the target range. 
        """
        ret_list = list()
        ret_list.append(np.array([arr[0], target_range[0]]))
        mapping_range = target_range[1] - target_range[0]
        interval = mapping_range/bins.size
        runningCount = interval
        num_bins = bins.size
        for i in range(0, num_bins - 1):
            ret_list.append(np.array([arr[bins[i]-1], runningCount]))
            runningCount += interval
        ret_list.append(np.array([arr[-1], target_range[1]]))
        for i in range(1, len(ret_list)):
            #for numerical stability in edge cases. Dose sometimes happen depending on mapping settings and random activations...
            if ret_list[i][0]<=ret_list[i-1][0]:
                ret_list[i][0] = ret_list[i-1][0] + 0.0001
        return ret_list
    
    @classmethod
    def generate_mapping(cls, arr_mapping: List[np.ndarray], interpolation_mode: InterpolationMode, q_value: float = np.pi/4) -> Callable[[np.ndarray], np.ndarray]:
        if interpolation_mode == InterpolationMode.LINEAR:
            return cls.pl_arr_transform_point_mapping(arr_mapping)
        elif interpolation_mode == InterpolationMode.TANGENS:
            return cls.pl_arr_transform_point_mapping_tangens(arr_mapping, q_value=q_value)

    @classmethod
    def pl_arr_transform_point_mapping(cls, arr_mapping: List[np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
        """Receives a set of point mappings in ascending order and generates a picewise linear array transformation function 
        that transforms input arrays linearly according to the specified transformation function.

        Args:
            arr_mapping (List[np.ndarray]): Specifies mapping from input range into target rage through pairs of points. 
            Has to include mappings for lowest and highest point in target range

        Returns:
            Callable[[np.ndarray], np.ndarray]: Returns a function that takes arrays and applies the piecewise linear transformation to them
        """
        linear_mappings = list() # should contain nd arrays with two elements: [offset, gradient]
        # contains function parameters only for individual ranges, so one less element then arr_mappin
        for i in range(1, len(arr_mapping)):
            a = arr_mapping[i][1] - arr_mapping[i-1][1]
            b = arr_mapping[i][0] - arr_mapping[i-1][0]
            c = arr_mapping[i-1][0]
            u = arr_mapping[i-1][1]
            if a == 0.0 or b==0.0:
                gradient = 0
            else:
                gradient = a/b
            offset = u - (gradient)*c
            linear_mappings.append(np.array([offset, gradient]))
        def piecewise_linear_mapping(data: np.ndarray, linear_mappings, arr_mapping) -> np.ndarray:
            """This function applies a piecewise linear transformation to all elements in the array. 
            Array can have any dimension
            

            Args:
                data (np.ndarray): Input array
            """
            ret_arr = np.zeros(data.shape)
            arr = np.copy(data)
            arr[arr < arr_mapping[0][0]] = arr_mapping[0][1]
            arr[arr >= arr_mapping[0][0]] = 0
            ret_arr += arr
            for i in range(0, len(linear_mappings)):
                offset = linear_mappings[i][0]
                gradient = linear_mappings[i][1]
                arr = np.copy(data)
                #lower bound is in
                out_of_range_indexes = np.logical_or(arr < arr_mapping[i][0], arr >= arr_mapping[i+1][0])
                arr[out_of_range_indexes] = 0
                ones_mask = np.ones(data.shape)
                ones_mask[out_of_range_indexes] = 0
                ret_arr += offset*ones_mask + arr*gradient
            arr = np.copy(data)
            mm = arr >= arr_mapping[-1][0] 
            m2 = arr < arr_mapping[-1][0] 
            arr[mm] = arr_mapping[-1][1]
            arr[m2] =0
            ret_arr += arr

            return ret_arr
        return lambda data, l_map=linear_mappings, arr_map=arr_mapping : piecewise_linear_mapping(data, l_map, arr_map)
    




    @classmethod
    def pl_arr_transform_point_mapping_tangens(cls, arr_mapping: List[np.ndarray], q_value: float = np.pi/4) -> Callable[[np.ndarray], np.ndarray]:
        """Receives a set of point mappings in ascending order and generates a tangens array transformation function 
        that transforms input arrays using a tangens function according to the specified transformation function.

        Args:
            arr_mapping (List[np.ndarray]): Specifies mapping from input range into target rage through pairs of points. 
            Has to include mappings for lowest and highest point in target range

        Returns:
            Callable[[np.ndarray], np.ndarray]: Returns a function that takes arrays and applies the piecewise linear transformation to them
        """
        tangens_mappings = list() # should contain nd arrays with two elements: [offset, gradient]
        # contains function parameters only for individual ranges, so one less element then arr_mappin
        if q_value <= 0 or q_value >= np.pi/2:
            raise Exception("Invalid Q value for tangens mapping")
        def piecewise_tangens_mapping(data: np.ndarray, arr_mapping, q_val: float) -> np.ndarray:
            """This function applies a piecewise linear transformation to all elements in the array. 
            Array can have any dimension
            

            Args:
                data (np.ndarray): Input array
            """
            ret_arr = np.zeros(data.shape)
            arr = np.copy(data)
            arr[arr < arr_mapping[0][0]] = arr_mapping[0][1]
            arr[arr >= arr_mapping[0][0]] = 0
            ret_arr += arr
            for i in range(0, len(arr_mapping) - 1):
                a = arr_mapping[i][0]
                b = arr_mapping[i+1][0]
                x = arr_mapping[i][1]
                y = arr_mapping[i+1][1]
                arr = np.copy(data)
                #lower bound is in
                out_of_range_indexes = np.logical_or(arr < arr_mapping[i][0], arr >= arr_mapping[i+1][0])
                arr[out_of_range_indexes] = 0
                ones_mask = np.ones(data.shape)
                ones_mask[out_of_range_indexes] = 0
                
                temp = (-q_val)*ones_mask + (arr - a)*((2*q_val)/((b-a)))
                z = np.tan(q_val)
                ret_arr += ones_mask*(x + (np.tan(temp) + z)*((y-x)/(2*z)))
                
            arr = np.copy(data)
            arr[arr >= arr_mapping[-1][0]] = arr_mapping[-1][1]
            arr[arr < arr_mapping[-1][0]] =0
            ret_arr += arr
            ret_arr[ret_arr < arr_mapping[0][1]] = arr_mapping[0][1]
            ret_arr[ret_arr > arr_mapping[-1][1]] = arr_mapping[-1][1]
            return ret_arr
        return lambda data, arr_map=arr_mapping, q_val=q_value : piecewise_tangens_mapping(data, arr_map, q_val)
    
    @classmethod
    def get_piecewise_linear_matrix_transform(cls, data: np.ndarray, num_bins: int, target_range: np.ndarray, interpolation_mode: InterpolationMode = InterpolationMode.LINEAR, 
                                              q_value: float = np.pi/4) -> Callable[[np.ndarray], np.ndarray]:
        """Takes a one dimensional array of data points and generates a piecewise linear mapping from the Value Range of the 
        Data Points into a target range.
        Points do not need to be sorted

        Args:
            data (np.ndarray): 1D ndarray for which the transformation is computed
            num_bins (int): number of equi height bins into which the data is split. 
            Higher number of bins means that the transformation is able to capture finer 
            detail in the value density
            target_range (np.ndarray): Target range onto which the data is mapped
            interpolation mode: Sets which function is used to interpolate
            q_value: q_value that is used for tangens interpolation

        Returns:
            Callable[[np.ndarray], np.ndarray]: Returns a function which takes an input array 
            (should have same value range as the data array given as a parameter, everything outside of that is set to lower or upper bound of target range) 
            and maps the individual values into the target range according to an equi height histogram linear transformation
        """
        
        sorted_input, bins = MathHelper.get_evenly_spaced_bins(data, num_bins)
        point_mapping = MathHelper.get_mapping_parameters_for_binned_array(sorted_input, bins, target_range)
        if interpolation_mode == InterpolationMode.LINEAR:            
            return MathHelper.pl_arr_transform_point_mapping(point_mapping)
        elif interpolation_mode == InterpolationMode.TANGENS:
            return MathHelper.pl_arr_transform_point_mapping_tangens(point_mapping, q_value=q_value)
    @classmethod
    def get_individual_piecewise_linear_matrix_transformation_extended(cls, data: List[np.ndarray], num_bins: int, target_range: np.ndarray, 
                interpolation_mode: InterpolationMode = InterpolationMode.LINEAR, q_value: float = np.pi/4) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], List[List[np.ndarray]], List[np.ndarray]]:
        """Takes a list of one dimensional array of data points and generates a list of piecewise linear mapping from the Value Range of the 
        Data Points into a target range.
        Points do not need to be sorted

        Args:
            data List[np.ndarray]: 1D ndarray for which the transformation is computed
            num_bins (int): number of equi height bins into which the data is split. 
            Higher number of bins means that the transformation is able to capture finer 
            detail in the value density
            target_range (np.ndarray): Target range onto which the data is mapped
            q_value: q_value that is used for tangens mapping.

        Returns:
            Callable[[np.ndarray], np.ndarray]: Returns a function which takes an input array 
            (should have same value range as the data array given as a parameter, everything outside of that is set to lower or upper bound of target range) 
            and maps the individual values into the target range according to an equi height histogram linear transformation
            Returns the point mapping that was generated as well as the bins in Addition
        """
        mapping_funcs = list()
        bin_mappins = list()
        bins_data = list()
        for d in data:    
            sorted_input, bins = MathHelper.get_evenly_spaced_bins(d, num_bins)
            point_mapping = MathHelper.get_mapping_parameters_for_binned_array(sorted_input, bins, target_range)
            if interpolation_mode == InterpolationMode.LINEAR:            
                mapping_funcs.append(MathHelper.pl_arr_transform_point_mapping(point_mapping))
            elif interpolation_mode == InterpolationMode.TANGENS:
                mapping_funcs.append(MathHelper.pl_arr_transform_point_mapping_tangens(point_mapping, q_value=q_value))
            

            bin_mappins.append(point_mapping)
            bins_data.append(bins)
        return (mapping_funcs, bin_mappins, bins_data)
        

    @classmethod
    def get_piecewise_linear_matrix_transform_extended(cls, data: np.ndarray, num_bins: int, target_range: np.ndarray, interpolation_mode: InterpolationMode = InterpolationMode.LINEAR, 
                                                       q_value: float = np.pi/4) -> Tuple[Callable[[np.ndarray], np.ndarray], List[np.ndarray], np.ndarray]:
        """Takes a one dimensional array of data points and generates a piecewise linear mapping from the Value Range of the 
        Data Points into a target range.
        Points do not need to be sorted

        Args:
            data (np.ndarray): 1D ndarray for which the transformation is computed
            num_bins (int): number of equi height bins into which the data is split. 
            Higher number of bins means that the transformation is able to capture finer 
            detail in the value density
            target_range (np.ndarray): Target range onto which the data is mapped
            q_value: q_value that is used for tangens mapping.

        Returns:
            Callable[[np.ndarray], np.ndarray]: Returns a function which takes an input array 
            (should have same value range as the data array given as a parameter, everything outside of that is set to lower or upper bound of target range) 
            and maps the individual values into the target range according to an equi height histogram linear transformation
            Returns the point mapping that was generated as well as the bins in Addition
        """
        
        sorted_input, bins = MathHelper.get_evenly_spaced_bins(data, num_bins)
        point_mapping = MathHelper.get_mapping_parameters_for_binned_array(sorted_input, bins, target_range)
        if interpolation_mode == InterpolationMode.LINEAR:            
            return (MathHelper.pl_arr_transform_point_mapping(point_mapping), point_mapping, bins)
        elif interpolation_mode == InterpolationMode.TANGENS:
            return (MathHelper.pl_arr_transform_point_mapping_tangens(point_mapping, q_value=q_value), point_mapping, bins)
        



    @classmethod
    def get_flat_channel(cls, steps_dict: Dict[int, np.ndarray], channel: int) -> np.ndarray:
        if len(list(steps_dict.keys()))==0:
            return np.zeros((0, ))
        keys = list(steps_dict.keys())
        d = steps_dict[keys[0]]
        if not (channel < d.shape[-1]):
            raise Exception("Cannot extract channel")
        ret_arr = list()
        i = 0
        for key in keys:
            channel_vals = steps_dict[key][..., channel].flatten()
            ret_arr.append(channel_vals)
        total_length = 0
        for arr in ret_arr:
            total_length += arr.shape[0]
        flat_ret = np.zeros((total_length, ))
        running_index = 0      
        for a in ret_arr:
            flat_ret[running_index: running_index+a.shape[0]] = a
            running_index += a.shape[0]

        return flat_ret
            