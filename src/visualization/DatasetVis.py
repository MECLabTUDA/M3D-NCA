import sys
import os
from typing import Tuple, List, Optional, Dict, Optional, Any, Callable
import numpy as np
from PIL import Image
from dotenv import load_dotenv

import cv2

from torchio.transforms.augmentation.random_transform import RandomTransform
import torch
from qtpy.QtGui import *
from qtpy.QtWidgets import *
from qtpy.QtCore import *


from src.utils.helper import merge_img_label_gt

from src.utils.DataFlowHandler import DataFlowHandler
import torchio as tio
from .image_manipulator import ImageEditorDialog
from .qimage_util import ndarray_to_gray, ndarray_to_rgb, pixmap_to_ndarray, pixmap_to_gray_ndarray

class SliceSelectionDialog(QDialog):

    def __init__(self, items, parent=None):
        """
            Used for pushing 2D slices through a network. Shows a dialog with a
            combobox of items and returns the selected items upon dismissal.
        """
        super().__init__(parent)
        self.setWindowTitle("Select slice for 2D network")
        # Set the minimum width for the dialog
        self.setMinimumWidth(300)
        layout = QVBoxLayout()

        self.comboBox = QComboBox()
        layout.addWidget(self.comboBox)

        button_layout = QHBoxLayout()
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

        self.comboBox.addItems(map(str, items))

    def get_selected_option(self):
        return self.comboBox.currentText()

    @staticmethod
    def get_selection(items, parent=None):
        dialog = SliceSelectionDialog(items, parent)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            return dialog.get_selected_option()
        else:
            return None

class SingleDatapointVis():
    """
    An object of this type contains a QPixmap for the base image
    as well as a QPixmap for the base image merged with the fitting label
    If the visualization uses sliced 3D data, each slice will be represented
    by one object of this class.
    """
    w: int
    h: int
    base: QPixmap #this one is grayscale
    merged: Optional[QPixmap] #this one is rgb
    # labels can not be altered but kept for overlaying altered base image
    slice: np.ndarray
    label: np.ndarray 

    def __init__(self, slice: np.ndarray, lbl: np.ndarray = None, merge_func: Optional[Callable] = None):
        """
            Initializes datapoint visualization with some initial data.

            Args: 
                slice: 2D array representing a slice
                lbl: 2D array representing desired output on that slice,
                this will be layed on top of the slice in the visual representation
        """
        assert(len(slice.shape) == 2), "Only 2 dimensional array can be interpreted as image"
        if lbl is not None:
            assert(slice.shape == lbl.shape), "Datapoint and label need to have the same dimensions"

        self.w, self.h = slice.shape
        self.merge_func = merge_func

        slice_img = slice.copy()
        self.base = QPixmap(ndarray_to_gray(img=slice_img))

        self.slice = slice

        if lbl is not None:
            self.label = lbl
            label = np.zeros_like(lbl) #fake output label, set to 0s to not affect output
            # now merge base image with ground truth labels
            final_img = self.merge_func(slice, label, lbl)
            self.merged = QPixmap(ndarray_to_rgb(final_img))
        else:
            self.merged = self.base


    def update_base(self, d: QPixmap):
        """
        Update pixmaps with a modified base pixmap as input
        -> merging has to be performed again...
        """
        gray = pixmap_to_gray_ndarray(d, self.w, self.h)
        new_base = gray/255
        self.base = QPixmap(ndarray_to_gray(new_base))

        if self.label is not None:
            label = np.zeros_like(self.label) #fake output label, set to 0s to not affect output
            # now merge base image with ground truth labels
            final_img = self.merge_func(new_base, label, self.label)
            self.merged = QPixmap(ndarray_to_rgb(final_img))
        else:
            self.merged = self.base


class DatapointViewer2D():
    """
    Display a single image 
    """
    pass


class DatapointViewer3D(QScrollArea):
    """
    Display all slices as 2D images in a scrollview
    """

    layout: QGridLayout

    def __init__(self, parent=None, merge_func: Optional[Callable] = None):
        """
            Creates all required GUI elements except the actual 
            dataslice visualizations.
        """
        super(DatapointViewer3D, self).__init__()

        self.widget = QWidget()

        self.layout = QGridLayout()
        self.current_scale = 1
        self.merge_func = merge_func



        self.labels: List[QLabel] = []
        self.buttons: List[List[Tuple[QPushButton, SingleDatapointVis]]] = []
        self.icon_size = QSize(64, 64)
        self.orig_icon_size = self.icon_size

        self.widget.setLayout(self.layout)

        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setWidgetResizable(True)
        self.setWidget(self.widget)
        
    def update_btn_icon(self, row: int, col: int):
        """
            Updates a single button icon, used for editing slices.
            
            Args:
                row, col: position of button in visualization grid
        """
        btn, dp_vis = self.buttons[row][col]
        curr_pixmap = dp_vis.base
        # pass only the base image to the manipulator
        edited_img = ImageEditorDialog.get_edited_image(init_img=curr_pixmap)
        dp_vis.update_base(edited_img)
        icon = QIcon(dp_vis.merged)
        btn.setIcon(icon)
        self.update_btn_scale_single(btn, self.icon_size)


    def update_btn_scale_single(self, btn: QPushButton, new_icon_size: QSize):
        """
            Updates scale of QPixmap on a single button
        """
        pixmap = btn.icon().pixmap(self.icon_size)
        pixmap = QPixmap(pixmap.toImage().scaled(new_icon_size))
        icon = QIcon(pixmap)
        btn.setIcon(icon)
        btn.setIconSize(new_icon_size)

    def update_btn_scale(self, scale):
        """
            Updates all buttons with a new scale
        """
        self.current_scale = scale
        new_icon_size = self.orig_icon_size * scale
        btn : QPushButton
        flattened = np.array(self.buttons).flatten()
        for i in range(0, len(flattened), 2):
            btn = flattened[i]
            dp_vis = flattened[i+1]
            if not btn is None:
                self.update_btn_scale_single(btn, new_icon_size)
        self.icon_size = new_icon_size


    def add_btn_to_layout(self, btn: QPushButton, idx: int, row: int, col: int, on_same_row: bool = False):
        """
            Adds a button to the GUI layout (the scrollview) given an index
            for the slice number (will be displayed in a label) as well 
            as row and col for button position in grid.
        """
        new_lbl = QLabel(f"Slice {idx}:")
        self.labels.append(new_lbl)
        if on_same_row:
            
            self.layout.addWidget(new_lbl, row*2, col)
            self.layout.addWidget(btn, row*2, col+1)
        else:
            self.layout.addWidget(new_lbl, row*2, col)
            self.layout.addWidget(btn, row*2+1, col)

    def set_datapoint(self, dp: np.ndarray, labels: np.ndarray):
        """
            Updates the datapoint visualization with new data.

            Args:
                dp: 3d array which will be displayed as slices
                labels: 3d array for desired net output
        """
        assert(len(dp.shape) == 3), "3D Datapoint viewer can only work with 3D numpy arrays"
        assert(dp.shape == labels.shape), "Datapoints and labels need to have the same dimensions"

        self.w, self.h, self.n_slices = dp.shape

        self.cols = 3
        self.rows = int(dp.shape[-1] / self.cols)

        rem = dp.shape[-1] % self.cols

        btn : QPushButton
        flattened = np.array(self.buttons).flatten()
        for i in range(0, len(flattened), 2):
            btn = flattened[i]
            dp_vis = flattened[i+1]
            if not btn is None:
                self.layout.removeWidget(btn)
                btn.deleteLater()
                del dp_vis

        label: QLabel
        for label in self.labels:
            self.layout.removeWidget(label)
            label.deleteLater()

        self.labels = []

        self.buttons = []
        def btn_from_dp(idx: int, row: int, col: int):
            dp_vis = SingleDatapointVis(dp[..., idx], labels[..., idx], merge_func=self.merge_func)
            # create button from generated pixmaps
            icon = QIcon(dp_vis.merged)
            btn = QPushButton()
            btn.setIcon(icon)
            self.update_btn_scale_single(btn, self.icon_size)
            # connect manipulator dialog to button click
            btn.clicked.connect(lambda row=row, col=col: self.update_btn_icon(row, col))      
            return dp_vis, btn  
  
        for i in range(self.rows):
            button_row = []
            for j in range(self.cols):
                idx = i*self.cols+j
                dp_vis, btn = btn_from_dp(idx, i, j)                
                self.layout.addWidget(btn, i, j)
                self.add_btn_to_layout(btn, idx, i, j)
                button_row.append((btn, dp_vis))
            self.buttons.append(button_row)

        last_row = [(None, None)] * (self.cols) #fill with None to have a homogenous shape
        for i in range(rem):
            idx = self.cols*self.rows + i
            dp_vis, btn = btn_from_dp(idx, self.rows, i)
            
            self.add_btn_to_layout(btn, idx, self.rows, i, on_same_row=(self.rows == 0))
            last_row[i] = (btn, dp_vis)

        self.buttons.append(last_row)

        self.widget.setLayout(self.layout)


    def get_current_slices(self) -> np.ndarray:        
        """
            Returns (potentially modified) slices taken from the visualization
            as a 3d array
        """
        slices = np.zeros(shape=(self.w, self.h, self.n_slices))

        flattened = np.array(self.buttons).flatten()
        for i in range(0, len(flattened), 2):
            dp_vis: SingleDatapointVis = flattened[i+1]
            if not dp_vis is None:
                base = dp_vis.slice
                modified_base = pixmap_to_gray_ndarray(dp_vis.base, dp_vis.w, dp_vis.h) / 255
                slices[..., i//2] = modified_base

        return np.array(slices)

class DatapointLoaderWidget(QWidget):
    """
    Load datapoint from image dataset.
    Offer option to manipulate single images.
    Select datapoint as input for agent.
    """

    def __init__(self, dataflow_handler: DataFlowHandler, parent=None, merge_func: Optional[Callable] = None, transforms: Optional[Dict[str, RandomTransform]] = {}):
        """
            Creates GUI for datapoint loader
        """
        super(DatapointLoaderWidget, self).__init__()

        self.layout = QGridLayout()

        self.transforms: Dict[str, RandomTransform] = transforms
        
        self.run_with_variance = False
        self.variance_runs = 5
        self.dataflow_handler = dataflow_handler

        self.label_info1 = QLabel("Dataset path:")
        path = self.dataflow_handler.d_cons["image_path"]
        self.dataset_path_lbl = QLabel(f"{path}")

        self.dataset_load_btn = QPushButton("Load")
        self.dataset_load_btn.clicked.connect(self.load_dataset)

        self.slice_lbl = QLabel(f"#slices:")
        self.slice_lbl.setStyleSheet("color:gray")
        self.point_filename: str = ""
        self.dataset_layout_lbl = QLabel(f"is3D:")
        self.dataset_layout_lbl.setStyleSheet("color:gray")

        self.dataset_select_combo = QComboBox()
        self.dataset_select_combo.setStyleSheet("QComboBox { combobox-popup: 0; }")
        #first row in layout
        # description | dataset path label (from config) | load button #
        self.layout.addWidget(self.label_info1, 0, 0)
        self.layout.addWidget(self.dataset_path_lbl, 0, 1)
        self.layout.addWidget(self.dataset_load_btn, 0, 6)

        #second row in layout
        # description | datapoint select box | spacer #
        self.datapoint_lbl = QLabel("Select datapoint:")
        self.layout.addWidget(self.datapoint_lbl, 1, 0)
        self.layout.addWidget(self.dataset_select_combo, 1, 1, 1, 1)
        self.layout.addWidget(self.dataset_layout_lbl, 1, 5, Qt.AlignmentFlag.AlignLeft)
        self.layout.addWidget(self.slice_lbl, 1, 6)

        #third row in layout
        self.scale_range = [1, 8]
        self.preview_scale = 1
        self.sc_lbl = QLabel(f"Preview scale {self.scale_range}: {self.preview_scale}")
        self.sc_slider = QSlider(Qt.Orientation.Horizontal)
        self.sc_slider.setRange(1, 8)
        self.sc_slider.valueChanged.connect(lambda v: self.update_preview_scale(v))
        self.layout.addWidget(self.sc_lbl, 2, 0)
        self.layout.addWidget(self.sc_slider, 2, 1, 1, 6)

        self.commit_btn = QPushButton("commit")
        self.commit_btn.clicked.connect(self.commit_current_datapoint)

        self.variance_toggle_btn = QCheckBox(text=f"Variance run [{self.variance_runs}]")
        self.variance_toggle_btn.stateChanged.connect(self.update_run_with_variance)
        self.variance_runs_slider = QSlider(Qt.Orientation.Horizontal)
        self.variance_runs_slider.setRange(1, 10)
        self.variance_runs_slider.setValue(self.variance_runs)
        self.variance_runs_slider.valueChanged.connect(lambda v: self.update_variance_runs(v))
        self.layout.addWidget(self.commit_btn, 3, 0)
        self.layout.addWidget(self.variance_toggle_btn, 3, 1)
        self.layout.addWidget(QLabel("#runs: "), 3, 2, 1, 1)
        self.layout.addWidget(self.variance_runs_slider, 3, 3, 1, 4)

        self.dataflow_handler.add_network_handler(self.network_output_callback)
        self.display_area = DatapointViewer3D(merge_func=merge_func)

        self.layout.addWidget(self.display_area, 5, 0, 1, 7)
        
        # fourth row is added. Contains transform selection. 
        self.transform_select_enabled: bool = (len(list(self.transforms.keys()))>0)
        self.transform_dropdown_select = QComboBox()
        if self.transform_select_enabled:
            if "no_transform" in self.transforms:
                raise Exception("You are not allowed to use a transform called no_transform.")
            
        if self.transform_select_enabled:
            
            self.transforms["no_transform"] = None    
            items: List[str] = list(self.transforms.keys())
            self.transform_dropdown_select.addItems(items)
            self.transform_dropdown_select.setCurrentIndex(self.transform_dropdown_select.findData("no_transform"))
            self.transform_dropdown_select.setCurrentText("no_transform")
        else:
            self.transform_dropdown_select.setEnabled(False)
        self.transform_dropdown_select.currentIndexChanged.connect(self._handle_transform_selected_current_index_changed)
        self.layout.addWidget(self.transform_dropdown_select, 4, 0, 1, 2)
          
        #Line Edit for seed selection
        self.seed_line_edit = QLineEdit()
        self.seed_line_edit_validator = QIntValidator()
        self.seed_line_edit_validator.setBottom(0)
        self.seed_line_edit.setValidator(self.seed_line_edit_validator)
        self.seed_line_edit.setText("1234567")
        if not self.transform_select_enabled:
            self.seed_line_edit.setEnabled(False)
        self.seed_line_edit.editingFinished.connect(self._handle_seet_edit_finished)
        self.sc_lbl = QLabel("TorchIO Seed:")
        
        self.layout.addWidget(self.sc_lbl, 4, 3, 1, 1)
        self.layout.addWidget(self.seed_line_edit, 4, 4, 1, 2)
        
        #ApplyButton for applying torchIO transformations!
        
        self.apply_transform_button = QPushButton()
        self.apply_transform_button.setText("Apply")
        self.apply_transform_button.clicked.connect(self._handle_apply_transformation)
        if not self.transform_select_enabled: 
            self.apply_transform_button.setEnabled(False)
        self.layout.addWidget(self.apply_transform_button, 4, 6, 1, 2)
        
        # Seed for TorchIO
        self.torchio_seed: Optional[int] = 1234567
        self.selected_transformation: Optional[RandomTransform] = None
        self.stashed_original_slices: np.ndarray = None
        self.stashed_original_labels: Any = None
        
        self.setLayout(self.layout)

    
    def _handle_transform_selected_current_index_changed(self, index: int): 
        
        transform_text: str =  self.transform_dropdown_select.currentText()
        if transform_text == "no_transform":
            self.selected_transformation = None
        else:
            self.selected_transformation = self.transforms[transform_text]
            
            
    def _handle_seet_edit_finished(self):
        seed_text: str = self.seed_line_edit.text()
        if seed_text is None:
            return
        
        self.torchio_seed = int(seed_text)
        
        
        
        
    
    def _handle_apply_transformation(self):
        # Shape is widh, hight, depth
        if self.stashed_original_slices is None:
            return None
        if self.selected_transformation is None:
            self.display_area.set_datapoint(self.stashed_original_slices, self.stashed_original_labels)
            
        else:
            
            if not isinstance(self.stashed_original_slices, np.ndarray):
                raise Exception("Right now, torchio transformations only work for datasets that provide data points as np.ndarray.")
            widht, height, depth = self.stashed_original_slices.shape
            corr_shape: np.ndarray = np.zeros(shape=(1, widht, height, depth))
            corr_shape[0, ...] = self.stashed_original_slices
            transform_input: tio.Image = tio.Image(tensor=corr_shape, type=tio.INTENSITY)
            use_transform: RandomTransform = self.selected_transformation
            torch.manual_seed(self.torchio_seed)
            transformed_image: tio.Image = use_transform(transform_input)
            transformed_tensor: np.ndarray = transformed_image.numpy()
            self.display_area.set_datapoint(transformed_tensor[0, ...], self.stashed_original_labels)
            
    
    def update_variance_runs(self, v):
        """
            Used for updating the variance runs info label
        """
        self.variance_runs = v
        self.variance_toggle_btn.setText(f"Variance run [{self.variance_runs}]")

    def network_output_callback(self, _):
        """
            Enables commit button after network finished
        """
        self.commit_btn.setEnabled(True)

    def commit_current_datapoint(self):
        """
            Pushes current datapoint into network, dataflow is managed
            by the dataflow handler. Passes the potentially modified network
            input taken from the visualization.
        """
        current_datapoint = self.display_area.get_current_slices()

        print(f"Got altered data: {current_datapoint.shape}")
        if not self.dataflow_handler.does_network_receive_3d():
            #select slice if network is 2D
            slice = SliceSelectionDialog.get_selection(np.arange(self.dataflow_handler.slices_per_datapoint(self.point_filename)))
            slice = int(slice)
            current_image = current_datapoint[..., slice][..., np.newaxis]
        else:
            current_image = current_datapoint
            slice = None
        
        self.commit_btn.setEnabled(False)
        print(f"run with variance: {self.run_with_variance}")
        print(f"network input shape: {current_image.shape}")
        self.dataflow_handler.process_image(image=current_image, filename=self.dataset_select_combo.currentText(), do_variance=self.run_with_variance, num_runs=self.variance_runs, slice_number=slice)     
        
    def update_run_with_variance(self, state):
        self.run_with_variance = self.variance_toggle_btn.isChecked()

    def update_preview_scale(self, v):
        """
            Update preview scale of datapoint visualizations.
        """
        self.preview_scale = v
        self.sc_lbl.setText(f"Preview scale {self.scale_range}: {self.preview_scale}")
        self.display_area.update_btn_scale(v)

    def load_dataset(self):
        """
            Load dataset using the dataflowhandler. Will also init the slice 
            visulization with the first datapoint in the dataset.
        """
        print(f"Loading dataset from {self.dataset_path_lbl.text()}...")

        _, image_names = self.dataflow_handler.get_filenames_in_dataset()
        self.dataset_select_combo.clear()
        self.dataset_select_combo.addItems(image_names)
        self.dataset_select_combo.currentTextChanged.connect(self.load_datapoint)
        
        net_3d = self.dataflow_handler.does_network_receive_3d()
        self.dataset_layout_lbl.setText(f"is3D: {net_3d}")

        first_datapoint = image_names[0]
        self.load_datapoint(first_datapoint)
        n_slices = self.dataflow_handler.slices_per_datapoint(first_datapoint)
        self.slice_lbl.setText(f"#slices: {n_slices}")

    def load_datapoint(self, name: str):
        """
            Updates the slice visualization with a different datapoint.
        """
        print(f"Switching display to: {name}")
        self.point_filename = name
        if self.dataflow_handler.slices_per_datapoint(name) >= 1: #"3D"
            slices, labels = self.dataflow_handler.get_image_for_filename(name)


            if len(labels.shape) != len(slices.shape):
                labels = labels.squeeze()
            self.stashed_original_slices = slices
            self.stashed_original_labels = labels
            print(f"{slices.shape}, {labels.shape}")
            self.display_area.set_datapoint(slices, labels)
        else:
            raise RuntimeError("Datapoint viewer needs at least 1 slice")
        
        n_slices = self.dataflow_handler.slices_per_datapoint(name)
        self.slice_lbl.setText(f"#slices: {n_slices}")


    def set_dataset_path(self, path: str):
        
        self.dataset_path_lbl.setText(path)
