from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget, QLabel, QComboBox, QRadioButton, QGroupBox, QProgressBar, QApplication, QScrollArea, QLineEdit, QCheckBox
from qtpy.QtGui import QIntValidator, QDoubleValidator
from qtpy import QtCore
from qtpy.QtCore import Qt

import napari
import numpy as np
from enum import Enum
from collections import deque, defaultdict
import inspect

import torch

from vispy.util.keys import CONTROL, SHIFT
import copy
import warnings
from tqdm import tqdm
from superqt.utils import qdebounced

import numbers

import urllib.request
from pathlib import Path
import os
from os.path import join

import cv2
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from typing import Optional, Tuple


class AnnotatorMode(Enum):
    NONE = 0
    CLICK = 1
    BBOX = 2
    AUTO = 3


class BboxState(Enum):
    CLICK = 0
    DRAG = 1
    RELEASE = 2


def slicer(arr, slices):
    """
    A dynamic N-dimensional array slicer that returns a tuple that can be used for slicing an N-dimensional array. Works exactly as Python and Numpy slicing, only with different syntax.
    The conventional slicing method has the drawback that one must know the dimensionality of the array beforehand. By contrast, this slicer can be adapted dynamically at runtime.
    :param arr: The array that should be sliced.
    :param slices: A list of slices that obeys the syntax described below.
    :return A tuple that can be used for slicing the array.

    Use case example - Center crop augmentation for an N-dimensional image:
    Imaging you have an N-dimensional array. You don't know what dimension it has. You want to crop each dimension to the center:
        image = ...
        print(image.shape)  # (32, 32) (48, 32, 32) or ...
        indices = comp_indices(image, width=8)
        # crop = image[???]  # Not possible to define a crop with arbitrary dimensions with regular slicing. Instead ...
        if len(image.shape) == 2:
            crop = image[indices[0][0]:indices[0][1], indices[1][0]:indices[1][1]]  # image[10:22, 10:22])
        elif len(image.shape) == 3:
            crop = image[indices[0][0]:indices[0][1], indices[1][0]:indices[1][1], indices[2][0]:indices[2][1]]  # image[18:30, 10:22, 10:22]
        elif ...
    As you see, this cannot be done with regular slicing as the slicing syntax needs to be hardcoded.
    You would need to create an if case for each possible dimension.
    However, this slicer function enables you to do just that dynamically for data with arbitrary dimensions:
        image = ...
        print(image.shape)  # (32, 32) (48, 32, 32) or ...
        indices = comp_indices(image, width=8)
        crop = data[slicer(data, indices)]  # Center crop the data

    Syntax:
    - ... -> None
    - : -> [None]
    - i -> i
    - i:j -> [i, j]
    - i: -> [i, None]
    - :j -> [None, j]

    Syntax examples:
    - sub_arr = arr[5] -> sub_arr = arr[slicer(arr, [5])]
    - sub_arr = arr[...] -> sub_arr = arr[slicer(arr, [None])]
    - sub_arr = arr[..., 5] -> sub_arr = arr[slicer(arr, [None, 5])]
    - sub_arr = arr[5:9] -> sub_arr = arr[slicer(arr, [[5, 9]])]
    - sub_arr = arr[14:30, 12:17] -> sub_arr = arr[slicer(arr, [[14, 30], [12, 17]])]
    - sub_arr = arr[14:30, 12:17, ...] -> sub_arr = arr[slicer(arr, [[14, 30], [12, 17], None])]
    - sub_arr = arr[14:30, 12:17, :] -> sub_arr = arr[slicer(arr, [[14, 30], [12, 17], [None]])]
    - sub_arr = arr[14:30, :-17, :] -> sub_arr = arr[slicer(arr, [[14, 30], [None, -17], [None]])]
    - arr[5] = 7 -> arr[slicer(arr, [5])] = 7
    - arr[15:30, 12:19] = np.zeros((15, 7)) -> arr[slicer(arr, [[15, 30], [12, 19]])] = np.zeros((15, 7))

    This function was inspired by the following stackoverflow thread: https://stackoverflow.com/questions/24398708/slicing-a-numpy-array-along-a-dynamically-specified-axis/37729566#37729566
    """
    slc = [slice(None)] * len(arr.shape)
    axis = 0
    for i in range(len(slices)):
        if slices[i] is None:  # Ellipsis: Take all values from multiple axes: array[..., ???]
            axis = len(arr.shape) - (len(slices) - 1)
        elif isinstance(slices[i], numbers.Number):  # Take single value from a single axis: array[???, i, ???]
            slc[axis] = slices[i]
            axis += 1
        elif len(slices[i]) == 1 and slices[i][0] is None:  # Colon: Take all values from a single axis: array[???, :, ???]
            axis += 1
        else:  # Take all values from the range i to j on a single axis: array[???, i:j, ???]. i or j can also be None
            slc[axis] = slice(slices[i][0], slices[i][1])
            axis += 1
    return tuple(slc)


def normalize(x, source_limits=None, target_limits=None):
    if source_limits is None:
        source_limits = (x.min(), x.max())

    if target_limits is None:
        target_limits = (0, 1)

    if source_limits[0] == source_limits[1] or target_limits[0] == target_limits[1]:
        return x * 0
    else:
        x_std = (x - source_limits[0]) / (source_limits[1] - source_limits[0])
        x_scaled = x_std * (target_limits[1] - target_limits[0]) + target_limits[0]
        return x_scaled
    

class OtsuWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.annotator_mode = AnnotatorMode.NONE

        if not torch.cuda.is_available():
            if not torch.backends.mps.is_available():
                self.device = "cpu"
            else:
                self.device = "mps"
        else:
            self.device = "cuda"

        main_layout = QVBoxLayout()

        self.layer_types = {"image": napari.layers.image.image.Image, "labels": napari.layers.labels.labels.Labels}

        self.btn_load_model = QPushButton("Load model")
        self.btn_load_model.clicked.connect(self._load_model)
        main_layout.addWidget(self.btn_load_model)

        l_image_layer = QLabel("Select input image layer:")
        main_layout.addWidget(l_image_layer)

        self.cb_image_layers = QComboBox()
        self.cb_image_layers.addItems(self.get_layer_names("image"))
        self.cb_image_layers.currentTextChanged.connect(self.on_image_change)
        main_layout.addWidget(self.cb_image_layers)
        
        l_label_layer = QLabel("Select output labels layer:") # click on label icon on left hand side then
        main_layout.addWidget(l_label_layer)

        self.cb_label_layers = QComboBox()
        self.cb_label_layers.addItems(self.get_layer_names("labels"))
        main_layout.addWidget(self.cb_label_layers)

        self.comboboxes = [{"combobox": self.cb_image_layers, "layer_type": "image"}, {"combobox": self.cb_label_layers, "layer_type": "labels"}]

        self.g_annotation = QGroupBox("Annotation mode")
        self.l_annotation = QVBoxLayout()

        self.rb_click = QRadioButton("Bounding Box") # ("Click && Bounding Box")
        self.rb_click.setChecked(True)
        self.rb_click.setToolTip("Positive Click: Middle Mouse Button\n \n"
                                 "Negative Click: Control + Middle Mouse Button \n \n"
                                 "Undo: Control + Z \n \n"
                                 "Select Point: Left Click \n \n"
                                 "Delete Selected Point: Delete")
        self.l_annotation.addWidget(self.rb_click)
        self.rb_click.clicked.connect(self.on_everything_mode_checked)


        self.rb_auto = QRadioButton("Everything")
        self.rb_auto.setToolTip("Creates automatically an instance segmentation \n"
                                            "of the entire image.\n"
                                            "No user interaction possible.")
        self.l_annotation.addWidget(self.rb_auto)
        self.rb_auto.clicked.connect(self.on_everything_mode_checked)

        self.g_annotation.setLayout(self.l_annotation)
        main_layout.addWidget(self.g_annotation)

        self.btn_activate = QPushButton("Activate")
        self.btn_activate.clicked.connect(self._activate) #?
        self.btn_activate.setEnabled(False)
        self.is_active = False
        main_layout.addWidget(self.btn_activate)

        self.btn_mode_switch = QPushButton("Switch to BBox Mode")
        self.btn_mode_switch.clicked.connect(self._switch_mode)
        self.btn_mode_switch.setEnabled(False)
        main_layout.addWidget(self.btn_mode_switch)

        self.check_prev_mask = QCheckBox ('Use Otsu')
        self.check_prev_mask.setEnabled(False)
        self.check_prev_mask.setChecked(True)
        main_layout.addWidget(self.check_prev_mask)

        self.check_auto_inc_bbox= QCheckBox('Auto increment bounding box label')
        self.check_auto_inc_bbox.setEnabled(False)
        self.check_auto_inc_bbox.setChecked(True)
        main_layout.addWidget(self.check_auto_inc_bbox)

        container_widget_info = QWidget()
        container_layout_info = QVBoxLayout(container_widget_info)

        self.g_size = QGroupBox ("Bounding Box Settings") #("Point && Bounding Box Settings")
        self.l_size = QVBoxLayout()

        l_point_size = QLabel("Point Size:")
        self.l_size.addWidget(l_point_size)
        validator = QIntValidator()
        validator.setRange(0, 9999)
        self.le_point_size = QLineEdit()
        self.le_point_size.setText("1")
        self.le_point_size.setValidator(validator)
        self.l_size.addWidget(self.le_point_size)

        l_bbox_edge_width = QLabel("Bounding Box Edge Width:")
        self.l_size.addWidget(l_bbox_edge_width)
        validator = QIntValidator()
        validator.setRange(0, 9999)
        self.le_bbox_edge_width = QLineEdit()
        self.le_bbox_edge_width.setText("1")
        self.le_bbox_edge_width.setValidator(validator)
        self.l_size.addWidget(self.le_bbox_edge_width)
        self.g_size.setLayout(self.l_size)
        container_layout_info.addWidget(self.g_size)

        self.g_info_tooltip = QGroupBox("Tooltip Information")
        self.l_info_tooltip = QVBoxLayout()
        self.label_info_tooltip = QLabel("Every mode shows further information when hovered over.")
        self.label_info_tooltip.setWordWrap(True)
        self.l_info_tooltip.addWidget(self.label_info_tooltip)
        self.g_info_tooltip.setLayout(self.l_info_tooltip)
        container_layout_info.addWidget(self.g_info_tooltip)

        self.g_info_contrast = QGroupBox("Contrast Limits")
        self.l_info_contrast = QVBoxLayout()
        self.label_info_contrast = QLabel("Otsu computes its image embedding based on the current image contrast.\n" 
                                          "Image contrast can be adjusted with the contrast slider of the image layer.") 
        self.label_info_contrast.setWordWrap(True)
        self.l_info_contrast.addWidget(self.label_info_contrast)
        self.g_info_contrast.setLayout(self.l_info_contrast)
        container_layout_info.addWidget(self.g_info_contrast)

        self.g_info_click = QGroupBox("Click Mode")
        self.l_info_click = QVBoxLayout()
        self.label_info_click = QLabel("Positive Click: Middle Mouse Button\n \n"
                                 "Negative Click: Control + Middle Mouse Button\n \n"
                                 "Undo: Control + Z\n \n"
                                 "Select Point: Left Click\n \n"
                                 "Delete Selected Point: Delete\n \n"
                                 "Pick Label: Control + Left Click\n \n"
                                 "Increment Label: M\n \n")
        self.label_info_click.setWordWrap(True)
        self.l_info_click.addWidget(self.label_info_click)
        self.g_info_click.setLayout(self.l_info_click)
        container_layout_info.addWidget(self.g_info_click)

        scroll_area_info = QScrollArea()
        scroll_area_info.setWidget(container_widget_info)
        scroll_area_info.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        main_layout.addWidget(scroll_area_info)

        self.scroll_area_auto = self.init_auto_mode_settings()
        main_layout.addWidget(self.scroll_area_auto)

        self.setLayout(main_layout)

        self.image_name = None
        self.image_layer = None
        self.label_layer = None
        self.label_layer_changes = None
        self.label_color_mapping = None
        self.points_layer = None
        self.points_layer_name = "Ignore this layer1"  # "Ignore this layer <hidden>"
        self.old_points = np.zeros(0)
        self.point_size = 10
        self.le_point_size.setText(str(self.point_size))
        self.bbox_layer = None
        self.bbox_layer_name = "Ignore this layer2"
        self.bbox_edge_width = 10
        self.le_bbox_edge_width.setText(str(self.bbox_edge_width))

        self.init_comboboxes()

        self.otsu_logits = None 

        self.points = defaultdict(list)
        self.point_label = None

        self.bboxes = defaultdict(list)

    def init_auto_mode_settings(self):
        container_widget_auto = QWidget()
        container_layout_auto = QVBoxLayout(container_widget_auto)

        self.g_auto_mode_settings = QGroupBox("Everything Mode Settings")
        self.l_auto_mode_settings = QVBoxLayout()

        l_points_per_side = QLabel("Points per side:")
        self.l_auto_mode_settings.addWidget(l_points_per_side)
        validator = QIntValidator()
        validator.setRange(0, 9999)
        self.le_points_per_side = QLineEdit()
        self.le_points_per_side.setText("32")
        self.le_points_per_side.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_points_per_side)

        l_points_per_batch = QLabel("Points per batch:")
        self.l_auto_mode_settings.addWidget(l_points_per_batch)
        validator = QIntValidator()
        validator.setRange(0, 9999)
        self.le_points_per_batch = QLineEdit()
        self.le_points_per_batch.setText("64")
        self.le_points_per_batch.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_points_per_batch)

        l_pred_iou_thresh = QLabel("Prediction IoU threshold:")
        self.l_auto_mode_settings.addWidget(l_pred_iou_thresh)
        validator = QDoubleValidator()
        validator.setRange(0.0, 1.0)
        validator.setDecimals(5)
        self.le_pred_iou_thresh = QLineEdit()
        self.le_pred_iou_thresh.setText("0.88")
        self.le_pred_iou_thresh.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_pred_iou_thresh)

        l_stability_score_thresh = QLabel("Stability score threshold:")
        self.l_auto_mode_settings.addWidget(l_stability_score_thresh)
        validator = QDoubleValidator()
        validator.setRange(0.0, 1.0)
        validator.setDecimals(5)
        self.le_stability_score_thresh = QLineEdit()
        self.le_stability_score_thresh.setText("0.95")
        self.le_stability_score_thresh.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_stability_score_thresh)

        l_stability_score_offset = QLabel("Stability score offset:")
        self.l_auto_mode_settings.addWidget(l_stability_score_offset)
        validator = QDoubleValidator()
        validator.setRange(0.0, 1.0)
        validator.setDecimals(5)
        self.le_stability_score_offset = QLineEdit()
        self.le_stability_score_offset.setText("1.0")
        self.le_stability_score_offset.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_stability_score_offset)

        l_box_nms_thresh = QLabel("Box NMS threshold:")
        self.l_auto_mode_settings.addWidget(l_box_nms_thresh)
        validator = QDoubleValidator()
        validator.setRange(0.0, 1.0)
        validator.setDecimals(5)
        self.le_box_nms_thresh = QLineEdit()
        self.le_box_nms_thresh.setText("0.7")
        self.le_box_nms_thresh.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_box_nms_thresh)

        l_crop_n_layers = QLabel("Crop N layers")
        self.l_auto_mode_settings.addWidget(l_crop_n_layers)
        validator = QIntValidator()
        validator.setRange(0, 9999)
        self.le_crop_n_layers = QLineEdit()
        self.le_crop_n_layers.setText("0")
        self.le_crop_n_layers.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_crop_n_layers)

        l_crop_nms_thresh = QLabel("Crop NMS threshold:")
        self.l_auto_mode_settings.addWidget(l_crop_nms_thresh)
        validator = QDoubleValidator()
        validator.setRange(0.0, 1.0)
        validator.setDecimals(5)
        self.le_crop_nms_thresh = QLineEdit()
        self.le_crop_nms_thresh.setText("0.7")
        self.le_crop_nms_thresh.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_crop_nms_thresh)

        l_crop_overlap_ratio = QLabel("Crop overlap ratio:")
        self.l_auto_mode_settings.addWidget(l_crop_overlap_ratio)
        validator = QDoubleValidator()
        validator.setRange(0.0, 1.0)
        validator.setDecimals(5)
        self.le_crop_overlap_ratio = QLineEdit()
        self.le_crop_overlap_ratio.setText("0.3413")
        self.le_crop_overlap_ratio.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_crop_overlap_ratio)

        l_crop_n_points_downscale_factor = QLabel("Crop N points downscale factor")
        self.l_auto_mode_settings.addWidget(l_crop_n_points_downscale_factor)
        validator = QIntValidator()
        validator.setRange(0, 9999)
        self.le_crop_n_points_downscale_factor = QLineEdit()
        self.le_crop_n_points_downscale_factor.setText("1")
        self.le_crop_n_points_downscale_factor.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_crop_n_points_downscale_factor)

        l_min_mask_region_area = QLabel("Min mask region area")
        self.l_auto_mode_settings.addWidget(l_min_mask_region_area)
        validator = QIntValidator()
        validator.setRange(0, 9999)
        self.le_min_mask_region_area = QLineEdit()
        self.le_min_mask_region_area.setText("0")
        self.le_min_mask_region_area.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_min_mask_region_area)

        self.g_auto_mode_settings.setLayout(self.l_auto_mode_settings)
        container_layout_auto.addWidget(self.g_auto_mode_settings)

        scroll_area_auto = QScrollArea()
        scroll_area_auto.setWidget(container_widget_auto)
        scroll_area_auto.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area_auto.hide()
        return scroll_area_auto

    def on_everything_mode_checked(self):
        if self.rb_auto.isChecked():
            # self.rb_semantic.setEnabled(False)
            # self.rb_semantic.setChecked(False)
            # self.rb_semantic.setStyleSheet("color: gray")
            # self.rb_instance.setChecked(True)
            self.scroll_area_auto.show()
            self.btn_activate.setText("Run")
        else:
            # self.rb_semantic.setEnabled(True)
            # self.rb_semantic.setStyleSheet("")
            self.scroll_area_auto.hide()
            self.btn_activate.setText("Activate")

    def on_image_change(self):
        pass
            
    def init_comboboxes(self):
        for combobox_dict in self.comboboxes:
            # If current active layer is of the same type of layer that the combobox accepts then set it as selected layer in the combobox.
            active_layer = self.viewer.layers.selection.active
            if combobox_dict["layer_type"] == "all" or isinstance(active_layer, self.layer_types[combobox_dict["layer_type"]]):
                index = combobox_dict["combobox"].findText(active_layer.name, QtCore.Qt.MatchFixedString)
                if index >= 0:
                    combobox_dict["combobox"].setCurrentIndex(index)

        # Inform all comboboxes on layer changes with the viewer.layer_change event
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.viewer.events.layers_change.connect(self._on_layers_changed)


        # Register an event to all existing layers
        for layer_name in self.get_layer_names():
            layer = self.viewer.layers[layer_name]

            @layer.events.name.connect
            def _on_rename(name_event):
                self._on_layers_changed()

        # Register an event to all layers that will be created
        @self.viewer.layers.events.inserted.connect
        def _on_insert(event):
            layer = event.value

            @layer.events.name.connect
            def _on_rename(name_event):
                self._on_layers_changed()

        self._init_comboboxes_callback()

    def _on_layers_changed(self):
        for combobox_dict in self.comboboxes:
            layer = combobox_dict["combobox"].currentText()
            layers = self.get_layer_names(combobox_dict["layer_type"])
            combobox_dict["combobox"].clear()
            combobox_dict["combobox"].addItems(layers)
            index = combobox_dict["combobox"].findText(layer, QtCore.Qt.MatchFixedString)
            if index >= 0:
                combobox_dict["combobox"].setCurrentIndex(index)
        self._on_layers_changed_callback()

    def get_layer_names(self, type="all", exclude_hidden=True):
        layers = self.viewer.layers
        filtered_layers = []
        for layer in layers:
            if (type == "all" or isinstance(layer, self.layer_types[type])) and ((not exclude_hidden) or (exclude_hidden and "<hidden>" not in layer.name)):
                filtered_layers.append(layer.name)
        return filtered_layers

    def _init_comboboxes_callback(self):
        self._check_activate_btn()
        self.on_image_change()

    def _on_layers_changed_callback(self):
        self._check_activate_btn()
        if (self.image_layer is not None and self.image_layer not in self.viewer.layers) or (self.label_layer is not None and self.label_layer not in self.viewer.layers):
            self._deactivate()

    def _check_activate_btn(self):
        if self.cb_image_layers.currentText() != "" and self.cb_label_layers.currentText() != "":
            self.btn_activate.setEnabled(True)
        else:
            self.btn_activate.setEnabled(False)

    def _load_model(self):
        self.otsu_predictor = OtsuPredictor()
        self._check_activate_btn()

    def _activate(self):
        self.btn_activate.setEnabled(False)
        if not self.is_active:
            self.image_name = self.cb_image_layers.currentText()
            self.image_layer = self.viewer.layers[self.cb_image_layers.currentText()]
            self.label_layer = self.viewer.layers[self.cb_label_layers.currentText()]
            self.label_layer_changes = None
            # Fixes shape adjustment by napari
            if self.image_layer.ndim == 3:
                self.image_layer_affine_scale = self.image_layer.affine.scale
                self.image_layer_scale = self.image_layer.scale
                self.image_layer_scale_factor = self.image_layer.scale_factor
                self.label_layer_affine_scale = self.label_layer.affine.scale
                self.label_layer_scale = self.label_layer.scale
                self.label_layer_scale_factor = self.label_layer.scale_factor
                self.image_layer.affine.scale = np.array([1, 1, 1])
                self.image_layer.scale = np.array([1, 1, 1])
                self.image_layer.scale_factor = 1
                self.label_layer.affine.scale = np.array([1, 1, 1])
                self.label_layer.scale = np.array([1, 1, 1])
                self.label_layer.scale_factor = 1
                pos = self.viewer.dims.point
                self.viewer.dims.set_point(0, 0)
                self.viewer.dims.set_point(0, pos[0])
                self.viewer.reset_view()

            if self.image_layer.ndim != 2 and self.image_layer.ndim != 3:
                raise RuntimeError("Only 2D and 3D images are supported.")

            if self.image_layer.ndim == 2:
                self.otsu_logits = None 
            else:
                self.otsu_logits = [None] * self.image_layer.data.shape[0] 

            if self.rb_click.isChecked():
                self.annotator_mode = AnnotatorMode.CLICK
                self.rb_auto.setEnabled(False)
                self.rb_auto.setStyleSheet("color: gray")

            elif self.rb_auto.isChecked():
                self.annotator_mode = AnnotatorMode.AUTO
                self.rb_click.setEnabled(False)
                self.rb_click.setStyleSheet("color: gray")
            else:
                raise RuntimeError("Annotator mode not implemented.")

            if self.annotator_mode != AnnotatorMode.AUTO:
                self.is_active = True
                self.btn_activate.setText("Deactivate")
                self.cb_image_layers.setEnabled(False)
                self.cb_label_layers.setEnabled(False)
                self.btn_mode_switch.setEnabled(True)
                self.check_prev_mask.setEnabled(True)
                self.check_auto_inc_bbox.setEnabled(True)
                self.check_auto_inc_bbox.setChecked(True)
                self.btn_mode_switch.setText("Switch to BBox Mode")
                self.annotator_mode = AnnotatorMode.CLICK
                selected_layer = None
                if self.viewer.layers.selection.active != self.points_layer:
                    selected_layer = self.viewer.layers.selection.active
                self.bbox_layer = self.viewer.add_shapes(name=self.bbox_layer_name)
                if selected_layer is not None:
                    self.viewer.layers.selection.active = selected_layer
                if self.image_layer.ndim == 3:
                    self.check_auto_inc_bbox.setChecked(False)
                    # This "fixes" the problem that the first drawn bbox is not visible.
                    self.update_bbox_layer({}, bbox_tmp=[[self.viewer.dims.current_step[0], 0, 0], [self.viewer.dims.current_step[0], 0, 10], [self.viewer.dims.current_step[0], 10, 10], [self.viewer.dims.current_step[0], 10, 0]])
                    self.update_bbox_layer({}, bbox_tmp=None)
                    self.viewer.dims.set_point(0, 0) #?
                    self.viewer.dims.set_point(0, pos[0]) #?
                self.bbox_layer.editable = False
                self.bbox_first_coords = None

                if self.image_layer.ndim == 2:
                    self.point_size = int(np.min(self.image_layer.data.shape[:2]) / 100)
                    if self.point_size == 0:
                        self.point_size = 1
                    self.bbox_edge_width = 10
                else:
                    self.point_size = 2
                    self.bbox_edge_width = 1
                self.le_point_size.setText(str(self.point_size))
                self.le_bbox_edge_width.setText(str(self.bbox_edge_width))
            
            if self.annotator_mode == AnnotatorMode.CLICK or self.annotator_mode == AnnotatorMode.BBOX:
                self.create_label_color_mapping()

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    self._history_limit = self.label_layer._history_limit
                self._reset_history()

                self.image_layer.events.contrast_limits.connect(qdebounced(self.on_contrast_limits_change, timeout=1000))

                self.set_image()
                self.update_points_layer(None)

                self.viewer.mouse_drag_callbacks.append(self.callback_click)
                self.viewer.keymap['Delete'] = self.on_delete
                self.label_layer.keymap['Control-Z'] = self.on_undo
                self.label_layer.keymap['Control-Shift-Z'] = self.on_redo

            elif self.annotator_mode == AnnotatorMode.AUTO:
                self.otsu_anything_predictor = OtsuPredictor()
                prediction = self.predict_everything()
                self.label_layer.data = prediction
        else:
            self._deactivate()
        self.btn_activate.setEnabled(True)

    def _deactivate(self):
        self.is_active = False
        self.btn_activate.setText("Activate")

        self.cb_image_layers.setEnabled(True)
        self.cb_label_layers.setEnabled(True)
        self.btn_mode_switch.setEnabled(False)
        self.btn_mode_switch.setText("Switch to BBox Mode")
        self.check_prev_mask.setEnabled(False)
        self.check_auto_inc_bbox.setEnabled(False)
        self.annotator_mode = AnnotatorMode.CLICK

        self.remove_all_widget_callbacks(self.viewer)
        if self.label_layer is not None:
            self.remove_all_widget_callbacks(self.label_layer)
        if self.points_layer is not None and self.points_layer in self.viewer.layers:
            self.viewer.layers.remove(self.points_layer)
        if self.bbox_layer is not None and self.bbox_layer in self.viewer.layers:
            self.viewer.layers.remove(self.bbox_layer)
        self.image_name = None
        self.image_layer = None
        self.label_layer = None
        self.label_layer_changes = None
        self.points_layer = None
        self.bbox_layer = None
        self.bbox_first_coords = None
        self.annotator_mode = AnnotatorMode.NONE
        self.points = defaultdict(list)
        self.bboxes = defaultdict(list)
        self.point_label = None
        self.otsu_logits = None
        self.rb_click.setEnabled(True)
        self.rb_auto.setEnabled(True)
        self.rb_click.setStyleSheet("")
        self.rb_auto.setStyleSheet("")
        self._reset_history()

    def _switch_mode(self):
        if self.annotator_mode == AnnotatorMode.CLICK:
            self.btn_mode_switch.setText("Switch to Click Mode")
            self.annotator_mode = AnnotatorMode.BBOX

        else:
            self.btn_mode_switch.setText("Switch to BBox Mode")
            self.annotator_mode = AnnotatorMode.CLICK

    def create_label_color_mapping(self, num_labels=1000):
        if self.label_layer is not None:
            self.label_color_mapping = {"label_mapping": {}, "color_mapping": {}}
            for label in range(num_labels):
                color = self.label_layer.get_color(label)
                self.label_color_mapping["label_mapping"][label] = color
                self.label_color_mapping["color_mapping"][str(color)] = label

    def callback_click(self, layer, event):
        data_coordinates = self.image_layer.world_to_data(event.position)
        coords = np.round(data_coordinates).astype(int)
        if self.annotator_mode == AnnotatorMode.CLICK:
            if (not CONTROL in event.modifiers) and event.button == 1:  # and event.button == 3:  # Positive middle click
                self.do_point_click(coords, 1)
                yield
            elif CONTROL in event.modifiers and event.button == 1: # and event.button == 3:  # Negative middle click
                self.do_point_click(coords, 0)
                yield
            elif (not CONTROL in event.modifiers) and event.button == 1 and self.points_layer is not None and len(self.points_layer.data) > 0:
                # Find the closest point to the mouse click
                distances = np.linalg.norm(self.points_layer.data - coords, axis=1)
                closest_point_idx = np.argmin(distances)
                closest_point_distance = distances[closest_point_idx]

                # Select the closest point if it's within self.point_size pixels of the click
                if closest_point_distance <= self.point_size:
                    self.points_layer.selected_data = {closest_point_idx}
                else:
                    self.points_layer.selected_data = set()
                yield
            elif (CONTROL in event.modifiers) and event.button == 1:
                picked_label = self.label_layer.data[slicer(self.label_layer.data, coords)]
                self.label_layer.selected_label = picked_label
                yield
        elif self.annotator_mode == AnnotatorMode.BBOX:
            if ((CONTROL in event.modifiers) or (SHIFT in event.modifiers)) and event.button == 1: # if (not CONTROL in event.modifiers) and event.button == 3:  # Positive middle click
                self.do_bbox_click(coords, BboxState.CLICK)
                yield
                while event.type == 'mouse_move':
                    data_coordinates = self.image_layer.world_to_data(event.position)
                    coords = np.round(data_coordinates).astype(int)
                    self.do_bbox_click(coords, BboxState.DRAG)
                    yield
                data_coordinates = self.image_layer.world_to_data(event.position)
                coords = np.round(data_coordinates).astype(int)
                self.do_bbox_click(coords, BboxState.RELEASE)

    def on_delete(self, layer):
        selected_points = list(self.points_layer.selected_data)
        if len(selected_points) > 0:
            self.points_layer.data = np.delete(self.points_layer.data, selected_points[0], axis=0)
            self._save_history({"mode": AnnotatorMode.CLICK, "points": copy.deepcopy(self.points), "bboxes": copy.deepcopy(self.bboxes), "logits": self.otsu_logits, "point_label": self.point_label})
            deleted_point, _ = self.find_changed_point(self.old_points, self.points_layer.data)
            label = self.find_point_label(deleted_point)
            index_to_remove = np.where((self.points[label] == deleted_point).all(axis=1))[0]
            self.points[label] = np.delete(self.points[label], index_to_remove, axis=0).tolist()
            if len(self.points[label]) == 0:
                del self.points[label]
            self.point_label = label
            if self.image_layer.ndim == 2:
                self.otsu_logits = None 
            elif self.image_layer.ndim == 3:
                self.otsu_logits[deleted_point[0]] = None 
            else:
                raise RuntimeError("Point deletion not implemented for this dimensionality.")
            self.predict_click(self.points, self.point_label, deleted_point, label)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self.label_layer._save_history((self.label_layer_changes["indices"], self.label_layer_changes["old_values"], self.label_layer_changes["new_values"]))

    def on_undo(self, layer):
        """Undo the last paint or fill action since the view slice has changed."""
        self.undo()
        self.label_layer.undo()
        self.label_layer.data = self.label_layer.data

    def on_redo(self, layer):
        """Redo any previously undone actions."""
        self.redo()
        self.label_layer.redo()
        self.label_layer.data = self.label_layer.data

    def on_contrast_limits_change(self):
        self.set_image()
    
    def set_image(self):

        if self.image_layer.ndim == 2:
            image = np.asarray(self.image_layer.data)
            if image.dtype != np.uint8:
                contrast_limits = self.image_layer.contrast_limits
                image = normalize(image, source_limits=contrast_limits, target_limits=(0, 255)).astype(np.uint8)
            self.otsu_predictor.set_image(image) 

        elif self.image_layer.ndim == 3:
            l_creating_features= QLabel("Segmenting:") 
            self.layout().addWidget(l_creating_features)
            progress_bar = QProgressBar(self)
            progress_bar.setMaximum(self.image_layer.data.shape[0])
            progress_bar.setValue(0)
            self.layout().addWidget(progress_bar)
            self.otsu_result = []
            for index in range(self.image_layer.data.shape[0]):        
                image_slice = np.asarray(self.image_layer.data[index, ...])
                image_slice = normalize(image_slice, source_limits=contrast_limits, target_limits=(0, 255)).astype(np.uint8)
                self.otsu_predictor.set_image(image)
                progress_bar.setValue(index+1)
                QApplication.processEvents()
                progress_bar.deleteLater()
                l_creating_features.deleteLater()
        else:
            raise RuntimeError("Only 2D and 3D images are supported.")

    def do_point_click(self, coords, is_positive):
        # Check if there is already a point at these coordinates
        for label, points in self.points.items():
            if np.any((coords == points).all(1)):
                warnings.warn("There is already a point in this location. This click will be ignored.")
                return

        self._save_history({"mode": AnnotatorMode.CLICK, "points": copy.deepcopy(self.points), "bboxes": copy.deepcopy(self.bboxes), "logits": self.otsu_logits, "point_label": self.point_label})

        self.point_label = self.label_layer.selected_label
        if not is_positive:
            self.point_label = 0

        self.points[self.point_label].append(coords)

        self.predict_click(self.points, self.point_label, coords, self.point_label)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.label_layer._save_history((self.label_layer_changes["indices"], self.label_layer_changes["old_values"], self.label_layer_changes["new_values"]))

    def predict_click(self, points, point_label, current_point, current_label):
        self.update_points_layer(points)

        if points:
            points_flattened = []
            labels_flattended = []
            for label, label_points in points.items():
                points_flattened.extend(label_points)
                label = int(label == point_label)
                labels = [label] * len(label_points)
                labels_flattended.extend(labels)

            x_coord = current_point[0]
            prediction = self.predict_otsu(points=copy.deepcopy(points_flattened), labels=copy.deepcopy(labels_flattended), bbox=None, x_coord=copy.deepcopy(current_point[0]))
            if self.image_layer.ndim == 2:
                x_coord = slice(None, None)
        else:
            prediction = np.zeros_like(self.label_layer.data)
            x_coord = slice(None, None)

        if prediction is not None:
            label_layer = np.asarray(self.label_layer.data)
            changed_indices = np.where(prediction == 255) 
            index_labels_old = label_layer[changed_indices]
            label_layer[x_coord][label_layer[x_coord] == point_label] = 0

            label_layer[(prediction == 255) & (label_layer == 0)] = point_label 
            index_labels_new = label_layer[changed_indices]
            self.label_layer_changes = {"indices": changed_indices, "old_values": index_labels_old, "new_values": index_labels_new}
            self.label_layer.data = label_layer
            self.old_points = copy.deepcopy(self.points_layer.data)

    def do_bbox_click(self, coords, bbox_state):
        if bbox_state == BboxState.CLICK:
            if not (self.image_layer.ndim == 2 or self.image_layer.ndim == 3):
                raise RuntimeError("Only 2D and 3D images are supported.")
            self.bbox_first_coords = coords
        elif bbox_state == BboxState.DRAG:
            if self.image_layer.ndim == 2:
                bbox_tmp = np.asarray([self.bbox_first_coords, (self.bbox_first_coords[0], coords[1]), coords, (coords[0], self.bbox_first_coords[1])])
            elif self.image_layer.ndim == 3:
                bbox_tmp = np.asarray([self.bbox_first_coords, (self.bbox_first_coords[0], self.bbox_first_coords[1], coords[2]), coords, (self.bbox_first_coords[0], coords[1], self.bbox_first_coords[2])])
            else:
                raise RuntimeError("Only 2D and 3D images are supported.")
            bbox_tmp = np.rint(bbox_tmp).astype(np.int32)
            self.update_bbox_layer(self.bboxes, bbox_tmp=bbox_tmp)
        else:
            self._save_history({"mode": AnnotatorMode.BBOX, "points": copy.deepcopy(self.points), "bboxes": copy.deepcopy(self.bboxes), "logits": self.otsu_logits, "point_label": self.point_label})
            if self.image_layer.ndim == 2:
                x_coord = slice(None, None)
                bbox_final = np.asarray([self.bbox_first_coords, (self.bbox_first_coords[0], coords[1]), coords, (coords[0], self.bbox_first_coords[1])])
            elif self.image_layer.ndim == 3:
                x_coord = self.bbox_first_coords[0]
                bbox_final = np.asarray([self.bbox_first_coords, (self.bbox_first_coords[0], self.bbox_first_coords[1], coords[2]), coords, (self.bbox_first_coords[0], coords[1], self.bbox_first_coords[2])])
            else:
                raise RuntimeError("Only 2D and 3D images are supported.")

            new_label = self.label_layer.selected_label
            if self.check_auto_inc_bbox.isChecked():
                new_label = np.max(self.label_layer.data) + 1
                self.label_layer.selected_label = new_label

            bbox_final = np.rint(bbox_final).astype(np.int32)
            self.bboxes[new_label].append(bbox_final)
            self.update_bbox_layer(self.bboxes)

            prediction = self.predict_otsu(points=None, labels=None, bbox=copy.deepcopy(bbox_final), x_coord=x_coord)

            label_layer = np.asarray(self.label_layer.data)
            changed_indices = np.where(prediction == 255) # np.where(prediction == 1)
            index_labels_old = label_layer[changed_indices]
            label_layer[(prediction == 255) & (label_layer == 0)] = new_label # label_layer[(prediction == 1) & (label_layer == 0)] = new_label
            index_labels_new = label_layer[changed_indices]
            self.label_layer_changes = {"indices": changed_indices, "old_values": index_labels_old, "new_values": index_labels_new}
            self.label_layer.data = label_layer
            self.old_points = copy.deepcopy(self.points_layer.data)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self.label_layer._save_history((self.label_layer_changes["indices"], self.label_layer_changes["old_values"], self.label_layer_changes["new_values"]))

            if self.check_auto_inc_bbox.isChecked():
                # Update the label here too. This way the label stays incremented when switching to click mode
                new_label = np.max(self.label_layer.data) + 1
                self.label_layer.selected_label = new_label

    def predict_otsu(self, points, labels, bbox, x_coord=None):
        if self.image_layer.ndim == 2:
            if points is not None:
                points = np.flip(points, axis=-1)
                labels = np.asarray(labels)
            if bbox is not None:
                top_left_coord, bottom_right_coord = self.find_corners(bbox)
                bbox = [np.flip(top_left_coord), np.flip(bottom_right_coord)]
                bbox = np.asarray(bbox).flatten()
            logits = self.otsu_logits
            if not self.check_prev_mask.isChecked():
                logits = None
            prediction = self.otsu_predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=bbox,
                mask_input=logits,
                multimask_output=False,
            )
        elif self.image_layer.ndim == 3:
            prediction = np.zeros_like(self.label_layer.data)
            if points is not None:
                points = np.asarray(points)
                x_coords = np.unique(points[:, 0])
                groups = {x_coord: list(points[points[:, 0] == x_coord]) for x_coord in x_coords}  # Group points if they are on the same image slice
                group_points = groups[x_coord]
                group_labels = [labels[np.argwhere(np.all(points == point, axis=1)).flatten()[0]] for point in group_points]
                group_points = [point[1:] for point in group_points]
                points = np.flip(group_points, axis=-1)
                labels = np.asarray(group_labels)
            if bbox is not None:
                bbox = bbox[:, 1:]
                top_left_coord, bottom_right_coord = self.find_corners(bbox)
                bbox = [np.flip(top_left_coord), np.flip(bottom_right_coord)]
                bbox = np.asarray(bbox).flatten()
            logits = self.otsu_logits[x_coord] 
            if not self.check_prev_mask.isChecked():
                logits = None
            prediction_slice = self.otsu_predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=bbox,
                mask_input=logits,
                multimask_output=False,
            )
            prediction[x_coord, :, :] = prediction_slice 
        else:
            raise RuntimeError("Only 2D and 3D images are supported.")
        return prediction

    def predict_everything(self):
            if self.image_layer.ndim == 2:
                image = np.asarray(self.image_layer.data)
                if image.dtype != np.uint8:
                    contrast_limits = self.image_layer.contrast_limits
                    image = normalize(image, source_limits=contrast_limits, target_limits=(0, 255)).astype(np.uint8)
                self.otsu_anything_predictor.set_image(image) 
                prediction = self.otsu_anything_predictor.predict()
            elif self.image_layer.ndim == 3:
                l_creating_features = QLabel("Predicting everything:")
                self.layout().addWidget(l_creating_features)
                progress_bar = QProgressBar(self)
                progress_bar.setMaximum(self.image_layer.data.shape[0])
                progress_bar.setValue(0)
                self.layout().addWidget(progress_bar)
                prediction = []
                for index in tqdm(range(self.image_layer.data.shape[0]), desc="Predicting everything"):
                    image_slice = np.asarray(self.image_layer.data[index, ...])
                    contrast_limits = self.image_layer.contrast_limits
                    image_slice = normalize(image_slice, source_limits=contrast_limits, target_limits=(0, 255)).astype(np.uint8)
                    self.otsu_anything_predictor.set_image(image_slice)
                    records_slice = self.otsu_anything_predictor.predict()
                    masks_slice = np.asarray([record["segmentation"] for record in records_slice])
                    prediction_slice = np.argmax(masks_slice, axis=0)
                    prediction.append(prediction_slice)
                    progress_bar.setValue(index+1)
                    QApplication.processEvents()
                    progress_bar.deleteLater()
                    l_creating_features.deleteLater()
                prediction = np.stack(prediction, axis=0)
                prediction = self.merge_classes_over_slices(prediction)
            else:
                raise RuntimeError("Only 2D and 3D images are supported.")
            return prediction

    def merge_classes_over_slices(self, prediction, threshold=0.5):  # Currently only computes overlap from next_slice to current_slice but not vice versa
        for i in range(prediction.shape[0] - 1):
            current_slice = prediction[i]
            next_slice = prediction[i+1]
            next_labels, next_label_counts = np.unique(next_slice, return_counts=True)
            next_label_counts = next_label_counts[next_labels != 0]
            next_labels = next_labels[next_labels != 0]
            new_next_slice = np.zeros_like(next_slice)
            if len(next_labels) > 0:
                for next_label, next_label_count in zip(next_labels, next_label_counts):
                    current_roi_labels = current_slice[next_slice == next_label]
                    current_roi_labels, current_roi_label_counts = np.unique(current_roi_labels, return_counts=True)
                    current_roi_label_counts = current_roi_label_counts[current_roi_labels != 0]
                    current_roi_labels = current_roi_labels[current_roi_labels != 0]
                    if len(current_roi_labels) > 0:
                        current_max_count = np.max(current_roi_label_counts)
                        current_max_count_label = current_roi_labels[np.argmax(current_roi_label_counts)]
                        overlap = current_max_count / next_label_count
                        if overlap >= threshold:
                            new_next_slice[next_slice == next_label] = current_max_count_label
                        else:
                            new_next_slice[next_slice == next_label] = next_label
                    else:
                        new_next_slice[next_slice == next_label] = next_label
                prediction[i+1] = new_next_slice
        return prediction
    
    def update_points_layer(self, points):
        self.point_size = int(self.le_point_size.text())
        selected_layer = None
        if self.viewer.layers.selection.active != self.points_layer:
            selected_layer = self.viewer.layers.selection.active
        if self.points_layer is not None:
            self.viewer.layers.remove(self.points_layer)

        points_flattened = []
        colors_flattended = []
        if points is not None:
            for label, label_points in points.items():
                points_flattened.extend(label_points)
                color = self.label_color_mapping["label_mapping"][label]
                colors = [color] * len(label_points)
                colors_flattended.extend(colors)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.points_layer = self.viewer.add_points(name=self.points_layer_name, data=np.asarray(points_flattened), face_color=colors_flattended, edge_color="white", size=self.point_size)
        self.points_layer.editable = False

        if selected_layer is not None:
            self.viewer.layers.selection.active = selected_layer
        self.points_layer.refresh()
    
    def update_bbox_layer(self, bboxes, bbox_tmp=None):
        self.bbox_edge_width = int(self.le_bbox_edge_width.text())
        bboxes_flattened = []
        edge_colors = []
        for _, bbox in bboxes.items():
            bboxes_flattened.extend(bbox)
            edge_colors.extend(['skyblue'] * len(bbox))
        if bbox_tmp is not None:
            bboxes_flattened.append(bbox_tmp)
            edge_colors.append('steelblue')
        self.bbox_layer.data = bboxes_flattened
        self.bbox_layer.edge_width = [self.bbox_edge_width] * len(bboxes_flattened)
        self.bbox_layer.edge_color = edge_colors
        self.bbox_layer.face_color = [(0, 0, 0, 0)] * len(bboxes_flattened)

    def find_changed_point(self, old_points, new_points):
        if len(new_points) == 0:
            old_point = old_points
        else:
            old_point = np.array([x for x in old_points if not np.any((x == new_points).all(1))])
        if len(old_points) == 0:
            new_point = new_points
        else:
            new_point = np.array([x for x in new_points if not np.any((x == old_points).all(1))])

        if len(old_point) == 0:
            deleted_point = None
        else:
            deleted_point = old_point[0]

        if len(new_point) == 0:
            new_point = None
        else:
            new_point = new_point[0]

        return deleted_point, new_point

    def find_point_label(self, point):
        for label, label_points in self.points.items():
            if np.in1d(point, label_points).all(axis=0):
                return label
        raise RuntimeError("Could not identify label.")

    def remove_all_widget_callbacks(self, layer):
        callback_types = ['mouse_double_click_callbacks', 'mouse_drag_callbacks', 'mouse_move_callbacks',
                          'mouse_wheel_callbacks', 'keymap']
        for callback_type in callback_types:
            callbacks = getattr(layer, callback_type)
            if isinstance(callbacks, list):
                for callback in callbacks:
                    if inspect.ismethod(callback) and callback.__self__ == self:
                        callbacks.remove(callback)
            elif isinstance(callbacks, dict):
                for key in list(callbacks.keys()):
                    if inspect.ismethod(callbacks[key]) and callbacks[key].__self__ == self:
                        del callbacks[key]
            else:
                raise RuntimeError("Could not determine callbacks type.")

    def find_corners(self, coords):
        # convert the coordinates to numpy arrays
        coords = np.array(coords)

        # find the indices of the leftmost, rightmost, topmost, and bottommost coordinates
        left_idx = np.min(coords[:, 0])
        right_idx = np.max(coords[:, 0])
        top_idx = np.min(coords[:, 1])
        bottom_idx = np.max(coords[:, 1])

        top_left_coord = [left_idx, top_idx]
        bottom_right_coord = [right_idx, bottom_idx]

        return top_left_coord, bottom_right_coord

    def _reset_history(self, event=None):
        self._undo_history = deque()
        self._redo_history = deque()

    def _save_history(self, history_item):
        """Save a history "atom" to the undo history.

        A history "atom" is a single change operation to the array. A history
        *item* is a collection of atoms that were applied together to make a
        single change. For example, when dragging and painting, at each mouse
        callback we create a history "atom", but we save all those atoms in
        a single history item, since we would want to undo one drag in one
        undo operation.

        Parameters
        ----------
        history_item : 2-tuple of region prop dicts
        """
        self._redo_history = deque()
        self._undo_history.append(history_item)

    def _load_history(self, before, after, undoing=True):
        """Load a history item and apply it to the array.

        Parameters
        ----------
        before : list of history items
            The list of elements from which we want to load.
        after : list of history items
            The list of element to which to append the loaded element. In the
            case of an undo operation, this is the redo queue, and vice versa.
        undoing : bool
            Whether we are undoing (default) or redoing. In the case of
            redoing, we apply the "after change" element of a history element
            (the third element of the history "atom").

        See Also
        --------
        Labels._save_history
        """
        if len(before) == 0:
            return

        history_item = before.pop()
        after.append(history_item)

        self.points = history_item["points"]
        self.point_label = history_item["point_label"]
        self.otsu_logits = history_item["logits"] 
        self.bboxes = history_item["bboxes"]
        self.update_points_layer(self.points)
        self.update_bbox_layer(self.bboxes)

    def undo(self):
        self._load_history(
            self._undo_history, self._redo_history, undoing=True
        )

    def redo(self):
        self._load_history(
            self._redo_history, self._undo_history, undoing=False
        )
        raise RuntimeError("Redo currently not supported.")

    def download_with_progress(self, url, output_file):
        # Open the URL and get the content length
        req = urllib.request.urlopen(url)
        content_length = int(req.headers.get('Content-Length'))

        l_creating_features = QLabel("Downloading model:")
        self.layout().addWidget(l_creating_features)
        progress_bar = QProgressBar(self)
        progress_bar.setMaximum(int(content_length / 1024))
        progress_bar.setValue(0)
        self.layout().addWidget(progress_bar)

        # Set up the progress bar
        progress_bar_tqdm = tqdm(total=content_length, unit='B', unit_scale=True, desc="Downloading model")

        # Download the file and update the progress bar
        with open(output_file, 'wb') as f:
            downloaded_bytes = 0
            while True:
                buffer = req.read(8192)
                if not buffer:
                    break
                downloaded_bytes += len(buffer)
                f.write(buffer)
                progress_bar_tqdm.update(len(buffer))
                progress_bar.setValue(int(downloaded_bytes / 1024))
                QApplication.processEvents()
                progress_bar.deleteLater()
                l_creating_features.deleteLater()

        # Close the progress bar and the URL
        progress_bar_tqdm.close()
        req.close()


class OtsuPredictor:
    def __init__(self) -> None:
        """
        Uses Otsu thresholding to calculate a binary mask for an image.
        """
        super().__init__()

    def set_image(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
        """
        Prepares the image for Otsu thresholding.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects a
            grayscale or RGB image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in ["RGB", "BGR"], "image_format must be in ['RGB', 'BGR']."
        if image_format == "BGR":
            image = image[..., ::-1]  # Convert BGR to RGB if necessary

        # Convert to grayscale if the image is RGB
        if image.ndim == 3:
            image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])  # Grayscale conversion

        self.image = image
        self.original_size = image.shape[:2]
        self.is_image_set = True

    def create_bounding_box(self, point: Tuple[int, int], box_size: int = 20) -> Tuple[int, int, int, int]:
        """
        Creates a bounding box around a given point with a fixed size.

        Arguments:
        point (tuple): The (x, y) coordinates of the center point.
        box_size (int): The size of the bounding box (defaults to 20 pixels).

        Returns:
        tuple: Bounding box coordinates as (x_min, y_min, x_max, y_max).
        """
        x, y = point
        half_size = box_size // 2
        x_min = max(x - half_size, 0)
        y_min = max(y - half_size, 0)
        x_max = min(x + half_size, self.original_size[1])
        y_max = min(y + half_size, self.original_size[0])
        return x_min, y_min, x_max, y_max
    
    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies Otsu thresholding to create a binary mask.

        Arguments (maintained for compatibility):
          point_coords (np.ndarray or None): Ignored for Otsu thresholding.
          point_labels (np.ndarray or None): Ignored for Otsu thresholding.
          box (np.ndarray or None): Ignored for Otsu thresholding.
          mask_input (np.ndarray): Ignored for Otsu thresholding.
          multimask_output (bool): Ignored for Otsu thresholding.
          return_logits (bool): If true, returns the raw thresholded image instead of a binary mask.

        Returns:
          (np.ndarray): The output binary mask or the raw thresholded image.
          (np.ndarray): An array of quality scores (not applicable, so returns a placeholder).
          (np.ndarray): Low-resolution logits or a low-res version of the mask (not applicable, so returns a placeholder).
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            full_mask = np.zeros_like(self.image, dtype=np.uint8)
            for point in point_coords:
                bbox = self.create_bounding_box(point, box_size=20)
                x_min, y_min, x_max, y_max = bbox

                # Crop to bounding box
                cropped_image = self.image[y_min:y_max, x_min:x_max]

                # Apply Otsu thresholding to the cropped region
                otsu_thresh = threshold_otsu(cropped_image)
                binary_mask = (cropped_image > otsu_thresh).astype(np.uint8) * 255

                # Place the binary mask back into the full-size mask
                full_mask[y_min:y_max, x_min:x_max] = binary_mask

        if box is not None:
            x_min, y_min, x_max, y_max = box
            cropped_image = self.image[y_min:y_max, x_min:x_max]  # Crop to bounding box
        
        else:
            cropped_image = self.image

        # Apply Otsu thresholding
        otsu_thresh = threshold_otsu(cropped_image)
        binary_mask = cropped_image > otsu_thresh

        if box is not None:
            full_mask = np.zeros_like(self.image, dtype=np.uint8)
            full_mask[y_min:y_max, x_min:x_max] = binary_mask.astype(np.uint8) * 255
        else:
            full_mask = binary_mask.astype(np.uint8) * 255

        output_mask = cropped_image - otsu_thresh if return_logits else full_mask

        
        return output_mask










