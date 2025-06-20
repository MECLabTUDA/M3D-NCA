from enum import Enum

class JobType(Enum):
    DISPATCH_JOB = 0
    UPDATE_RECURSIVE_ARGUMENTS = 1


class BinningMode(Enum): 
    EQUI_WIDTH = 0
    MULTI_OTSU = 1

class MappingMode(Enum):
    SINGLE = 0
    STEPPED = 1


class InterpolationMode(Enum):
    LINEAR = 0
    TANGENS = 1

class FlowDataChangeType(Enum):
    NETWORK = 0
    ABSDIFF = 1

class InterpolationAction(Enum):
    GENERATE = 0
    SET_MAPPING_MODE = 1 # This is stepped or single
    SET_BIN_NR = 2
    CHANNEL_CHANGED = 3
    HANDLE_NEW_DATA = 4
    SET_BINNING_MODE = 5
    SET_INTERPOLATION_MODE = 6
    SET_Q_VALUE = 7
    CROPPING_CHANGE = 8

class Surface3DPlotOperation(Enum):
    CONTENT_UPDATE = 0
    DISPLAY_FIGURE = 1

