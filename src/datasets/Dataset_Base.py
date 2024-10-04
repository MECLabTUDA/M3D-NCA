from torch.utils.data import Dataset
from src.datasets.Data_Instance import Data_Container
from src.utils.Experiment import Experiment

class Dataset_Base(Dataset):
    r"""Base class for any dataset within this project
        .. WARNING:: Not to be used directly!
    """
    def __init__(self, resize: bool = True) -> None: 
        self.resize = resize
        self.count = 42
        self.data = Data_Container()

    def set_size(self, size: tuple) -> None:
        r"""Set size of images
            #Args
                size (int, int): Size of images
        """
        self.size = tuple(size)

    def set_experiment(self, experiment: Experiment) -> None:
        r"""Set experiment
            #Args
                experiment: The experiment class
        """
        self.exp = experiment

    def setPaths(self, images_path: str, images_list: str, labels_path: str, labels_list: str) -> None:
        r"""Set the important image paths
            #Args
                images_path (String): The path to the images
                images_list ([String]): A list of the names of all images
                labels_path (String): The path to the labels
                labels_list ([String]): A list of the names of all labels
            .. TODO:: Refactor
        """
        self.images_path = images_path
        self.images_list = images_list
        self.labels_path = labels_path
        self.labels_list = labels_list
        self.length = len(self.images_list)

    def getImagePaths(self) -> list:
        r"""Get a list of all images in dataset
            #Returns:
                list ([String]): List of images
        """
        return self.images_list

    def __len__(self) -> int:
        r"""Get number of items in dataset"""
        return self.length

    def getItemByName(self, name: str) -> tuple:
        r"""Get item by its name
            #Args
                name (String)
            #Returns:
                item (tensor): The image tensor
        """
        idx = self.images_list.index(name)
        return self.__getitem__(idx)

    def getFilesInPath(self, path: str):
        raise NotImplementedError("Subclasses should implement this!")

    def __getitem__(self, idx: str):
        raise NotImplementedError("Subclasses should implement this!")

    def setState(self, state: str) -> None:
        self.state = state