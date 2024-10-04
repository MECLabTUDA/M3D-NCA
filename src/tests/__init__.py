"""
.. note:: Under active development. Anything can change at any time.

#Quickstart:

To train a new model run one of the predefined Jupyter notebooks.

When starting and experiment you first need to define a configuration. Depending on the model you need to provide more or less info.

---

##Config:
The configuration file to define hyperparameters of model.
### Basic configuration
**img_path:** Path to the image files \n
**label_path:** Path to the label files (Name has to be equal to image files) \n
**model_path:** Path where the model will be stored \n
**device:** Device to use use e.g. 'cuda:0' or 'cpu' \n
**unlock_CPU:** Set this to true to use more then one cpu worker \n
### Optimizer (Adam)
**lr:** Learning rate of optimizer \n
**lr_gamma:** Learning gamme of optimizer \n
**betas:** Betas of optimizer \n
### Training
**save_interval:** Interval in which the model will be saved \n
**evaluate_interval:** Internval in which the model will be evaluated on the test set \n
**n_epoch:** Number of training epochs \n
**batch_size:** The batch size during training \n
### Model
**channel_n:** Number of channels in each cell \n
**inference_steps:** Number of inference steps \n
**cell_fire_rate:** Cell fire rate in each step \n
**input_channels:** The number of input channels e.g. grayscale = 1, rgb = 3 \n
**output_channels:** Number of output channels e.g. for single label = 1 \n
### Data
**input_size:** Input data size \n
**data_split:** Data split [train, validation, test] \n


"""