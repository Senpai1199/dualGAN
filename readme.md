# dualGAN
PyTorch implementation of the paper: [DualGAN: Unsupervised Dual Learning for Image-to-Image Translation](https://arxiv.org/abs/1704.02510) <br/>
[Link to Complete Dataset](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/)

## Folders in the repository:

1. **datasets** : Contains 2 datasets namely "facades.zip" and "day-night.zip" which were used to train the model.
2. **results** : Contains the results obtained from our implementation for each of the above dataset. <br />
  2.1 Receptive fields used for the day-night dataset: **70x70, 16x16, 1x1** <br />
  2.2 Receptive fields used for the facades dataset: **70x70**

## Individual files:

1. **training_parameters.json** : Contains the parameters for training the model such as no. of epochs, dataset_name, batch size, directory paths.
2. **tf1_dualgan.py** : File containing the core dual GAN architecture and the whole logic for generators, discriminators, training, testing. 
3. **helper.py** : Contains the Dataset class with helper functions such as loading and fetching images.
4. **calc_receptive_field.py** : Contains logic for calculating patch sizes for different receptive field sizes.
