# VRGAN
Code used for the paper "Adversarial regression training for visualizing the progression of chronic obstructive pulmonary disease with chest x-rays" (arxiv link placeholder), accepted for MICCAI 2019. This code implementation is done in PyTorch and was partially inspired by [Orobix VAGAN](https://github.com/orobix/Visual-Feature-Attribution-Using-Wasserstein-GANs-Pytorch). 

## Setup

The first run of the code will automatically generate the synthetic dataset. We are going to try to provide a model pre-trained on chest x-rays in the future, depending on release approval. The chest x-ray dataset used to train this model is not publically available, and there are no plans of making it available in the future.

To install all the needed libraries, you can use the requirements.txt file. Use `pip install -r requirements.txt` if you want to install them with pip.

## Usage

You can train the model on the synthetic dataset using:

`python train.py --experiment=my_toy_training`

To run scoring on the test set, run:

`python train.py --nepochs=1 --split_validation=test --skip_train=true --load_checkpoint_g=<path to a checkpoint to load> --experiment=my_toy_testing`

You can run `python train.py --help` to see all available options for modifying the training script.

## Outputs
All the outputs of the model are saved in the runs folder, inside a folder for the specific experiment you are running (<experiment name>\_<timestamp>). These are the files that are saved:
  * tensorboard/events.out.tfevents.<...>: tensorboard file for following the training losses and validation score in real-time and for checking their evolution through the epochs.
  * real_samples: a fixed batch of validation examples for which outputs will be printed
  * delta_x_gt.png: ground truth for the delta_x (disease effect map, or difference map) of the fixed set of validation examples
  * real_samples_desired.txt: the desired PFT values for each of the fixed validation images
  * real_samples_gt.txt: the original PFT values for each of the fixed validation images
  * delta_x_samples_<epoch>.png: delta_x output of the generator at the end of that epoch for the fixed set of examples
  * xprime_samples_<epoch>: modified input image (sum of original image and delta_x) at the end of that epoch for the fixed set of examples
  * generator_state_dict_<epoch>: checkpoint for the generator model
  * regressor_state_dict_<epoch>: checkpoint for the regressor model
  * log.txt: a way to check the configurations used for that run and to check the losses and scores of the model in text format, without loading tensorboard.

## Results
The model gets a normalized cross-correlation with the ground-truth delta around 0.85 in the validation set in about 10 epochs of training. After choosing the best validation epoch, the same model should be able to get the same score for the test set.

These are the kinds of results you can expect:

Input image (x) |  Desired change (Δx ground truth)  |  Produced change (Δx)  | Modified image (x')
--- | --- | --- | ---
![](https://github.com/ricbl/vrgan/images/x.png)  |  ![](https://github.com/ricbl/vrgan/images/delta_x_gt.png)  | ![](https://github.com/ricbl/vrgan/images/x.png) | ![]()

## License

This project is licensed under the MIT License

By: Ricardo Bigolin Lanfredi, [ricbl@sci.utah.edu](mailto:ricbl@sci.utah.edu). 