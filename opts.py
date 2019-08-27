"""User configuration file

File organinzing all configurations that may be set by user when running the 
train.py script. 
Call "python train.py for a complete and formatted list of available user options.
"""

import argparse
import time
from random import randint

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def get_opt():
    parser = argparse.ArgumentParser(description='Configuration for running VRGAN code')
    
    parser.add_argument('--skip_train', type=str2bool, nargs='?', default='false',
                        help='If you just want to run validation, set this value to true.')
    parser.add_argument('--lambda_reg', type=float, nargs='?', default=0.03,
                    help='Multiplier for the generator regularization loss L_{REG}. Appears on Eq. 6 on the paper.')
    parser.add_argument('--lambda_gxprime', type=float, nargs='?', default=0.3,
                        help='Multiplier for the generator loss L_{Gx\'}. Appears on Eq. 6 on the paper.')
    parser.add_argument('--lambda_rx', type=float, nargs='?', default=1.0,
                        help='Multiplier for the regressor loss L_{Rx}. Appears on Eq. 6 on the paper.')
    parser.add_argument('--lambda_rxprime', type=float, nargs='?', default=0.3,
                            help='Multiplier for the regressor loss L_{Rx\'}. Appears on Eq. 6 on the paper.')
    parser.add_argument('--batch_size', type=int, nargs='?', default=20,
                            help='Batch size for training the toy dataset.')
    parser.add_argument('--folder_toy_dataset', type=str, nargs='?', default='./',
                                    help='If you want to load/save toy dataset files in a folder other than the local folder, change this variable.')
    parser.add_argument('--save_folder', type=str, nargs='?', default='./runs',
                                help='If you want to save files and outputs in a folder other than \'./runs\', change this variable.')
    parser.add_argument('--learning_rate_g', type=float, nargs='?', default=1e-4,
                            help='Learning rate for the optimizer used for updating the weigths of the generator')
    parser.add_argument('--learning_rate_r', type=float, nargs='?', default=1e-4,
                                help='Learning rate for the optimizer used for updating the weigths of the regressor')
    # parser.add_argument('--use_xray_dataset', type=str2bool, nargs='?', default='false',
    #                     help='The model will run for the toy dataset by default. \
    #                      If you want to run a demo for the xray dataset ,set this to true. \
    #                      Training will be skipped if true. If this variable is true, \
    #                      you should also provide the variables xray_x, xray_y and xray_yprime.')
    # parser.add_argument('--inference_x', type=str, nargs='?', default='',
    #                             help='Set a path for an input image to use for a single inference of the model.')
    # parser.add_argument('--inference_y', type=str, nargs='?', default=0.7,
    #                             help='Set a value to use as the original PFT output (FEV1/FVC) of the input image')
    # parser.add_argument('--inference_yprime', type=str, nargs='?', default=0.7,
    #                             help='Set a value to use as desired PFT output (FEV1/FVC) for the input image')
    parser.add_argument('--gpus', type=str, nargs='?', default=None,
                                help='Set the gpus to use, using CUDA_VISIBLE_DEVICES syntax.')
    parser.add_argument('--experiment', type=str, nargs='?', default='',
                                help='Set the name of the folder where to save the run.')
    parser.add_argument('--nepochs', type=int, nargs='?', default=1000,
                                help='Number of epochs to run training and validation')
    parser.add_argument('--split_validation', type=str, nargs='?', default='val',
                                    help='Use \'val\' to use the validation set for calculating scores every epoch. Use \'test\' for using the test set')
    parser.add_argument('--load_checkpoint_g', type=str, nargs='?', default=None,
                                    help='Set a filepath locating a model checkpoint for the generator that you want to load')
    parser.add_argument('--load_checkpoint_r', type=str, nargs='?', default=None,
                                        help='Set a filepath locating a model checkpoint for the regressor that you want to load')
    args = parser.parse_args()
    
    timestamp = time.strftime("%Y%m%d-%H%M%S") + '-' + str(randint(1000,9999))
    args.timestamp = timestamp
    
    return args
