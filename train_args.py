import argparse

available_models = [
    'vgg11',
    'vgg13',
    'vgg16',
    'vgg19',
    'densenet121',
    'densenet169',
    'densenet161',
    'densenet201'
]


def get_args():
    """
    Basic usage: python train_args.py data_directory
    Prints out training loss, validation loss, and validation accuracy as the network trains
    Options:
    Set directory to save checkpoints: python train_args.py data_dir --save_dir save_directory
    Choose architecture: python train_args.py data_dir --arch "vgg13"
    Set hyper-parameters: python train_args.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    Use GPU for training: python train_args.py data_dir --gpu
    """

    parser = argparse.ArgumentParser(
        description="Train a new network on a dataset and save the model as a checkpoint.",
        usage="python train.py flowers/train --save_dir aipnd-project-master --arch densenet121 --gpu --hidden_units \
              512 --epochs 7 --weights DenseNet121_Weights.IMAGENET1K_V1 ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('train_directory', action="store")
    parser.add_argument('--save_dir', type=str, action="store",
                        help='Directory to save checkpoints')
    parser.add_argument('--categories', action="store",
                        default="cat_to_name.json", type=str,
                        help='Path to file containing the categories.', )
    parser.add_argument('--arch', default='vgg16', action='store', type=str,
                        help='Choose architecture from' + str(available_models))
    parser.add_argument('--weights', default=None, action='store', type=str,
                        help='Choose weights for the selected architecture. For its most up to date weights,\
                        use {arch}_Weights.IMAGENET1K_V1')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Use GPU')

    hp = parser.add_argument_group('hyper_parameters')

    hp.add_argument('--learning_rate',
                    action="store",
                    default=0.003,
                    type=float,
                    help='Learning rate')

    hp.add_argument('--hidden_units', '-hu',
                    action="store",
                    dest="hidden_units",
                    default=[3136, 784],
                    type=int,
                    nargs='+',
                    help='Hidden layer units')

    hp.add_argument('--epochs',
                    action="store",
                    dest="epochs",
                    default=1,
                    type=int,
                    help='Epochs')

    parser.parse_args()
    return parser


def main():
    """
        Main Function
    """
    print(f'Command line argument utility for train_args.py.\nTry "python train_args.py -h".')


if __name__ == '__main__':
    main()
"""
 main() is called if script is executed on it's own.
"""
