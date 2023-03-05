import argparse


def get_args():
    """
    Basic usage: python predict.py /path/to/image checkpoint
    Options:
    Return top K  most likely classes: python predict.py input checkpoint --top_k 3
    Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
    Use GPU for inference: python predict.py input checkpoint --gpu
    """
    parser = argparse.ArgumentParser(
        description="Uses a trained network to predict the class for an input image",
        usage="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('path_to_image',
                        help='Path to image file.',
                        action="store")
    parser.add_argument('--checkpoint', type=str, action='store',
                        help='Directory to load checkpoint from')
    parser.add_argument('--top_k', type=int, action='store',
                        help='return top K most likely classes', default=3)
    parser.add_argument('--category', action='store', default="cat_to_name.json",
                        type=str, help='Directory to json file')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Use GPU for inference')

    parser.parse_args()
    return parser


def main():
    """
        Main Function
    """
    print(f'Command line argument utility for predict.py.\nTry "python predict.py -h".')


if __name__ == '__main__':
    main()
"""
 main() is called if script is executed on it's own.
"""
