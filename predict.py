import sys
import json
import numpy as np
from PIL import Image
import torch
from torchvision import models
from torchvision import transforms
import predict_args
import warnings


def load_checkpoint(device, file='checkpoint.pth'):
    """
    Loads model checkpoint saved by train.py
    """
    # Loading weights for CPU model while trained on GPU
    model_state = torch.load(file, map_location=lambda storage, loc: storage)

    model = models.__dict__[model_state['arch']](weights = model_state['weights'])
    model = model.to(device)

    model.classifier = model_state['classifier']
    model.load_state_dict(model_state['state_dict'])
    model.class_to_idx = model_state['class_to_idx']

    return model


def process_image(image_path):
    """ Scales, crops, and normalizes a PyTorch tensor image for a
        PyTorch model, returns a Numpy array
    """
    # Define the transformations for the input image
    with Image.open(image_path) as im:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Apply the transformations to the input image
        image = transform(im)

    return image


def predict(image_path, model, topk=5):
    """ Predict the class (or classes) of an image using a trained deep learning model.
    """
    model.eval()
    model.cpu()
    # Implement the code to predict the class from an image file
    image = process_image(image_path)
    image = image.unsqueeze(0)
    # Calculate the class probabilities (softmax) for image
    with torch.no_grad():
        output = model.forward(image)
        top_p, top_class = torch.topk(output, topk)
        top_p = top_p.to('cpu')
        top_class = top_class.to('cpu')
        top_p = top_p.exp()
        class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
        mapped_classes = list()

    for label in top_class.numpy()[0]:
        mapped_classes.append(class_to_idx_inv[label])
    np.set_printoptions(suppress=True, formatter={'float_kind': '{:f}'.format})

    return top_p.numpy()[0], mapped_classes


def main():
    """
        Image Classification Network Prediction
    """
    parser = predict_args.get_args()
    cli_args = parser.parse_args()

    # label mapping
    with open(cli_args.category, 'r') as f:
        cat_to_name = json.load(f)

    # Start with CPU
    device = torch.device("cpu")

    # Requested GPU, cuda for Nvidia GPU, mps for Apple M-Silicon
    try:
        if cli_args.gpu and torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif cli_args.gpu and torch.backends.mps.is_available():
            device = torch.device("mps:0")
        else:
            print("GPU is not available. Using CPU.")

        print(f"Sending model to device {device}.")
    except ModuleNotFoundError:
        print("cuda module is not available. Using CPU.")
        device = torch.device("cpu")
    # load model
    cli_model = load_checkpoint(device, cli_args.checkpoint)

    top_prob, top_classes = predict(cli_args.path_to_image, cli_model, cli_args.top_k)
    label = top_classes[0]
    prob = top_prob[0]

    print(f'Parameters\n---------------------------------')

    print(f'Image  : {cli_args.path_to_image}')
    print(f'Model  : {cli_args.checkpoint}')
    print(f'Device : {device}')

    print(f'\nPrediction\n---------------------------------')

    print(f'Flower      : {cat_to_name[label]}')
    print(f'Label       : {label}')
    print(f'Probability : {prob * 100:.2f}%')

    print(f'\nTop K\n---------------------------------')

    for i in range(len(top_prob)):
        print(f"{cat_to_name[top_classes[i]]:<25} {top_prob[i] * 100:.2f}%")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted.")
        sys.exit(0)
"""
 Ensures main() only gets called if the script is
 executed directly and not as an include. 
 Prevent stacktrace on Cmd-C
"""
