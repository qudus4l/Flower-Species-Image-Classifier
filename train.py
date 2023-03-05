import sys
import os
import json
import train_args
import torch

from collections import OrderedDict
from torchvision import models
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def main():
    """
        Image Classification Network
    """
    parser = train_args.get_args()
    cli_args = parser.parse_args()

    # check for data directory
    if not os.path.isdir(cli_args.train_directory):
        print(f'Data directory {cli_args.train_directory} was not found.')
        exit(1)

    # check for save directory
    if not os.path.isdir(cli_args.save_dir):
        print(f'Directory {cli_args.save_dir} does not exist. Creating...')
        os.makedirs(cli_args.save_dir)

    # Load the data
    expected_means = [0.485, 0.456, 0.406]
    expected_std = [0.229, 0.224, 0.225]
    max_image_size = 224
    batch_size = 32

    # Define  transforms for the training set
    training_transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=0.25),
                                              transforms.RandomRotation(30),
                                              transforms.RandomGrayscale(p=0.02),
                                              transforms.RandomResizedCrop(max_image_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize(expected_means, expected_std)])

    training_dataset = datasets.ImageFolder(cli_args.train_directory, transform=training_transforms)

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    # label mapping
    with open(cli_args.categories, 'r') as f:
        cat_to_name = json.load(f)

    # set output to the number of categories
    output_size = len(cat_to_name)
    print(f"Images are labeled with {output_size} categories.")

    if not cli_args.arch.startswith("densenet") and not cli_args.arch.startswith("vgg"):
        print("This app supports ResNet and DenseNet only")
        exit(1)

    print(f"Using a pre-trained {cli_args.arch} network.")
    cli_model = models.__dict__[cli_args.arch](weights=cli_args.weights)

    densenet_input = {
        'densenet121': 1024,
        'densenet169': 1664,
        'densenet161': 2208,
        'densenet201': 1920
    }

    input_size = 0

    # Input size from current classifier if VGG
    if cli_args.arch.startswith("vgg"):
        input_size = cli_model.classifier[0].in_features

    # Input size from current classifier if resnet
    if cli_args.arch.startswith("densenet"):
        input_size = densenet_input[cli_args.arch]

    # Turn off gradients, freeze parameters
    for param in cli_model.parameters():
        param.requires_grad = False

    od = OrderedDict()
    hidden_sizes = cli_args.hidden_units

    hidden_sizes.insert(0, input_size)

    print(f"Building a {len(cli_args.hidden_units)} hidden layer classifier with inputs {cli_args.hidden_units}")

    for i in range(len(hidden_sizes) - 1):
        od['fc' + str(i + 1)] = nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
        od['relu' + str(i + 1)] = nn.ReLU()
        od['dropout' + str(i + 1)] = nn.Dropout(p=0.15)

    od['output'] = nn.Linear(hidden_sizes[i + 1], output_size)
    od['softmax'] = nn.LogSoftmax(dim=1)

    classifier = nn.Sequential(od)

    cli_model.classifier = classifier
    cli_model.zero_grad()
    criterion = nn.NLLLoss()

    print(f"Setting optimizer learning rate to {cli_args.learning_rate}.")
    optimizer = optim.Adam(cli_model.classifier.parameters(), lr=cli_args.learning_rate)

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
    cli_model = cli_model.to(device)

    data_set_len = len(training_dataloader.batch_sampler)

    chk_every = 50

    print(f'Using the {device} device to train.')
    print(f'Training on {data_set_len} batches of {training_dataloader.batch_size}.')
    print(f'Displaying average loss and accuracy for {cli_args.epochs} epochs every {chk_every} batches.')

    for e in range(cli_args.epochs):
        e_loss = 0
        prev_chk = 0
        total = 0
        correct = 0
        print(f'\nEpoch {e + 1} of {cli_args.epochs}\n----------------------------')
        for ii, (images, labels) in enumerate(training_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            # Set gradients of all parameters to zero.
            optimizer.zero_grad()

            # Propagate forward and backward
            outputs = cli_model.forward(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Keep a running total of loss for
            # this epoch
            e_loss += loss.item()

            # Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Keep a running total of loss for
            # this epoch
            itr = (ii + 1)
            if itr % chk_every == 0:
                avg_loss = f'avg. loss: {e_loss / itr:.4f}'
                acc = f'accuracy: {(correct / total) * 100:.2f}%'
                print(f'  Batches {prev_chk:03} to {itr:03}: {avg_loss}, {acc}.')
                prev_chk = (ii + 1)

    print('Training complete.')
    # Save the checkpoint
    cli_model.class_to_idx = training_dataset.class_to_idx
    model_state = {
        'epoch': cli_args.epochs,
        'state_dict': cli_model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'classifier': cli_model.classifier,
        'class_to_idx': cli_model.class_to_idx,
        'arch': cli_args.arch,
        'weights': cli_args.weights
    }

    save_location = f'{cli_args.save_dir}/checkpoint.pth'
    print(f"Saving checkpoint to {save_location}...")

    torch.save(model_state, save_location)
    print('Saved successfully!')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted.")
        sys.exit(0)
"""
 Ensures main() only gets called if the script is
 executed directly and not as an include. 
 Prevent stacktrace on Ctrl-C
"""
