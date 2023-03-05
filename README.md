# Flower Species Image Classifier

Project code for Udacity's AI Programming with Python Nanodegree program. 
In this project, students first develop code for an image classifier built with PyTorch, 
then convert it into a command line application.

## Jupyter Notebook Details

The model was developed using Jupyter Notebook, with gpu support for Apple's MPS only. 
The notebook includes code for data loading, pre-processing, training the model, and testing the model. 
The training was done with resnet50 and 7 epochs. It achieved a validation accuracy of 92.5%. 
The testing accuracy on a previously unseen dataset is 92.3%.

The notebook also includes a function to load the saved checkpoint so that the model doesn't need to be retrained 
each time. In addition, there are three functions for image classification: 
one to preprocess an image, another to display the preprocessed image, and a final one to predict the class of the image. 
These functions are used in the sanity checking section to ensure the accuracy of the model's predictions.

## CLI Application Details

The command line application allows users to train an image classifier with a specified pre-trained model, using their own dataset. 
The command line application includes code for data loading, pre-processing, training the model, and testing the model. 
The training was done with densenet 121, 7 epochs and achieved 77% accuracy. 
The command line application supports gpu for Apple's MPS and Nvidia's Cuda.
The testing accuracy on a previously unseen dataset is relative. This is an output:
<img width="453" alt="Screenshot 2023-03-05 at 1 46 39 PM" src="https://user-images.githubusercontent.com/104724221/222961328-cd490de6-2f09-4601-8490-b65228f68182.png">

The user can specify the pre-trained model to use by passing a command-line argument. The following pre-trained models are available:

vgg11
vgg13
vgg16
vgg19
densenet121
densenet169
densenet161
densenet201
To run the training script, use the following command:

python train.py [data_directory] --arch [pretrained_model] --hidden_units [hidden_units] --epochs [num_epochs] --gpu --weights [pretrained_model_Weights.IMAGENET1K_V1]
To avoid warnings when using the pretrained argument in Pytorch, use the weights argument instead. 
For example, instead of using pretrained=True with VGG-16, use weights = VGG_16_Weights.IMAGENET1K_V1.
It's important to note that the weights argument is case-sensitive. 
For instance, the correct format for DenseNet is DenseNet_Weights.IMAGENET1K_V1, not Densenet_Weights.IMAGENET1K_V1 or DENSENET_Weights.IMAGENET1K_V1.
To get help on the parameters, run:

python train.py -h
The predict.py script allows users to predict the class of an image using a trained model. 
To predict the class of an image, use the following command:

python predict.py [path_to_image] [path_to_checkpoint] --top_k [num_predictions] --category_names [class_to_name_mapping] --gpu
To get help on the parameters, run:

python predict.py -h

### Dependencies

The notebook and command line application use Python 3 and the following libraries:

PyTorch (Nightly)
Matplotlib
NumPy
PIL
argparse
sys
os
collections
Please ensure that these libraries are installed and up-to-date.

