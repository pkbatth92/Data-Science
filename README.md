# Image Classification

Project code for Udacity's Data Science Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

The application has been trained on VGG flowers dataset and leverages off-the-shelf convoluted neural nets like VGG16 for the feature extraction part and uses multi-layer neural network for the classification part. 

The codebase includes 
1. a Jupyter notebook version and 
2. a command line version that has a 
  a. training section: that takes in parameters like convoluted net architecture for feature extraction, number of hidden units, epochs and learning rate for classification and saves the trained model/checkpoint in the specified location. Model can be also be run on a GPU.
  b. prediction section: that takes in a sample image and the checkpoint location and returns the class it belongs to.
  
On training, the model was able to achieve 83% accuracy.
