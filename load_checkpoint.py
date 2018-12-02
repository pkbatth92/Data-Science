from torchvision import models
from collections import OrderedDict
import torch
from torch import nn
from network import Network

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = Network(checkpoint['pretrained_net'])
    for param in model.model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(25088, checkpoint['hidden_units'])),
                                ('relu1',nn.ReLU()),
                                ('fc3', nn.Linear(checkpoint['hidden_units'], 102)),
                                ('output', nn.LogSoftmax(dim = 1))
    ]))
    model.model.classifier = classifier
    
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model