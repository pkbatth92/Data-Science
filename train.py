import numpy as np
import time
import torch
import torch.nn.functional as F
import torchvision
from torchvision import models
import argparse
from PIL import Image
from utility import *
from network import Network

parser = argparse.ArgumentParser()
parser.add_argument('data_directory')
parser.add_argument('--save_dir')
parser.add_argument('--arch')
parser.add_argument('--learning_rate')
parser.add_argument('--hidden_units')
parser.add_argument('--epochs')
parser.add_argument('--gpu',action='store_true')

args = parser.parse_args()

# Reading arguments from the command line
data_directory = args.data_directory

if args.save_dir is not None:
    save_directory = args.save_dir
else:
    save_directory = '/home/workspace/paind-project'

if args.arch is not None:
    architecture = args.arch
else:
    architecture = "vgg16"

if args.learning_rate is not None:
    learning_rate = float(args.learning_rate)
else:
    learning_rate = 0.001

if args.hidden_units is not None:
    hidden_units = int(args.hidden_units)
else:
    hidden_units = 1000

if args.epochs is not None:
    epochs = int(args.epochs)
else:
    epochs = 10

if args.gpu:
    processor = 'cuda'
else:
    processor = 'cpu'

# Loading and Pre-Processing data
train_dir = data_directory + '/train'
valid_dir = data_directory + '/valid'
test_dir = data_directory + '/test'
train_loader, train_data, valid_loader, test_loader = load_process_data(train_dir, valid_dir, test_dir)

model = Network(architecture)
model.spec_classifier(hidden_units)

model.train_classifier(train_loader, epochs, valid_loader,learning_rate,processor)

model.check_accuracy_on_test(test_loader,processor)

model.class_to_idx = train_data.class_to_idx

checkpoint = {
    'pretrained_net': architecture,
    'hidden_units': hidden_units,
    'model_state_dict': model.model.state_dict(),
    'class_to_idx': model.class_to_idx,
    'epochs': epochs
}

torch.save(checkpoint, save_directory + '/checkpoint.pth')
