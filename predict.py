import time
import json
import torchvision
from torchvision import datasets, transforms, models
import argparse
from utility import *
from load_checkpoint import *

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('checkpoint')
parser.add_argument('--top_k')
parser.add_argument('--category_names')
parser.add_argument('--gpu',action='store_true')

args = parser.parse_args()

input_image = args.input
checkpoint = args.checkpoint

if args.top_k is not None:
    top_k = args.top_k
else:
    top_k = 1
top_k = int(top_k)

if args.category_names is not None:
    category_names = args.category_names
else:
    category_names = 'cat_to_name.json'

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
    
if args.gpu:
    processor = 'cuda'
else:
    processor = 'cpu'
    
model = load_checkpoint(checkpoint)

im_with_top_k_classes(input_image,model,cat_to_name,top_k)