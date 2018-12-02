import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image

def load_process_data(train_dir, valid_dir, test_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

    valid_test_transforms = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = valid_test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
    valid_loader = DataLoader(valid_data, batch_size = 32)
    test_loader = DataLoader(test_data, batch_size = 32)
    
    return train_loader, train_data, valid_loader, test_loader

def im_with_top_k_classes(image_path,model,cat_to_name,top_k):
    image = process_image(image_path)
    title = cat_to_name[image_path.split('/')[6]]
    probs,classes = predict(image,model,top_k)
    types = [cat_to_name[class_] for class_ in classes]
    print('The image chosen is: ' + title)
    print('The prediction(s) is/are:')
    print(types,probs)
    
def process_image(image):
    im = Image.open(image)
    width, height = im.size
    if im.size[0]>im.size[1]:
        im.thumbnail((256*width/height,256))
    else:
        im.thumbnail((256,256*height/width))
    left = (im.width-224)/2
    top = (im.height-224)/2
    right = (left+224)
    bottom = (top+224)
    cropped_image = im.crop((left,top,right,bottom))
    np_image = np.array(cropped_image)/255
    mean = np.array([[0.485, 0.456, 0.406]])
    std = np.array([[0.229, 0.224, 0.225]])
    np_image = (np_image - mean)/std
    np_image = np_image.transpose((2,0,1))
    return np_image

def predict(image, model, top_k):
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image.unsqueeze_(0)
    log_output = model.model.forward(image)
    output = torch.exp(log_output)
    probs,indices = output.topk(k=top_k)
    probs = probs.detach().numpy().tolist()[0] 
    indices = indices.detach().numpy().tolist()[0]
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    classes = [idx_to_class[index] for index in indices]
    return probs,classes