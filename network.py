import torch
from torchvision import models
from torch import nn
from torch import optim
from collections import OrderedDict

class Network:
    def __init__(self, architecture):
        if architecture == "vgg13":
            self.model = models.vgg13(pretrained=True)
        elif architecture == "vgg19":
            self.model = models.vgg19(pretrained=True)
        else:
            self.model = models.vgg16(pretrained=True)
        
    def spec_classifier(self,hidden_units):
        # Freezing parameters of the model
        for param in self.model.parameters():
            param.requires_grad = False

        # Defining classifier
        classifier = nn.Sequential(OrderedDict([
                                        ('fc1', nn.Linear(25088, hidden_units)),
                                        ('relu1',nn.ReLU()),
                                        ('fc3', nn.Linear(hidden_units, 102)),
                                        ('output', nn.LogSoftmax(dim = 1))
        ]))

        self.model.classifier = classifier

    def train_classifier(self,train_loader, epochs, valid_loader,learning_rate,processor):
        # training the network (classifier)
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=learning_rate)
        epochs = epochs
        print_every = 15
        steps = 0

        self.model.to(processor)

        for e in range(epochs):
            running_loss = 0
            for ii, (inputs, labels) in enumerate(train_loader):
                steps += 1
                inputs, labels = inputs.to(processor), labels.to(processor)

                self.optimizer.zero_grad()

                outputs = self.model.forward(inputs)
                training_loss = self.criterion(outputs, labels)
                training_loss.backward()
                self.optimizer.step()

                running_loss += training_loss.item()
                if steps % print_every == 0:
                    print('Epoch: {}/{}...'.format(e+1,epochs))
                    running_loss = 0

                    self.check_accuracy_on_valid(valid_loader,processor)

    def check_accuracy_on_valid(self,valid_loader,processor):
        correct = 0
        total = 0
        running_loss = 0
        self.model.to(processor)
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(processor), labels.to(processor)
                outputs = self.model.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)
                validation_loss = self.criterion(outputs, labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += validation_loss

        print('Accuracy on validation set: %d %%' % (correct*100/total),
             'Loss on validation set: {:.4f}'.format(running_loss/total))

    def check_accuracy_on_test(self,test_loader,processor):
        correct = 0
        total = 0
        self.model.to(processor)
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(processor), labels.to(processor)
                outputs = self.model.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy on test set: %d %%' % (correct*100/total))
        
    
