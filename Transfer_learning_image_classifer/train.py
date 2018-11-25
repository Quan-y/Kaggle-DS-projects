import torch
import torch.utils.data as data
from torch import nn
from torch import optim

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from collections import OrderedDict

#run_train(data_dir = 'flowers')
def run_train(data_dir):
    # define transform
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])
    }
    # load data
    image_datasets = {
    x: datasets.ImageFolder(root = data_dir+'/'+x, transform=data_transforms[x])
    for x in list(data_transforms.keys())}

    dataloaders = {
    x: data.DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=2)
    for x in list(image_datasets.keys())}
    # train model
    model = load_model()
    criterion1 = nn.NLLLoss()
    optimizer1 = optim.Adam(model.classifier.parameters(), lr=0.001)
    model = train_model(model, dataloaders, criterion1, optimizer1)
    return model

# load and tune trained model
def load_model(arch='vgg19'):
    # Load a pre-trained model
    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    # Freeze its parameters
    for param in model.parameters():
        param.requires_grad = False
    # build classifer
    classifier = nn.Sequential(OrderedDict([
                         ('fc1', nn.Linear(25088, 4096)),
                         ('relu1', nn.ReLU()),
                         ('dropout1', nn.Dropout(0.2)),
                         ('fc2', nn.Linear(4096, 1000)),
                         ('relu2', nn.ReLU()),
                         ('dropout2', nn.Dropout(0.2)),
                         ('fc3', nn.Linear(1000, 102)),
                         ('output', nn.LogSoftmax(dim=1))
                         ]))
    model.classifier = classifier
    return model

# Validation function
def validation(model, dataloaders_valid, criterion):
    test_loss = 0
    accuracy = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for images, labels in dataloaders_valid:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)        
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy

# train model
def train_model(model, dataloaders, criterion, optimizer):
    '''dataloaders: dict; keys: train, valid, test'''
    # prepare data
    dataloaders_train = dataloaders['train']
    dataloaders_valid = dataloaders['valid']
    # initialization
    epochs = 3
    print_every = 40
    steps = 0
    running_loss = 0
    # GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # training
    for e in range(epochs):
        # train model
        model.train()
        for ii, (images, labels) in enumerate(dataloaders_train):
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # calculate the 
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy = validation(model, dataloaders_valid, criterion)
                print("Epoch: {}/{}  ".format(e+1, epochs),
                    "Training Loss: {:.2f}  ".format(running_loss/print_every),
                    "Test Loss: {:.2f}  ".format(test_loss/len(dataloaders_valid)),
                    "Test Accuracy: {:.2f}".format(accuracy/len(dataloaders_valid)))
                running_loss = 0
                model.train()
    return model

#test_accuracy = check_accuracy_on_test(model, dataloaders['test'])
def check_accuracy_on_test(model, dataloaders_test):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for images, labels in dataloaders_test:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    return 100 * correct / total

#save_checkpoint(image_datasets, model, 'SavedModel.pth')
def save_checkpoint(dataset, model, checkpoint_path):
    model.class_to_idx = dataset['train'].class_to_idx
    checkpoint = {
        'arch': 'vgg19',
        'class_to_idx': model.class_to_idx, 
        'state_dict': model.state_dict(),
        'classifier': model.classifier
    }
    torch.save(checkpoint, checkpoint_path)



