# Imports here
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
import train

import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def process_image(image):
    pil_image = Image.open(image)
    image_ratio= (pil_image.size[0]/pil_image.size[1])
    if pil_image.size[0]>pil_image.size[1]:
        pil_image= pil_image.resize(((int)(256*image_ratio),256))
    else:
        pil_image= pil_image.resize((256,(int)(256/image_ratio)))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    pil_image= pil_image.crop((16,16,256,256))
    np_image = np.array(pil_image)    
    np_image= (((np_image/255)-mean)/std)
    np_image = np.transpose(np_image,(2, 0, 1))
    return np_image

#imshow(process_image('flowers/test/1/image_06760.jpg'))
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax

#model = load_checkpoint('SavedModel.pth')
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    model = train.load_model(arch=arch)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

#predict('flowers/test/1/image_06764.jpg','SavedModel.pth', topk = 5)
def predict(image_path, model_path, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = torch.from_numpy(process_image(image_path)).float()
    img.unsqueeze_(0)
    model = load_checkpoint(model_path)
    model, img = model.to(device), img.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model.forward(img)
        ps = torch.exp(output)
        result = ps.topk(topk)

    probs, classes = result[0].data.cpu().numpy()[0], result[1].data.cpu().numpy()[0]
    idx_to_class = {key: value for value, key in model.class_to_idx.items()}
    classes = [idx_to_class[classes[i]] for i in range(classes.size)]
    return probs, classes

#sanitycheck('flowers/test/2/image_05125.jpg', cat_to_name, 'SavedModel.pth')
def sanitycheck(flower_path, name_dict, model_path):
    fig, axs = plt.subplots(2, 1, figsize=(5, 10))
    axs[0].imshow(mpimg.imread(flower_path))
    probs, classes = predict(flower_path, model_path)
    class_names = [name_dict[clas] for clas in classes]
    y_pos = np.arange(len(class_names))
    axs[1].barh(y_pos, probs)
    axs[1].set_yticks(y_pos)
    axs[1].set_yticklabels(class_names)
    axs[1].invert_yaxis()
    axs[1].set_xlabel('Probability')
    axs[0].set_title(class_names[0])
    axs[1].set_title('*Sanity Checking*')
    return




