from classification import Alexnet, EffNet2, MobileNetV2, VGG16
from loader import create_image_data_loaders
from helper import create_folders_if_not_exist
import torch
import gc
torch.cuda.empty_cache()
gc.collect()

create_folders_if_not_exist()

model_class = [Alexnet, EffNet2, MobileNetV2, VGG16]

for mc in model_class:
    # Clear GPU memory before creating a new model
    torch.cuda.empty_cache()
    
    m = mc()
    train, validate, classes = create_image_data_loaders('cat-and-dog-images-dataset/Dog and Cat .png')
    
    if hasattr(m, 'weights'):
        train, validate, classes = create_image_data_loaders('cat-and-dog-images-dataset/Dog and Cat .png', preprocess=m.weights.transforms())
    
    data = m.train(train, validate, 1)
    m.save_model()
    m.save_training_data(data[0], data[1], data[2], data[3])