from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

def predict(img_path='./test_image_1.jpg', model_path='./model_cifar10_cnn_trained_100_epocs.h5'):
    classes = {0: 'самолет',
               1: 'автомобиль',
               2: 'птица',
               3: 'кот',
               4: 'олень',
               5: 'собака',
               6: 'лягушка',
               7: 'лошадь',
               8: 'корабль',
               9: 'грузовик'}

    model = load_model(model_path)
