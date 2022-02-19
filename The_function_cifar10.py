import PIL.ImageShow
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model


def predict(img_path='./test_image_1.jpg', model_path='./model_cifar10_cnn_trained_100_epocs.h5'):
    classes = {0: 'airplane',
               1: 'car',
               2: 'bird',
               3: 'cat',
               4: 'deer',
               5: 'dog',
               6: 'frog',
               7: 'horse',
               8: 'ship',
               9: 'truck'}

    model = load_model(model_path)

    img_width, img_height = 32, 32

    img = Image.open(img_path).resize((img_height, img_width))
    image = np.array(img, dtype='float64') / 255.

    image = np.expand_dims(image, axis=0)
    cls_image = np.argmax(model.predict(image))

    print(classes[cls_image])


# Time to test the function:
img_path = 'test_images/5.jpg'
predict(img_path=img_path, model_path='model_cifar10_cnn_trained_100_epocs.h5')

img_test = Image.open(img_path)
PIL.ImageShow.show(img_test)