import PIL.ImageShow
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import gdown
from PIL import Image
import numpy as np

model = load_model('model_cifar10_cnn_trained_100_epocs.h5')
# print(model.summary())
# download https://storage.googleapis.com/datasets_ai/Knowledge/test_images.zip
test_path = 'test_images/1.jpg'
img = Image.open(test_path)
# PIL.ImageShow.show(img)

img_width, img_height = 32, 32  # the needed img sizes

img32 = Image.open(test_path).resize((img_width, img_height))
# PIL.ImageShow.show(img32)

image = np.array(img32, dtype='float64') / 255.  # image into numpy array with normalization of pixel values
image = np.expand_dims(image, axis=0)  #  adding a layer for model shape to correspond the input shape

pred = model.predict(image)
classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print('Recognition result: ')
for i, cl in enumerate(classes):
    print('{:<14s}{:6.2%}'.format(cl, pred[0, i]))
print()

cls_image = np.argmax(model.predict(image))
print("It's a", classes[cls_image])

