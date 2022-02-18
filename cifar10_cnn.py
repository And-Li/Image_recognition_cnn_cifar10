from tensorflow.keras.datasets import cifar10
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

(x_train10, y_train10), (x_test10, y_test10) = cifar10.load_data()

classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

x_train10 = x_train10 / 255.
x_test10 = x_test10 / 255.
y_train10 = utils.to_categorical(y_train10)
y_test10 = utils.to_categorical(y_test10)

batch_size = 128

print(x_train10.shape)
print(x_test10.shape)
print(y_train10.shape)
print(y_test10.shape)

model = Sequential()

model.add(BatchNormalization(input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss="categorical_crossentropy",
              optimizer=Adam(learning_rate=0.0001),
              metrics=["accuracy"])
model.fit(x_train10,
          y_train10,
          batch_size=batch_size,
          epochs=100,
          validation_data=(x_test10, y_test10),
          verbose=1)

model.save('model_cifar10_cnn_trained_100_epocs.h5')
# model = load_model('model_cifar10_100epochs.h5')
model_eval = model.evaluate(x_train10, y_train10, verbose=1)
print()
print('Accuracy', model_eval[1])



