import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

import cv2

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')


train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

batch_size = 128
epochs = 10
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
"""
train_image_generator = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )
"""
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

sample_training_images, _ = next(train_data_gen)


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
#plotImages(sample_training_images[:5])

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])
#model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


#test_image = os.path.()
#image = cv2.imread('Image path with extension”',color/gray/binary)


predictions = model.predict(val_data_gen)

print(predictions[0])
print(predictions[1])
print(predictions[2])
print(predictions[3])
print(predictions[4])

plotImages(sample_training_images[:5])

PATH1 = os.path.join(os.getcwd(), 'animal')

_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
_data_gen = _image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=PATH1,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')
prediction = model.predict(_data_gen)
print(prediction[0])
print(prediction[1])
print(prediction[2])
print(prediction[3])
print(prediction[4])

_training_images, _ = next(_data_gen)

plotImages(_training_images[:5])


cap = cv2.VideoCapture("/dev/video2")

while(True):
    ret, frame = cap.read()
    #print(frame)
    #gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    roi = cv2.resize(frame, (IMG_HEIGHT,IMG_WIDTH))

    #capture_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data

    #capture_data_gen = capture_image_generator.flow_from_dataframe(dataframe=frame, batch_size=batch_size, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode="binary")
    #capture_images, _ = next(capture_data_gen)

    """
    capture_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory=train_dir,
                                                               shuffle=True,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='binary')
    """
    #plotImages(capture_images[:1])
    predictions = model.predict(roi[np.newaxis, :, :, :])
    print(predictions[0])

    text = "val= "
    if predictions[0][0] != None:
        text = text + str(predictions[0][0])
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    stroke = 2
    cv2.putText(frame, text, (100, 100), font, 1, color, stroke, cv2.LINE_AA)

    cv2.imshow("frame", frame)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()