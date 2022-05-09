# import pandas as pd
from preprocess import train_images, val_images, test_images
import tensorflow as tf
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, GlobalAveragePooling2D, \
    Dropout
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
import pickle
import json

resnet_model = Sequential()
base_model = ResNet50(include_top=False, weights='imagenet', classes=len(train_images.class_indices),
                      input_shape=(224, 224, 3), pooling='avg')
for layer in base_model.layers:
    layer.trainable = False
resnet_model.add(base_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(1024, activation='relu'))
resnet_model.add(Dense(1024, activation='relu'))  # neurons
# resnet_model.add(Dropout(0.5))
resnet_model.add(Dense(len(train_images.class_indices), activation='softmax'))

resnet_model.summary()

resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = resnet_model.fit(
    train_images,
    validation_data=val_images,
    epochs=50,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
)

with open('histories.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
# history_dict = history.history
# json.dump(history_dict, open('histories.json', 'w'))

resnet_model.save('C:\CLASS\python\Rec\saved2.h5')
