import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
# from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, GlobalAveragePooling2D, \
#     Dropout
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

train_df = pd.read_pickle('C:\CLASS\python\Rec\\ftrain1.pkl')
test_df = pd.read_pickle('C:\CLASS\python\Rec\\ftest1.pkl')
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True, )

batch_size_trainingft = 64
batch_size_validationft = 64

train_images = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=batch_size_trainingft,
    shuffle=True,
    seed=42,
    subset='training'
)
# print(train_images)
# train_images.to_pickle('C:\CLASS\python\Rec\\ftrain_image.pkl')

val_images = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=batch_size_validationft,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=batch_size_trainingft,
    shuffle=False
)
