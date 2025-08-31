import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.utils import class_weight



DATA_DIR = 'my_webcam_data'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  # increased batch size for faster training
EPOCHS_HEAD = 3  # reduced epochs
EPOCHS_FINE_TUNE = 8  # reduced epochs
EPOCHS_FULL = 12  # reduced epochs

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    shear_range=0.3,
    rotation_range=30,       # further increased rotation
    width_shift_range=0.15,  # increased shift
    height_shift_range=0.15, # increased shift
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],  # wider brightness range
    channel_shift_range=0.2,       # increased channel shift
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    color_mode='rgb'
    subset='training',
    color_mode='rgb'
)

validation_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    color_mode='rgb'
    subset='validation',
    color_mode='rgb'
)

steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = validation_generator.samples // BATCH_SIZE

# Compute class weights to handle imbalance
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
class_weights = dict(enumerate(class_weights))

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
base_model.trainable = False


x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

checkpoint = ModelCheckpoint('best_asl_model.h5', monitor='val_loss', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)


print("\n🔹 Stage 1: Training head only")
model.compile(optimizer=Adam(1e-3), loss=CategoricalCrossentropy(label_smoothing=0.05), metrics=['accuracy'])
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS_HEAD,
    callbacks=[checkpoint, early_stop, lr_reduction],
    class_weight=class_weights
)

print("\n🔹 Stage 2: Fine-tuning layers")
for layer in base_model.layers[-75:]:
    layer.trainable = True
model.compile(optimizer=Adam(1e-4), loss=CategoricalCrossentropy(label_smoothing=0.05), metrics=['accuracy'])
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS_FINE_TUNE,
    callbacks=[checkpoint, early_stop, lr_reduction],
    class_weight=class_weights
)

print("\n🔹 Stage 3: Fine-tuning full backbone")
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
for layer in base_model.layers:
    layer.trainable = True
model.compile(optimizer=Adam(2e-5), loss=CategoricalCrossentropy(label_smoothing=0.05), metrics=['accuracy'])
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS_FULL,
    callbacks=[checkpoint, early_stop, lr_reduction]
    epochs=EPOCHS_FULL
)

model.save('asl_model2.h5')
print("Model saved as asl_model2.h5")
