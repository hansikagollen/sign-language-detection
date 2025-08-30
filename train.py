import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

DATA_DIR = 'my_webcam_data'  
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_HEAD = 5
EPOCHS_FINE_TUNE = 15
EPOCHS_FULL = 30

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    shear_range=0.2,
    rotation_range=20,       # random rotation
    width_shift_range=0.2,   # horizontal shift
    height_shift_range=0.2,  # vertical shift
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
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
)

steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = validation_generator.samples // BATCH_SIZE

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
base_model.trainable = False


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

checkpoint = ModelCheckpoint('best_asl_model.h5', monitor='val_accuracy', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)


print("\n🔹 Stage 1: Training head only")
model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS_HEAD,
    callbacks=[checkpoint, early_stop, lr_reduction]
)

print("\n🔹 Stage 2: Fine-tuning last 50 layers")
for layer in base_model.layers[-50:]:
    layer.trainable = True
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS_FINE_TUNE,
    callbacks=[checkpoint, early_stop, lr_reduction]
)

print("\n🔹 Stage 3: Fine-tuning full backbone")
for layer in base_model.layers:
    layer.trainable = True
model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS_FULL,
    callbacks=[checkpoint, early_stop, lr_reduction]
)

model.save('asl_model2.h5')
print("Model saved as asl_model2.h5")
