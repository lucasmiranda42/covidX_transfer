# @author lucasmiranda42
"""

Main training pipeline for the covidX_transfer project

"""

import argparse
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.optimizers import Adam

parser = argparse.ArgumentParser(
    description="Training script for the covidX_transfer project"
)

parser.add_argument("--path", "-p", help="set path", type=str)
parser.add_argument(
    "--verbose",
    "-v",
    help="sets the verbosity of the output. Possible values: 0, 1, 2",
    default=1,
)
parser.add_argument(
    "--blend",
    "-b",
    help="defines the number of blending models to train",
    type=int,
    default=1,
)
parser.add_argument(
    "--fine-tune",
    "-f",
    default=False,
    help="Fine tune the whole pretrained model if True, and only the last layer if False",
    type=bool,
)

args = parser.parse_args()
path = args.path
verb = int(args.verbose)
blend = args.blend
fine_tune = args.fine_tune


if not path:
    raise ValueError("set a valid data path for the training to run")
if verb not in [0, 1, 2]:
    raise ValueError("verbose has to be one of 0, 1 or 2")


pre_trained_model = NASNetLarge(
    input_shape=(331, 331, 3), include_top=False, weights="imagenet"
)

if not fine_tune:
    for layer in pre_trained_model.layers:
        layer.trainable = False

if verb == 2:
    print(pre_trained_model.summary())

datagen = ImageDataGenerator(
    rotation_range=10, zoom_range=0.10, width_shift_range=0.1, height_shift_range=0.1
)

last_layer = pre_trained_model.get_layer(pre_trained_model.layers[-1].name)
last_output = last_layer.output

"""toy model for testing the generators"""

# Adds a global average pooling to reduce the dimensionality of the output
x = layers.GlobalAveragePooling2D()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation="relu")(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(3, activation="softmax")(x)

model = Model(pre_trained_model.input, x)

model.compile(
    optimizer=Adam(lr=0.0001), loss="binary_crossentropy", metrics=["accuracy"]
)

if verb == 2:
    print(model.summary())

train_dir = "{}/train".format(path)
test_dir = "{}/test".format(path)

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    vertical_flip=False,
)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dir, batch_size=32, class_mode="categorical", target_size=(331, 331)
)

# Flow validation images in batches of 20 using test_datagen generator
test_generator = test_datagen.flow_from_directory(
    test_dir, batch_size=32, class_mode="categorical", target_size=(331, 331)
)

history = model.fit(
    train_generator,
    validation_data=test_generator,
    steps_per_epoch=100,
    epochs=1,
    validation_steps=50,
    verbose=verb,
)

# Load hypermodel and run hyperparameter tuning

# Train and deploy the resulting model

# Explore model blending
# https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist
