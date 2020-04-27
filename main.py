# @author lucasmiranda42
"""

Main training pipeline for the covidX_transfer project

"""

import argparse
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.nasnet import NASNetLarge

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

args = parser.parse_args()
path = args.path
verb = int(args.verbose)
blend = args.blend

if not path:
    raise ValueError("set a valid path for the training to run")
if verb not in [0, 1, 2]:
    raise ValueError("verbose has to be one of 0, 1 or 2")


pre_trained_model = NASNetLarge(
    input_shape=(331, 331, 3), include_top=False, weights="imagenet"
)

if verb == 2:
    print(pre_trained_model.summary())

datagen = ImageDataGenerator(
    rotation_range=10, zoom_range=0.10, width_shift_range=0.1, height_shift_range=0.1
)


# Load data using generators

# Load hypermodel and run hyperparameter tuning

# Train and deploy the resulting model

# Explore model blending
# https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist
