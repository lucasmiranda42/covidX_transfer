# @author lucasmiranda42
"""

Main training pipeline for the covidX_transfer project

"""

import argparse
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from hypermodel import tune_search

parser = argparse.ArgumentParser(
    description="Training script for the covidX_transfer project"
)

parser.add_argument("--path", "-p", help="set path", type=str)
parser.add_argument(
    "--verbose",
    "-v",
    help="sets the verbosity of the output. Possible values: 0, 1, 2",
    default=1,
    type=int,
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
parser.add_argument(
    "--bayopt",
    "-n",
    help="sets the number of Bayesian optimization iterations to run. Default is 25",
    default=25,
    type=int,
)

args = parser.parse_args()
path = args.path
verb = int(args.verbose)
blend = args.blend
fine_tune = args.fine_tune
bayopt_trials = args.bayopt

if not path:
    raise ValueError("set a valid data path for the training to run")
if verb not in [0, 1, 2]:
    raise ValueError("verbose has to be one of 0, 1 or 2")

train_dir = "{}/train".format(path)
val_dir = "{}/validation".format(path)

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(
    samplewise_std_normalization=True,
    samplewise_center=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    vertical_flip=False,
)

# Note that the validation data should not be augmented!
val_datagen = ImageDataGenerator(
    samplewise_center=True, samplewise_std_normalization=True
)

# Flow training images in batches using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size=20,
    class_mode="categorical",
    target_size=(331, 331),
    shuffle=True,
)

# Flow validation images in batches using test_datagen generator
val_generator = val_datagen.flow_from_directory(
    val_dir,
    batch_size=20,
    class_mode="categorical",
    target_size=(331, 331),
    shuffle=True,
)


print("Starting hyperparameter tuning...")
best_models, best_params = tune_search(
    train_generator, val_generator, fine_tune, "COVIDx", verb, bayopt_trials
)

for i, model in enumerate(best_models):
    model.save("COVIDx_transfer_best_model_{}.h5".format(i))
for j, param in enumerate(best_params):
    with open("COVIDx_transfer_best_params.h5", "wb") as handle:
        pickle.dump(param, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Done!")
