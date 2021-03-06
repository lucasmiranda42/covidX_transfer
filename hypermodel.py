# @author: lucasmiranda42
"""

Transfer-learning based hypermodel for the covidX_transfer project
To be used under the keras-tuner framework

"""
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.applications.nasnet import NASNetLarge
from kerastuner import *
import tensorflow as tf
import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


class NASnet_transfer(HyperModel):
    def __init__(self, input_shape, fine_tune):
        self.input_shape = input_shape
        self.finetune = fine_tune

    def build(self, hp):

        pretrained_model = NASNetLarge(
            input_shape=(331, 331, 3), include_top=False, weights="imagenet"
        )

        if not self.finetune:
            for layer in pretrained_model.layers:
                layer.trainable = False

        # Hyperparameters to tune
        Dense_layers = hp.Int(
            "number of dense layers", min_value=0, max_value=2, step=1, default=0
        )
        Dense_units = hp.Int(
            "dense units of the first dense layer",
            min_value=128,
            max_value=512,
            step=32,
            default=256,
        )
        DROPOUT_RATE = hp.Float(
            "dropout_rate", min_value=0.0, max_value=0.5, default=0.25, step=0.05
        )

        last_layer = pretrained_model.get_layer(pretrained_model.layers[-1].name)
        last_output = last_layer.output

        # Adds a global average pooling to reduce the dimensionality of the output
        x = layers.GlobalAveragePooling2D()(last_output)
        # Add a fully connected layer with 1,024 hidden units and ReLU activation
        x = layers.Dense(Dense_units, activation="relu")(x)
        # Tries out adding more dense layers
        for i in range(Dense_layers):
            x = layers.Dense(Dense_units / (2 if i == 0 else 4 * i), activation="relu")(
                x
            )
        # Add a tunable dropout rate
        x = layers.Dropout(DROPOUT_RATE)(x)
        # Add a final sigmoid layer for classification
        x = layers.Dense(3, activation="softmax")(x)

        model = Model(pretrained_model.input, x)

        print(model.summary())

        model.compile(
            loss=categorical_crossentropy,
            optimizer=Adam(
                lr=hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling="LOG",
                    default=1e-3,
                ),
            ),
            metrics=[
                keras.metrics.TruePositives(),
                keras.metrics.FalsePositives(),
                keras.metrics.TrueNegatives(),
                keras.metrics.FalseNegatives(),
                "categorical_accuracy",
            ],
        )

        return model


def tune_search(train, test, fine_tune, project_name, verb, bayopt_trials):
    """Define the search space using keras-tuner and bayesian optimization"""
    hypermodel = NASnet_transfer(input_shape=(331, 331, 3), fine_tune=fine_tune)

    tuner = BayesianOptimization(
        hypermodel,
        max_trials=bayopt_trials,
        executions_per_trial=1,
        seed=42,
        objective="val_categorical_accuracy",
        directory="BayesianOptx",
        project_name=project_name,
        distribution_strategy=tf.distribute.MirroredStrategy(),
    )

    if verb == 2:
        print(tuner.search_space_summary())

    tuner.search(
        train,
        epochs=30,
        validation_data=(test),
        verbose=2,
        callbacks=[EarlyStopping("val_loss", patience=3), tensorboard_callback],
    )

    if verb == 2:
        print(tuner.results_summary())

    return (
        tuner.get_best_models(num_models=bayopt_trials-1),
        tuner.get_best_hyperparameters(num_trials=1),
    )


### TODO:
###       1) Revise metrics and weighted loss for class imbalance correction
