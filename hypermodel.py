"""

Transfer-learning based hypermodel for the covidX_transfer project
To be used under the keras-tuner framework

"""

from keras import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras_applications.densenet import layers
from kerastuner import *


class NASnet_transfer(HyperModel):
    def __init__(self, input_shape, pretrained_model):
        self.input_shape = input_shape
        self.pretrained_model = pretrained_model

    def build(self, hp):
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

        last_layer = self.pretrained_model.get_layer(
            self.pretrained_model.layers[-1].name
        )
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

        model = Model(pre_trained_model.input, x)

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
            metrics=[""],
        )

        return model


def tune_search(train, test, pretrained_model, project_name, verb):
    """Define the search space using keras-tuner and bayesian optimization"""
    hypermodel = NASnet_transfer(
        input_shape=(331, 331, 3), pretrained_model=pretrained_model
    )

    tuner = BayesianOptimization(
        hypermodel,
        max_trials=100,
        executions_per_trial=3,
        seed=42,
        objective="val_accuracy",
        directory="BayesianOptx",
        project_name=project_name,
        # distribution_strategy=tf.distribute.MirroredStrategy(),
    )

    if verb == 2:
        print(tuner.search_space_summary())

    tuner.search(
        train,
        train,
        epochs=30,
        validation_data=(test, test),
        verbose=verb,
        batch_size=128,
        callbacks=[tf.keras.callbacks.EarlyStopping("val_loss", patience=3)],
        class_weight={"normal": 1, "pneumonia": 1, "COVID-19": 1},
    )

    if verb == 2:
        print(tuner.results_summary())

    return tuner.get_best_models()[0]
