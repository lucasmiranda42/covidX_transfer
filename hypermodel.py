"""

Transfer-learning based hypermodel for the covidX_transfer project
To be used under the keras-tuner framework

"""

class SEQ_2_SEQ_AE(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        # Hyperparameters to tune
        Dense_layers = hp.Int(
            "number of dense layers", min_value=1, max_value=3, step=1, default=2
        )
        Dense_units = hp.Int(
            "dense units of the first dense layer", min_value=128, max_value=512, step=32, default=256
        )
        DROPOUT_RATE = hp.Float(
            "dropout_rate", min_value=0.0, max_value=0.5, default=0.25, step=0.05
        )


        model.compile(
            loss=tf.keras.losses.Huber(reduction="sum", delta=100.0),
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