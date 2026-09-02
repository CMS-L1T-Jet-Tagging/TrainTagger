import numpy as np
import tensorflow as tf
import keras

@keras.saving.register_keras_serializable(package="Tagger")
class StandardScaler(keras.layers.Layer):
    def __init__(self, input_dim: int, epsilon: float = 1e-6, **kwargs,):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.epsilon = epsilon

        self.mean = self.add_weight(
            name="mean",
            shape=(input_dim,),
            initializer="zeros",
            trainable=False,
        )

        self.std = self.add_weight(
            name="std",
            shape=(input_dim,),
            initializer="ones",
            trainable=False,
        )

    def call(self, inputs):
        return (inputs - self.mean) / self.std

    def fit_reference(self, x, verbose: int = 0):
        x = np.asarray(x, dtype=np.float32,)
        self.mean.assign(np.mean(x, axis=0))
        self.std.assign(np.maximum(np.std(x, axis=0), self.epsilon))

        if verbose == 1:
            print(f"Fitted StandardScaler:")
            print(f"  Means: {self.mean.numpy():.4f}")
            print(f"  Stds:  {self.std.numpy():.4f}")

    def get_config(self):
        config = super().get_config()
        config.update({"input_dim": self.input_dim, "epsilon": self.epsilon,})
        return config

    def inverse_transform(self, x):
        return (x * self.std) + self.mean


@keras.saving.register_keras_serializable(package="Tagger")
class Autoencoder(keras.Model):
    def __init__(self, input_dim: int, model_layers: list[int], **kwargs,):
        super().__init__(**kwargs)

        nlayers = len(model_layers)
        assert nlayers % 2 == 1, "model_layers must have an odd number of layers for symmetric autoencoder."

        self.input_dim = input_dim
        self.model_layers = model_layers
        self.lr = 1e-3
        self.loss_name = "mse"

        self.scaler = StandardScaler(input_dim=input_dim, name="standard_scaler",)

        self.model = self.build_model(input_shape=(input_dim,), output_shape=(input_dim,))
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        self.compile()

    def get_layer_config(input_dim: int):

        return {
            10: [8, 6, 4, 6, 8],
            16: [12, 8, 4, 8, 12],
            32: [16, 8, 4, 8, 16],
            64: [32, 16, 8, 16, 32],
            128: [64, 32, 16, 32, 64],
        }

    def build_model(self, input_shape, output_shape):

        num_layers = len(self.model_layers)
        assert num_layers % 2 == 1, "model_layers must have an odd number of layers for symmetric autoencoder."

        inputs = keras.layers.Input(shape=input_shape, name="input_layer")

        for i, units in enumerate(self.model_layers[: num_layers // 2]):
            x = keras.layers.Dense(units=units, activation="relu", name=f"encoder_dense_{i}")(x if i > 0 else inputs)

        x = keras.layers.Dense(units=self.model_layers[num_layers // 2], activation=None, name="bottleneck")(x)

        for i, units in enumerate(self.model_layers[num_layers // 2 + 1 :]):
            x = keras.layers.Dense(units=units, activation="relu", name=f"decoder_dense_{i}")(x)

        x = keras.layers.Dense(units=output_shape[0], activation=None, name="reconstruction")(x)

        return keras.Model(inputs=inputs, outputs=x, name="autoencoder")


    def fit_scaler(self, x: np.ndarray):
        self.scaler.fit_reference(x)

    def compile(self, **kwargs):
        super().compile(optimizer=self.optimizer, loss=self.loss_name, **kwargs,)

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def fit(
            self, 
            x: np.ndarray,
            callbacks=None, 
            **kwargs,
        ):

        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True,),
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6,),
            ]

        return super().fit(x, y=x, callbacks=callbacks, **kwargs,)

    def score(self, x: np.ndarray, batch_size: int = 20_000, verbose: int = 1,) -> np.ndarray:
        x_recon = self.model.predict(x, batch_size=batch_size, verbose=verbose,)
        return np.sqrt(np.mean((x - x_recon) ** 2, axis=1,))

    def get_config(self):
        config = super().get_config()

        config.update({
            "input_dim": self.input_dim,
            "model_layers": self.model_layers,
        })

        return config



@keras.saving.register_keras_serializable(package="Tagger")
class MahalanobisModel(keras.Model):
    """
    Mahalanobis anomaly score using a fitted background mean
    and inverse covariance matrix.
    """

    def __init__(
        self,
        input_dim: int,
        regularization: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.regularization = regularization

        self.mean = self.add_weight(
            name="mean",
            shape=(input_dim,),
            initializer="zeros",
            trainable=False,
        )

        self.inverse_covariance = self.add_weight(
            name="inverse_covariance",
            shape=(input_dim, input_dim),
            initializer="identity",
            trainable=False,
        )

    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, self.compute_dtype)

        delta = inputs - self.mean

        mahalanobis_squared = tf.einsum(
            "bi,ij,bj->b",
            delta,
            self.inverse_covariance,
            delta,
        )

        return tf.sqrt(tf.maximum(mahalanobis_squared, 0.0,)
        )

    def fit_reference(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64,)

        mean = np.mean(x, axis=0,)
        covariance = np.cov(x, rowvar=False,)

        covariance += (np.eye(self.input_dim, dtype=np.float64,) * self.regularization)
        inverse_covariance = np.linalg.pinv(covariance)

        self.mean.assign(mean.astype(np.float32))
        self.inverse_covariance.assign(inverse_covariance.astype(np.float32))

        # Build the model so it can be saved/predicted immediately.
        if not self.built:
            self(tf.zeros((1, self.input_dim), dtype=tf.float32))

    def score(self, x: np.ndarray, batch_size: int = 20_000, verbose: int = 0,) -> np.ndarray:
        return self.predict(x, batch_size=batch_size, verbose=verbose,)

    def get_config(self):
        config = super().get_config()

        config.update({
            "input_dim": self.input_dim,
            "regularization": self.regularization,
        })

        return config