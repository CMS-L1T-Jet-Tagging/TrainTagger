import tensorflow as tf
import keras
import numpy as np

from tagger.model.common.tensorflow import effective_rank, alignment, embedding_dim_std


class ContrastiveTrainer(keras.Model):
    """
    Contrastive trainer for embedding models.

    This class implements a custom training loop for contrastive learning,
    optionally including a hyperbolic quantization (HGQ) loss.

    Note, this is a keras.Model so that it can be used with keras callbacks and fit() method. 
    The embedding model is passed in as a submodel, and the loss function is also passed in.
    Callbacks by default will operate on the outer model, but can be wrapped to operate on the inner embedding model if desired.
    """
    
    def __init__(self, embedding_model, loss_fn, use_hgq=False, **kwargs,):
        super().__init__(**kwargs)

        self.embedding_model = embedding_model
        self.loss_fn = loss_fn
        self.use_hgq = use_hgq

        self.loss_tracker = keras.metrics.Mean(name="loss")

        if self.use_hgq:
            self.emb_loss_tracker = keras.metrics.Mean(name="emb_loss")
            self.hgq_loss_tracker = keras.metrics.Mean(name="hgq_loss")

    @property
    def metrics(self):
        metrics = [self.loss_tracker]
        if self.use_hgq:
            metrics += [self.emb_loss_tracker, self.hgq_loss_tracker]
        return metrics

    def _maybe_add_hgq_loss(self, emb_loss):
        if self.use_hgq:
            hgq_loss = (
                tf.add_n(self.embedding_model.losses)
                if self.embedding_model.losses
                else tf.constant(0.0, dtype=emb_loss.dtype)
            )
            loss = emb_loss + hgq_loss
        else:
            hgq_loss = tf.constant(0.0, dtype=emb_loss.dtype)
            loss = emb_loss
        return loss, hgq_loss

    def train_step(self, data):
        (x1, x2), y, w = data

        with tf.GradientTape() as tape:
            z1 = self.embedding_model(x1, training=True)
            z2 = self.embedding_model(x2, training=True)
            z = tf.stack([z1, z2],axis=0,) # Shape: [2, batch_size, embedding_dim]
            emb_loss = self.loss_fn(features=z, labels=y, weights=w,)
            loss, hgq_loss = self._maybe_add_hgq_loss(emb_loss)

        variables = self.embedding_model.trainable_variables
        gradients = tape.gradient(loss, variables,)
        gradients_and_variables = [(g, v) for g, v in zip(gradients, variables) if g is not None]
        self.optimizer.apply_gradients(gradients_and_variables)

        self.loss_tracker.update_state(loss)
        if self.use_hgq:
            self.emb_loss_tracker.update_state(emb_loss)
            self.hgq_loss_tracker.update_state(hgq_loss)

        return {
            "loss": self.loss_tracker.result(),
            **({"emb_loss": self.emb_loss_tracker.result(), "hgq_loss": self.hgq_loss_tracker.result()} if self.use_hgq else {}),
        }

    def test_step(self, data):
        (x1, x2), y, w = data

        z1 = self.embedding_model(x1, training=False,)
        z2 = self.embedding_model(x2, training=False,)
        z = tf.stack([z1, z2], axis=0,)
        emb_loss = self.loss_fn(features=z, labels=y, weights=w,)
        loss, hgq_loss = self._maybe_add_hgq_loss(emb_loss)

        self.loss_tracker.update_state(loss)
        if self.use_hgq:
            self.emb_loss_tracker.update_state(emb_loss)
            self.hgq_loss_tracker.update_state(hgq_loss)

        return {
            "loss": self.loss_tracker.result(),
            **({"emb_loss": self.emb_loss_tracker.result(), "hgq_loss": self.hgq_loss_tracker.result()} if self.use_hgq else {}),
        }


class EmbeddingDiagnostics(keras.callbacks.Callback):

    def __init__(
        self,
        backbone_model,
        x_train,
        x_val,
        max_samples=100_000,
        batch_size=20_000,
        seed=42,
    ):
        super().__init__()

        self.backbone_model = backbone_model
        self.batch_size = batch_size

        rng = np.random.default_rng(seed)
        self.x_train = self._subsample(x_train, max_samples, rng,)
        self.x_val = self._subsample(x_val, max_samples, rng,)

    @staticmethod
    def _subsample(x, max_samples, rng):
        if len(x) <= max_samples:
            return x
        indices = rng.choice(len(x),size=max_samples,replace=False,)
        return x[indices]

    def _compute_metrics(self, x):
        z = self.backbone_model.predict(x, batch_size=self.batch_size, verbose=0,)

        return {
            "norm": np.mean(np.linalg.norm(z, axis=1)),
            "eff_rank": float(effective_rank(z, norm=True,)),
            "min_std": float(embedding_dim_std(z, norm=False,)[0]),
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_metrics = self._compute_metrics(self.x_train)
        val_metrics = self._compute_metrics(self.x_val)
        logs.update(train_metrics)
        logs.update({f"val_{name}": value for name, value in val_metrics.items()})


class InnerModelCallback(keras.callbacks.Callback):
    """
    Wrap a callback so it operates on a nested Keras model.

    Example:
        InnerModelCallback(
            FreeEBOPs(),
            model_attr="embedding_model",
        )
    """
    def __init__(self, callback: keras.callbacks.Callback, model_attr: str,):
        super().__init__()
        self.callback = callback
        self.model_attr = model_attr

    def set_model(self, model):
        # Keep the outer model associated with the wrapper.
        super().set_model(model)
        # But make the wrapped callback see the inner model.
        inner_model = getattr(model, self.model_attr,)
        self.callback.set_model(inner_model)

    def set_params(self, params):
        super().set_params(params)
        self.callback.set_params(params)

    def on_train_begin(self, logs=None):
        self.callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs)

    def on_epoch_begin(self, epoch, logs=None):
        self.callback.on_epoch_begin(epoch, logs,)

    def on_epoch_end(self, epoch, logs=None):
        self.callback.on_epoch_end(epoch, logs,)

    def on_train_batch_begin(self, batch, logs=None,):
        self.callback.on_train_batch_begin(batch, logs,)

    def on_train_batch_end(self, batch, logs=None,):
        self.callback.on_train_batch_end(batch, logs,)

    def on_test_begin(self, logs=None):
        self.callback.on_test_begin(logs)

    def on_test_end(self, logs=None):
        self.callback.on_test_end(logs)

    def on_test_batch_begin(self, batch, logs=None,):
        self.callback.on_test_batch_begin(batch, logs,)

    def on_test_batch_end(self, batch, logs=None,):
        self.callback.on_test_batch_end(batch, logs,)