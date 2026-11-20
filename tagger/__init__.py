"""Quiet settings for the training stack, set here so they apply before it is imported."""

import os
import re
import warnings

# Hide TensorFlow's C++ start-up messages (no GPU/CUDA/TensorRT found etc.); set to 0 to see them again
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Hide harmless warnings raised inside Keras/QKeras, not by this repo:
# - deprecation of set_learning_phase
# - serialisation of QKeras quantizers, which QKeras stores as Keras constraints
for _message in (
    "`tf.keras.backend.set_learning_phase` is deprecated",
    "The `keras.constraints.serialize()` API should only be used for objects of type",
):
    warnings.filterwarnings("ignore", message=re.escape(_message), category=UserWarning)
del _message
