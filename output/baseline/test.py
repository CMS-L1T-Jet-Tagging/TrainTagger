from qkeras.utils import load_qmodel

model = load_qmodel("saved_model.h5")

model.summary()