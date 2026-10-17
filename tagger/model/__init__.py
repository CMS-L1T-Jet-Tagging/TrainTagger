# flake8: noqa
try:
    from tagger.model.QKeras.DeepSetModel import DeepSetModel
    from tagger.model.QKeras.WeightedAverageSimpleModel import WeightedAverageSimpleModel
except:
    from tagger.model.Float.DeepSetModelFloat import DeepSetModelFloat
    from tagger.model.HGQ2.DeepSetModelHGQ2 import DeepSetModelHGQ2
    from tagger.model.HGQ2.JEDILinearHGQ2 import JEDILinearHGQ2
    from tagger.model.HGQ2.MLPMixerHGQ2 import MLPMixerHGQ2
    from tagger.model.HGQ2.LinformerHGQ2 import LinformerHGQ2
    from tagger.model.PQuant.DeepSetModelPQuant import DeepSetModelPQuant