# flake8: noqa
try:
    from tagger.model.DeepSetModel import DeepSetModel
    from tagger.model.DeepSetModelHGQ import DeepSetModelHGQ
    from tagger.model.InteractionNetModel import InteractionNetModel
    from tagger.model.QKerasModel import QKerasModel
except:
    from tagger.model.FloatingDeepSetModel import FloatingDeepSetModel
    from tagger.model.TorchDeepSetModel import TorchDeepSetModel
    from tagger.model.DeepSetModelHGQ2 import DeepSetModelHGQ2
    from tagger.model.JEDILinearHGQ2 import JEDILinearHGQ2
    from tagger.model.MLPmixerHGQ2 import MLPmixerHGQ2
    from tagger.model.PQuantDeepSetModel import PQuantDeepSetModel

