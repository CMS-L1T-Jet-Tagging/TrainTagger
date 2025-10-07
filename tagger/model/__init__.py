# flake8: noqa
try:
    import qkeras
    from tagger.model.DeepSetModel import DeepSetModel
    from tagger.model.DeepSetModelHGQ import DeepSetModelHGQ
    from tagger.model.InteractionNetModel import InteractionNetModel
    from tagger.model.QKerasModel import QKerasModel
except:
    from tagger.model.FloatingDeepSetModel import FloatingDeepSetModel
