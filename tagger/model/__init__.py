# flake8: noqa
try:
    from tagger.model.DeepSetModel import DeepSetModel
    from tagger.model.DeepSetModel import DeepSetEmbeddingModel
    from tagger.model.DeepSetModelHGQ import DeepSetModelHGQ
    from tagger.model.InteractionNetModel import InteractionNetModel
    from tagger.model.QKerasModel import QKerasModel
except:
    from tagger.model.FloatingDeepSetModel import FloatingDeepSetModel, FloatingDeepSetEmbeddingModel
    from tagger.model.DeepSetModelHGQ2 import DeepSetModelHGQ2, DeepSetHGQ2EmbeddingModel
    from tagger.model.JEDILinear import JEDILinearModel, JEDILinearEmbeddingModel
    from tagger.model.JEDILinearHGQ2 import JEDILinearHGQ2, JEDILinearHGQ2EmbeddingModel
    from tagger.model.LinformerModelHGQ2 import LinformerModelHGQ2, LinformerHGQ2EmbeddingModel
    from tagger.model.TransformerModel import TransformerModel, TransformerEmbeddingModel
