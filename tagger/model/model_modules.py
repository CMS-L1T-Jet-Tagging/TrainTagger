"""
Registry of available jet tag models

If you want to add a new model:
1. Write a new model class that implements the functions of JetTagModel
2. Decorate that class with @JetModelFactory.register("MyModelName")
3. Write a yaml config with the key name: MyModelName
4. Add a line to the dictionary below of the form "MyModelName" : "tagger.model...."

Don't add any processing code or imports to this file
"""


MODEL_MODULES = {
        "DeepSetModel": "tagger.model.QKeras.DeepSetModel",
        "WeightedAverageSimpleModel": "tagger.model.QKeras.WeightedAverageSimpleModel",
        "DeepSetModelFloat": "tagger.model.Float.DeepSetModelFloat",
        "DeepSetModelHGQ2": "tagger.model.HGQ2.DeepSetModelHGQ2",
        "JEDILinearHGQ2": "tagger.model.HGQ2.JEDILinearHGQ2",
        "MLPMixerHGQ2": "tagger.model.HGQ2.MLPMixerHGQ2",
        "LinformerHGQ2": "tagger.model.HGQ2.LinformerHGQ2",
        "LEAFformerHGQ2": "tagger.model.HGQ2.LEAFformerHGQ2",
}