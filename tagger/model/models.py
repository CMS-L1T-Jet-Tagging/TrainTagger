

class JetTagModel():
    def __init__(self,inputs_shape, outputs_shape):
        self.inputs_shape = inputs_shape
        self.outputs_shape = outputs_shape

        self.model = None

        self.hyperparameters = {'batch_size':1024,
                                'epochs':100,
                                'initial_sparsity':0.0,
                                'final_sparsity':0.1}

        self.output_id_name = 'jet_id_output'
        self.output_pt_name = 'pT_output'


        self.callbacks = [EarlyStopping(monitor='val_loss', patience=10),
                          ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)]

    def build_model(self):
        pass

    def compile_model(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def hls4ml_convert(self):
        pass