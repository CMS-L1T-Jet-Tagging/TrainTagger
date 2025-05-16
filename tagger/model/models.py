from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
from tagger.plot.basic import loss_history, basic



class JetTagModel():
    def __init__(self,inputs_shape, outputs_shape,output_directory):
        self.inputs_shape = inputs_shape
        self.outputs_shape = outputs_shape
        self.output_directory = output_directory

        self.model = None

        self.hyperparameters = {'batch_size':1024,
                                'epochs':10,
                                'initial_sparsity':0.0,
                                'final_sparsity':0.1,
                                'validation_split':0.1}

        self.output_id_name = 'jet_id_output'
        self.output_pt_name = 'pT_output'
        self.loss_name = ''


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

    def plot_loss(self,out_dir=None):
        if out_dir is None:
          out_dir = self.output_directory

        #Produce some basic plots with the training for diagnostics
        plot_path = os.path.join(out_dir, "plots/training")
        os.makedirs(plot_path, exist_ok=True)

        #Plot history
        loss_history(plot_path, history)



# class logs(object):

#     _mlflow_file = 'mlflow_run_id.txt'

#     def __init__(self, func):
#         self.func = func

#     def __call__(self, *args):
#         log_string = self.func.__name__ + " was called"
#         print(log_string)
#         # Open the logfile and append
#         with open(self._logfile, 'a') as opened_file:
#             # Now we log to the specified logfile
#             opened_file.write(log_string + '\n')
#         # Now, send a notification
#         self.notify()

#         # return base func
#         return self.func(*args)



#     def notify(self):
#         # logit only logs, no more
#         pass