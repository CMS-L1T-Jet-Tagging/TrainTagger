import json
import os

import hls4ml
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from HGQ import FreeBOPs, ResetMinMax, to_proxy_model, trace_minmax
from HGQ.layers import HConv1D, HConv1DBatchNorm, HDense, HQuantize, PAveragePooling1D, PFlatten, Signature

# Qkeras
from qkeras.utils import load_qmodel
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Activation

from tagger.data.tools import load_data, to_ML
from tagger.model.JetTagModel import JetModelFactory, JetTagModel

num_threads = 24
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = str(num_threads)

tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)

# GLOBAL PARAMETERS TO BE DEFINED WHEN TRAINING
tf.keras.utils.set_random_seed(420)  # not a special number


@JetModelFactory.register('DeepSetModelHGQ')
class DeepSetModelHGQ(JetTagModel):
    def __init__(self, output_dir):
        super().__init__(output_dir)

    def build_model(self, inputs_shape, outputs_shape):
        # Define a dictionary for common arguments

        beta = self.model_config["beta"]

        # Initialize inputs
        inputs = tf.keras.layers.Input(shape=inputs_shape, name='model_input')
        L, C = inputs_shape
        # Main branch
        main = HQuantize(name='quant1', beta=beta)(inputs)
        main = HConv1DBatchNorm(filters=10, activation='relu', kernel_size=1, beta=beta, parallel_factor=1, name='Conv1D_1')(
            main
        )
        main = HConv1D(filters=10, activation='relu', kernel_size=1, beta=beta, parallel_factor=1, name='Conv1D_2')(main)

        # Make Conv1D layers

        main = PAveragePooling1D(L, name='avgpool')(main)
        main = PFlatten()(main)

        # Now split into jet ID and pt regression
        jet_id = HDense(32, beta=beta, activation='relu', parallel_factor=1, name='Dense_1_jetID')(main)
        jet_id = HDense(16, name='Dense_2_jetID', beta=beta, activation='relu', parallel_factor=1)(jet_id)

        jet_id = HDense(outputs_shape[0], beta=beta, name='Dense_3_jetID')(jet_id)
        jet_id = Activation('softmax', name='act_jet')(jet_id)
        jet_id = Signature(bits=18, int_bits=8, keep_negative=0, name='jet_id_output')(jet_id)
        # pT regression branch
        pt_regress = HDense(10, name='Dense_1_pT', parallel_factor=1, beta=beta, activation='relu')(main)

        pt_regress = HDense(1, beta=beta, parallel_factor=1, name='pT_out')(pt_regress)
        pt_regress = Signature(
            bits=self.quantization_config['pt_output_quantization'][0],
            int_bits=self.quantization_config['pt_output_quantization'][1],
            keep_negative=0,
            name='pT_output',
        )(pt_regress)

        # Define the model using both branches
        self.jet_model = tf.keras.Model(inputs=inputs, outputs=[jet_id, pt_regress])

        print(self.jet_model.summary())

    def compile_model(self, num_samples):

        self.callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.training_config['EarlyStopping_patience']),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.training_config['ReduceLROnPlateau_factor'],
                patience=self.training_config['ReduceLROnPlateau_patience'],
                min_lr=self.training_config['ReduceLROnPlateau_min_lr'],
            ),
            ResetMinMax(),
            FreeBOPs(),
        ]

        self.jet_model.compile(
            optimizer='adam',
            loss={
                self.loss_name + self.output_id_name: 'categorical_crossentropy',
                self.loss_name + self.output_pt_name: tf.keras.losses.Huber(),
            },
            loss_weights=self.training_config['loss_weights'],
            metrics={
                self.loss_name + self.output_id_name: 'categorical_accuracy',
                self.loss_name + self.output_pt_name: ['mae', 'mean_squared_error'],
            },
            weighted_metrics={
                self.loss_name + self.output_id_name: 'categorical_accuracy',
                self.loss_name + self.output_pt_name: ['mae', 'mean_squared_error'],
            },
        )

    def fit(self, X_train, y_train, pt_target_train, sample_weight):
        self.history = self.jet_model.fit(
            {'model_input': X_train},
            {self.loss_name + self.output_id_name: y_train, self.loss_name + self.output_pt_name: pt_target_train},
            sample_weight=sample_weight,
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            verbose=self.run_config['verbose'],
            validation_split=self.training_config['validation_split'],
            callbacks=self.callbacks,
            shuffle=True,
        )

    @JetTagModel.save_decorator
    def save(self, out_dir):
        # Export the model
        model_export = tfmot.sparsity.keras.strip_pruning(self.jet_model)
        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        export_path = os.path.join(out_dir, "model/saved_model.h5")
        model_export.save(export_path)
        print(f"Model saved to {export_path}")

    @JetTagModel.load_decorator
    def load(self, out_dir=None):
        # Load model

        self.jet_model = load_qmodel(f"{out_dir}/model/saved_model.h5")

    def hls4ml_convert(self, firmware_dir, build=False):

        # Remove the old directory if they exist
        hls4ml_outdir = firmware_dir + '/' + self.hls4ml_config['project_name']
        os.system(f'rm -rf {hls4ml_outdir}')
        # Write HLS
        data_train, data_test, class_labels, input_vars, extra_vars = load_data("training_data/", percentage=10)
        X_train, y_train, pt_target_train, truth_pt_train, reco_pt_train = to_ML(data_train, class_labels)
        # Make into ML-like data for training
        # compute necessary bitwidth for each layer against a calibration dataset
        trace_minmax(self.jet_model, X_train, cover_factor=1.0)
        # convert HGQ model to a hls4ml-compatible proxy model
        proxy = to_proxy_model(self.jet_model, aggressive=False)

        # Create default config
        config = hls4ml.utils.config_from_keras_model(proxy, granularity='name')
        config['IOType'] = 'io_parallel'
        config['LayerName']['input_1']['Precision']['result'] = self.hls4ml_config['input_precision']

        # Configuration for conv1d layers
        # hls4ml automatically figures out the paralellization factor
        # config['LayerName']['Conv1D_1']['ParallelizationFactor'] = 8
        # config['LayerName']['Conv1D_2']['ParallelizationFactor'] = 8

        # Additional config

        config["LayerName"]["act_jet"]["Precision"]["result"] = self.hls4ml_config['class_precision']
        config["LayerName"]["act_jet"]["Implementation"] = "latency"
        config["LayerName"]["pT_out"]["Precision"]["result"] = self.hls4ml_config['class_precision']
        config["LayerName"]["pT_out"]["Implementation"] = "latency"

        self.hls_model = hls4ml.converters.convert_from_keras_model(
            proxy,
            backend='Vitis',
            project_name=self.hls4ml_config['project_name'],
            clock_period=2.8,  # 1/360MHz = 2.8ns
            hls_config=config,
            output_dir=f'{hls4ml_outdir}',
            part='xcvu9p-flga2104-2L-e',
        )

        # Compile and build the project
        self.hls_model.compile()

        # Save config  as json file
        print("Saving default config as config.json ...")
        with open(hls4ml_outdir + '/config.json', 'w') as fp:
            json.dump(config, fp)

        if build == True:
            self.hls_model.build(csim=False, reset=True)
