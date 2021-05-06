from Model import E_C_MultiMnist
from utils.tools import MultiAccuracy, marginLoss
from utils import process_multimnist
import tensorflow as tf
import numpy as np
import json
from tqdm.notebook import tqdm
import os
from utils.dataset import Dataset


class Model(object):
    """

    """

    def __init__(self, model_name, mode='test', config_path='config.json', verbose=True):
        self.model_name = model_name
        self.model = None
        self.mode = mode
        self.config_path = config_path
        self.config = None
        self.verbose = verbose
        self.load_config()

    def load_config(self):
        with open(self.config_path) as json_load_file:
            self.config = json.load(json_load_file)

    def predict(self, dataset_any):
        return self.model.predict(dataset_any)

    def evaluate(self, X_dataset, Y_dataset):
        print('=' * 40 + f'{self.model_name} Evaluation' + '=' * 40)
        dataset_test = process_multimnist.generate_tf_data_test(X_dataset, Y_dataset, self.config["shift_multimnist"],
                                                                n_multi=self.config['n_overlay_multimnist'])
        acc = []
        for X, y in tqdm(dataset_test, total=len(X_dataset)):
            y_pred, X_gen1, X_gen2 = self.model.predict(X)
            acc.append(MultiAccuracy(y, y_pred))
        acc = np.mean(acc)

        test_error = 1 - acc
        print("Test acc", acc)
        print(f"Test error [%]:{(test_error):.4%}")

    def save_graph_weights(self):
        self.model.save_weights(self.model_path)


class EfficientCapsNet(Model):
    """

    """

    def __init__(self, model_name, mode='test', config_path='config.json', custom_path=None, verbose=True):
        Model.__init__(self, model_name, mode, config_path, verbose)
        if custom_path != None:
            self.model_path = custom_path
        else:
            self.model_path = os.path.join(self.config['saved_model_dir'], f"efficient_capsnet_{self.model_name}.h5")
        self.model_path_new_train = os.path.join(self.config['saved_model_dir'],
                                                 f"efficient_capsnet_{self.model_name}_new_train.h5")
        self.tb_path = os.path.join(self.config['tb_log_save_dir'], f"efficient_capsnet_{self.model_name}")
        self.load_graph()

    def load_graph(self):
        self.model = E_C_MultiMnist.build_graph(self.config['MULTIMNIST_INPUT_SHAPE'], mode='train', verbose=True)

    def train(self, dataset=None, initial_epoch=0):

        if dataset is None:
            dataset = Dataset(self.model_name, self.config_path)
        dataset_train, dataset_val = dataset.get_tf_data()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
                           loss=[marginLoss, 'mse', 'mse'],
                           loss_weights=[1., self.config['lmd_gen'] / 2, self.config['lmd_gen'] / 2],
                           metrics={'Efficient_CapsNet': 'accuracy'})
        steps = 10 * int(dataset.y_train.shape[0] / self.config['batch_size'])

        print('-' * 30 + f'{self.model_name} train' + '-' * 30)

        history = self.model.fit(dataset_train,
                                 epochs=self.config['epoch'],
                                 steps_per_epoch= steps,
                                 validation_data=(dataset_val),
                                 batch_size=self.config['batch_size'],
                                 initial_epoch=initial_epoch)
        return history
