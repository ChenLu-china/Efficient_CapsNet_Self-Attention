import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
from utils import process_multimnist
import json


class Dataset(object):
    """
    A class used to share common dataset functions and attributes.

    ...

    Attributes
    ----------
    model_name: str
        name of the model (Ex. 'MNIST')
    config_path: str
        path configuration file

    Methods
    -------
    load_config():
        load configuration file
    get_dataset():
        load the dataset defined by model_name and pre_process it
    get_tf_data():
        get a tf.data.Dataset object of the loaded dataset.
    """

    def __init__(self, model_name, config_path='config.json'):
        self.model_name = model_name
        self.config_path = config_path
        self.config = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.class_names = None
        self.X_test_patch = None
        self.load_config()
        self.get_dataset()

    def load_config(self):
        """
        Load config file
        """
        with open(self.config_path) as json_data_file:
            self.config = json.load(json_data_file)

    def get_dataset(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = tf.keras.datasets.mnist.load_data(
            path=self.config['mnist_path'])
        # prepare the data
        self.X_train = process_multimnist.pad_dataset(self.X_train, self.config["pad_multimnist"])
        self.X_test = process_multimnist.pad_dataset(self.X_test, self.config["pad_multimnist"])
        self.X_train, self.y_train = process_multimnist.pre_process(self.X_train, self.y_train)
        self.X_test, self.y_test = process_multimnist.pre_process(self.X_test, self.y_test)
        self.class_names = list(range(10))
        print("[INFO] Dataset loaded!")

    def get_tf_data(self):
        dataset_train, dataset_test = process_multimnist.generate_tf_data(self.X_train, self.y_train,
                                                                          self.X_test, self.y_test,
                                                                          self.config['batch_size'],
                                                                          self.config["shift_multimnist"])
        return dataset_train, dataset_test