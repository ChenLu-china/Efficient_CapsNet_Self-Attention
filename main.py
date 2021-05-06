# This is a sample Python script.
from utils.dataset import Dataset
from Model.model import EfficientCapsNet
import tensorflow as tf
import numpy as np

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(tf.__version__)
    model_name = 'MULTIMNIST'
    dataset = Dataset(model_name, config_path='config.json')
    model_train = EfficientCapsNet(model_name, 'train', verbose=True)
    history = model_train.train(dataset)
    print(history)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
