from data_handler import DataHandler
from config import Config
from train_handler import TrainHandler

"""
To run this project, go to config.py and set the parameters in __init__ function.
It is not necessary to do anything especial in other classes.
"""

if __name__ == "__main__":
    conf = Config()
    data_address = conf.get_dataset_address()

    data_handler = DataHandler()
    train_set, test_set = data_handler.load_dataset()

    train_handler = TrainHandler()
    train_handler.train_model(train_set, test_set)


