import os


class Config:
    def __init__(self):
        self.code_address = os.getcwd() + '/'
        self.dataset_address = self.code_address + 'Dataset/'
        self.dataset_name = 'cifar'  # 'caltech' or 'cifar'
        self.train_ratio = 0.9  # 0.7, 0.9

        self.conv_properties = [(64, 3, 'relu'), (128, 3, 'relu'), (64, 3, 'relu')]  # you can add/remove conv layers
        self.dense_properties = [(256, 'relu')]  # you can add more dense layers
        self.pooling = 'max'  # 'max' or 'avg'
        self.loss = 'sparse_categorical_crossentropy'  # 'mse', 'sparse_categorical_crossentropy', 'hinge'
        self.optimizer = 'adam'  # 'sgd' or 'adam'
        self.epoch = 10  # 10, 20, 100
        self.batch_size = 16  # 32, 48, 16

    def get_code_address(self):
        return self.code_address

    def get_dataset_address(self):
        return self.dataset_address + self.dataset_name + '/'

    def get_dataset_name(self):
        return self.dataset_name

    def get_image_size(self):
        if self.get_dataset_name() == 'caltech':
            image_size = 256
        else:
            image_size = 32
        return image_size

    def get_train_ratio(self):
        return self.train_ratio

    def get_net_dense_list(self):
        return self.dense_properties

    def get_net_conv_list(self):
        return self.conv_properties

    def get_loss_function(self):
        return self.loss

    def get_optimizer(self):
        return self.optimizer

    def get_num_epochs(self):
        return self.epoch

    def get_batch_size(self):
        return self.batch_size

    def get_pooling(self):
        return self.pooling
