import tensorflow as tf
import keras
from config import Config


class TrainHandler:
    def __init__(self):
        self.conf = Config()
        self._create_model()

    def _create_model(self):
        image_size = self.conf.get_image_size()
        dense_list = self.conf.get_net_dense_list()
        conv_list = self.conf.get_net_conv_list()
        pooling = self.conf.get_pooling()
        dataset_name = self.conf.get_dataset_name()

        layers = []
        input_layer = keras.Input(shape=(image_size, image_size, 3))
        layers.append(input_layer)
        rescaling = keras.layers.Rescaling(1./255)
        layers.append(rescaling)

        for tuple in conv_list:
            num_filters = tuple[0]
            kernel_size = tuple[1]
            activation_func = tuple[2]
            hidden_layer = keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, activation=activation_func)
            layers.append(hidden_layer)
            if pooling == 'avg':
                pooling_layer = keras.layers.AveragePooling2D(pool_size=(2, 2))
            else:
                pooling_layer = keras.layers.MaxPooling2D(pool_size=(2, 2))
            layers.append(pooling_layer)
        flatten = keras.layers.Flatten()
        layers.append(flatten)
        for tuple in dense_list:
            num_nodes = tuple[0]
            activation_func = tuple[1]
            hidden_layer = keras.layers.Dense(units=num_nodes, activation=activation_func)
            layers.append(hidden_layer)
        if dataset_name == 'caltech':
            output_layer = keras.layers.Dense(units=102, activation=tf.nn.softmax)
        else:
            output_layer = keras.layers.Dense(units=100, activation=tf.nn.softmax)
        layers.append(output_layer)
        self.model = keras.Sequential(layers)
        self.model.summary()

    def train_model(self, train_set, test_set):
        loss_name = self.conf.get_loss_function()
        if loss_name == 'mse':
            loss = tf.losses.MeanSquaredError()
        elif loss_name == 'categorical_crossentropy':
            loss = tf.losses.CategoricalCrossentropy()
        elif loss_name == 'sparse_categorical_crossentropy':
            loss = tf.losses.SparseCategoricalCrossentropy()
        elif loss_name == 'hinge':
            loss = tf.losses.Hinge()
        elif loss_name == 'Huber':
            loss = tf.losses.Huber()
        elif loss_name == 'KLD':
            loss = tf.losses.KLDivergence()
        else:
            loss = tf.losses.MeanSquaredError()

        optimizer_name = self.conf.get_optimizer()
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.00003)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
        epochs = self.conf.get_num_epochs()
        
        for epoch in range(epochs):
            print("epoch: ", epoch+1, '/', epochs)
            if epoch > 0:
                self.model.load_weights('./model.h5')
            self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
            self.model.fit(train_set, epochs=1)
            self.model.save_weights('./model.h5')
            test_loss, test_acc = self.model.evaluate(test_set)
            print('test_Loss:', test_loss, ', test_Acc:', test_acc)




