from config import Config
import tensorflow as tf


class DataHandler:
    def __init__(self):
        self.conf = Config()

    """
    def load_dataset(self, address):

        image_size = self.conf.get_image_size()

        images = []
        labels = []
        labels_name = []

        class_folder_list = os.listdir(address)
        for index, class_folder in enumerate(class_folder_list):
            image_address_list = os.listdir(address + '\\' + class_folder)
            for image_address in image_address_list:
                image = cv2.imread(address + '\\' + class_folder + '\\' + image_address)
                image = cv2.resize(image, [image_size, image_size])
                images.append(image)
                labels.append(index)

        X_train, X_test, y_train, y_test = self.split_data(images, labels)

        labels_name = class_folder_list
        return X_train, X_test, y_train, y_test, labels_name

    def split_data(self, X, y):
        X, y = shuffle(X, y)
        conf = Config()
        train_ratio = conf.get_train_ratio()
        num_train = int(np.ceil(len(X) * train_ratio))

        X_train = X[:num_train]
        X_test = X[num_train:]

        y_train = y[:num_train]
        y_test = y[num_train:]

        return X_train, X_test, y_train, y_test
    """

    def load_dataset(self):
        dataset_dir = self.conf.get_dataset_address()
        validation_split = 1 - self.conf.get_train_ratio()
        image_size = (self.conf.get_image_size(), self.conf.get_image_size())
        batch_size = self.conf.get_batch_size()

        train_set = tf.keras.utils.image_dataset_from_directory(
            dataset_dir,
            validation_split=validation_split,
            subset="training",
            seed=2020,
            image_size=image_size,
            batch_size=batch_size)

        test_set = tf.keras.utils.image_dataset_from_directory(
            dataset_dir,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=image_size,
            batch_size=batch_size)

        return train_set, test_set
