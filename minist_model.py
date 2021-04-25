import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.layers as layers


class MinistModel(keras.Model):
    def __init__(self):
        super(MinistModel, self).__init__()
        self.conv1 = layers.Conv2D(16, kernel_size=3, strides=2, input_shape=(28, 28, 1), padding="same")
        self.drop = layers.Dropout(0.25)

        self.conv2 = layers.Conv2D(32, kernel_size=3, strides=2, padding="same")
        self.drop1 = layers.Dropout(0.25)
        self.bn = layers.BatchNormalization(momentum=0.8)
        self.conv3 = layers.Conv2D(10, kernel_size=3, strides=2, padding="same")
        self.drop2 = layers.Dropout(0.25)
        self.bn2 = layers.BatchNormalization(momentum=0.8)
        self.gap = layers.GlobalAveragePooling2D()
        self.fc3 = layers.Dense(10)

    def call(self, inputs, training=None, mask=None):
        inputs = tf.reshape(inputs, (-1, 28, 28, 1))
        x = self.drop(tf.nn.leaky_relu(self.conv1(inputs), alpha=0.2))
        x = self.bn(self.drop1(tf.nn.leaky_relu(self.conv2(x), alpha=0.2)))
        x = self.bn2(self.drop2(tf.nn.leaky_relu(self.conv3(x), alpha=0.2)))
        x = self.gap(x)
        category = tf.nn.softmax(self.fc3(x))
        return category

    def model(self):
        x = layers.Input(shape=(28, 28, 1))
        return keras.Model(inputs=x, outputs=self.call(x))


class SubstituteModel_mid(keras.Model):
    def __init__(self):
        super(SubstituteModel_mid, self).__init__()
        self.conv1 = layers.Conv2D(16, kernel_size=3, strides=2, input_shape=(28, 28, 1), padding="same")
        self.drop = layers.Dropout(0.25)

        self.conv2 = layers.Conv2D(32, kernel_size=3, strides=2, input_shape=(28, 28, 1), padding="same")
        self.drop1 = layers.Dropout(0.25)
        self.bn = layers.BatchNormalization(momentum=0.8)

        self.conv3 = layers.Conv2D(64, kernel_size=3, strides=2, padding="same")
        self.drop2 = layers.Dropout(0.25)
        self.bn2 = layers.BatchNormalization(momentum=0.8)

        self.conv4 = layers.Conv2D(128, kernel_size=3, strides=1, padding="same")
        self.drop3 = layers.Dropout(0.25)

        self.conv5 = layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
        self.drop4 = layers.Dropout(0.25)
        self.bn3 = layers.BatchNormalization(momentum=0.8)

        self.gap = layers.GlobalAveragePooling2D()
        self.fc3 = layers.Dense(10)

    def call(self, inputs, training=None, mask=None):
        inputs = tf.reshape(inputs, (-1, 28, 28, 1))
        x = self.drop(tf.nn.leaky_relu(self.conv1(inputs), alpha=0.2))
        x = self.bn(self.drop1(tf.nn.leaky_relu(self.conv2(x), alpha=0.2)))
        x = self.bn2(self.drop2(tf.nn.leaky_relu(self.conv3(x), alpha=0.2)))
        x = self.drop3(tf.nn.leaky_relu(self.conv4(x), alpha=0.2))
        x = self.bn3(self.drop4(tf.nn.leaky_relu(self.conv5(x), alpha=0.2)))
        x = self.gap(x)
        category = tf.nn.softmax(self.fc3(x))
        return category

    def model(self):
        x = layers.Input(shape=(28, 28, 1))
        return keras.Model(inputs=x, outputs=self.call(x))
