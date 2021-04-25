import tensorflow as tf

import tensorflow.keras as keras
import numpy as np
import minist_model

for gpu in tf.config.experimental.list_physical_devices(device_type='GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


def train_substitute_model(minist_substitute_m, model_name, X, Y, num, epoch=10):
    (valide_x, valide_y), (_, _) = keras.datasets.mnist.load_data()
    valide_data = (valide_x, valide_y)
    train_y = Y
    train_x = X
    optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    minist_substitute_m.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    minist_substitute_m.fit(train_x, train_y, epochs=epoch, validation_data=valide_data,
                            callbacks=[myCallback(num, model_name)])


# 创建一个保存模型的回调
class myCallback(keras.callbacks.Callback):
    def __init__(self, num, name):
        self.model_name = name
        self.num = num

    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        with open('D:\\wf_temp\\' + str(self.num) + '_' + self.model_name + '.csv', 'a') as f:
            f.write(str(epoch) + ',' + str(acc) + ',' + str(val_acc) + '\n')


def get_random(num):
    i, X, Y = 0, [], []
    while i < 10:
        data_class = np.load(
            'C:\\Users\\ASUS-4\\Desktop\\吴峰\wf\\' + str(i) + '.npz')
        index = np.random.randint(low=0, high=20000, size=num)
        x, y = data_class['x'], data_class['y']
        sub_x, sub_y = x[index], y[index]
        X.extend(sub_x)
        Y.extend(sub_y)
        i += 1
    shuffle = np.arange(num * 10)
    np.random.shuffle(shuffle)
    X, Y = np.asarray(X), np.asarray(Y)
    X, Y = X[shuffle], Y[shuffle]
    return X, Y


if __name__ == '__main__':
    model_par = [[minist_model.SubstituteModel_mid().model(), "substitute_mnist_model_mid"]]
    num_list = [4000, 8000, 12000, 16000]
    for model_p in model_par:
        for num in num_list:
            X, Y = get_random(num)
            train_substitute_model(model_p[0], model_p[1], X, Y, num, epoch=200)
