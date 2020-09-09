import tensorflow as tf

tf.test.is_gpu_available()
import tensorflow.keras as keras
import numpy as np
from experiment import minist_model
from experiment import augmenting_images
import os
import glob
import gc

for gpu in tf.config.experimental.list_physical_devices(device_type='GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

substitute_mnist = [[] for _ in range(10)]


def create_noise(shape, noise_name, loc=0, scale=1, low=0, high=1):
    if noise_name == 'laplace':
        return np.random.laplace(loc=loc, scale=scale, size=shape)
    elif noise_name == 'uniform':
        return np.random.uniform(low=low, high=high, size=shape)
    elif noise_name == 'normal':
        return np.random.normal(loc=loc, scale=scale, size=shape)


class myCallback(keras.callbacks.Callback):
    def __init__(self, size):
        self.model_size = size

    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('accuracy')
        loss = logs.get('loss')
        val_acc = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')
        with open('data_record/epoch_acc_' + self.model_size + '.csv', 'a') as f:
            f.write(str(epoch) + ',' + str(acc) + ',' + str(val_acc) + '\n')
        with open('data_record/epoch_loss_' + self.model_size + '.csv', 'a') as f:
            f.write(str(epoch) + ',' + str(loss) + ',' + str(val_loss) + '\n')


class myCallback2(keras.callbacks.Callback):
    def __init__(self, acc):
        self.model_acc = acc

    def on_batch_end(self, batch, logs=None):
        acc = logs.get('accuracy')
        loss = logs.get('loss')
        val_acc = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')

        if acc >= self.model_acc:
            print('\nacc is enough,stop training')
            self.model.stop_training = True


def VGG16():
    return keras.applications.VGG16(include_top=True, weights='imagenet')


def VGG19():
    return keras.applications.VGG19(include_top=True, weights='imagenet')


def load_model(model_name):
    """
    用来加载本地模型

    :param model_name: 模型名称
    :return: 有模型返回模型，没有返回None
    """
    try:
        return keras.models.load_model(model_name)
    except(ImportError, IOError):
        return None


def save_result_in_npy(random_datas, result):
    """
    将随机采样结果保存为npy
    :param random_datas:
    :param result:
    :return:
    """
    counter = [0 for _ in range(10)]
    labels = []
    confidence = []
    for j in range(len(result)):
        cur = result[j]
        confidence.append(np.max(cur))
        index = np.where(cur == confidence[j])[0][0]
        labels.append(index)
        if counter[index] < 5:
            substitute_mnist[index].append(random_datas[j])
            counter[index] += 1
        print(index, len(substitute_mnist[index]), end='\r')

    # for i in range(len(random_datas)):
    #
    #     temp_lable = labels[i]
    #     if counter[temp_lable] >= 2 or len(substitute_mnist[temp_lable]) >= 20000:
    #         continue
    #     print(temp_lable, len(substitute_mnist[temp_lable]), end='\r')
    #     substitute_mnist[temp_lable].append(random_datas[i])
    #     counter[temp_lable] += 1


def my_shuffle(x=[], y=[], z=[]):
    temp_x, temp_y, temp_z = [], [], []
    for i in enumerate(x):
        temp_x.extend(x[i])
        temp_y.extend(y[i])
        temp_z.extend(z[i])
    indexs = np.random.randint(low=0, high=len(x))
    x = np.asarray(temp_x)[indexs]
    y = np.asarray(temp_y)[indexs]
    z = np.asarray(temp_z)[indexs]
    return x, y, z


model = keras.models.load_model(r'E:\PyCharm 2020.1.3\project\Model_extractor\models\original_minist_model.h5')


def sub_random_search(loc, scale, bh):
    random_datas = np.random.normal(loc=loc, scale=scale, size=[100, 28, 28, 1])
    result = model.predict(random_datas)
    save_result_in_npy(random_datas, result)
    for i in range(10):
        cur = np.asarray(substitute_mnist[i])
        if len(cur) == 0:
            continue
        path = "E:/PyCharm 2020.1.3/project/Model_extractor/substitute_mnist/_" + str(i) + '.npz'
        if not os.path.exists(path) or len(np.load(path)['x']) < 20000:
            temp_result = model(cur)
            pro = np.amax(temp_result, axis=1)
            np.savez("E:/PyCharm 2020.1.3/project/Model_extractor/substitute_mnist/" + str(i) + "_" + str(bh),
                     x=cur, y=[i for _ in range(len(cur))], pro=pro)
            del cur, temp_result, pro
            gc.collect()
    simplify_result(result, 10, loc=loc, scale=scale, write=True)
    print('loc=%.2f,scale=%.2f' % (loc, scale))


def random_search(loc=[-1, 1, 1], scale=[1, 5, 1], bh=0):
    """
    进行随机采样

    :param loc:
    :param scale:
    :return:
    """
    loc_start, loc_end, loc_step, scale_start, scale_end, scale_step = loc[0], loc[1], loc[2], scale[0], scale[1], \
                                                                       scale[2]
    while loc_start <= loc_end:
        while scale_start <= scale_end:
            sub_random_search(loc_start, scale_start, bh)
            scale_start += scale_step
            gc.collect()
        for i in range(10):
            combin(i)
        loc_start += loc_step
        scale_start = scale[0]
        gc.collect()


def train_minist_model(epochs=5, training=True):
    """
    使用minist数据集训练一个标准模型做为目标模型

    :param epochs:
    :return:
    """
    minist_m = minist_model.MinistModel().model()
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    train_x = train_x / 255.0
    test_x = test_x / 255.0
    optimizer = keras.optimizers.Adam(lr=0.002, beta_1=0.5)
    minist_m.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    minist_m.fit(train_x, train_y, epochs=epochs)
    minist_m.evaluate(test_x, test_y)
    minist_m.save(r'E:\PyCharm 2020.1.3\project\Model_extractor\models\original_minist_model.h5')
    return minist_m


def train_minist_model_by_num(x, y, num, epochs=5, training=True):
    """
    使用minist数据集训练一个标准模型做为目标模型

    :param epochs:
    :return:
    """
    (_, _), (valide_x, valide_y) = keras.datasets.mnist.load_data()
    valide_data = (valide_x, valide_y)
    minist_m = minist_model.MinistModel().model()
    x = x / 255.0
    optimizer = keras.optimizers.Adam(lr=0.002, beta_1=0.5)
    minist_m.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    minist_m.fit(x, y, validation_data=valide_data, epochs=epochs)
    minist_m.save(r'E:\PyCharm 2020.1.3\project\Model_extractor\models\original_minist_model' + str(num) + '.h5')
    return minist_m


def train_fashion_minist_model(epochs=5, training=True):
    """
    使用fashion_minist数据集训练一个标准模型做为目标模型

    :param epochs:
    :return:
    """
    minist_m = minist_model.MinistModel().model()
    (train_x, train_y), (test_x, test_y) = keras.datasets.fashion_mnist.load_data()
    train_x = train_x / 255.0
    test_x = test_x / 255.0
    optimizer = keras.optimizers.Adam(lr=0.002, beta_1=0.5)
    minist_m.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    minist_m.fit(train_x, train_y, epochs=epochs)
    minist_m.evaluate(test_x, test_y)
    minist_m.save(r'E:\PyCharm 2020.1.3\project\Model_extractor\models\original_fashion_minist_model.h5')
    return minist_m


def simplify_result(result, class_num, loc=0, scale=1, write=True):
    """
    简化模型输出结果并将其保存到文件中，如[0.1,0.2,0.5,0.2],[0.2,0.1,0.5,0.2]->[label,times]->[2,2],表示第二类出现了两次

    :param result:
    :param class_num:
    :param loc:
    :param scale:
    :return:
    """
    labels = []
    for j in range(len(result)):
        cur = result[j]
        index = np.where(cur == np.max(cur))[0][0]
        labels.append(index)
    label_unique = set([i for i in range(class_num)])
    label_times = list(map(labels.count, label_unique))
    print('label_unique', label_unique)
    print('label_times', label_times)
    if write:
        with open('search_record4.csv', 'a') as f:
            f.write('\nloc=%0.1f scale=%.1f,%s' % (loc, scale, str(label_times)[1:-1]))
            f.flush()


def under_sampling(datas, sampling_num):
    """
    欠采样
    :param datas: 数据
    :param sampling_num: 采样数量
    :return:
    """
    for index, data in enumerate(datas):
        if len(data) <= sampling_num:
            continue
        sample_indexs = np.random.randint(low=0, high=len(data), size=sampling_num)
        datas[index] = np.asarray(datas)[sample_indexs]
    return datas


def augmenting_dataset(dataset, total_num):
    """
    使用数据进行数据增强

    :param dataset: 要增强的数据集
    :param total_num: 数据集所要到达的数据数量
    :return:
    """
    temp_return = [[], [], []]
    augmentor = augmenting_images.AugmentingImg()
    temp = []
    for index, datas in enumerate(dataset):
        generate_num_per = np.floor(total_num / len(datas)).astype(int)
        if generate_num_per <= 0:
            dataset[index] = datas[:total_num]
            continue
        for img in datas:
            aug_imgs = augmentor.augment(img, [28, 28], generate_num_per)
            temp.extend(aug_imgs)
            if len(temp) >= total_num - len(datas):
                break
        # o_model = keras.models.load_model(
        #     r'E:\PyCharm 2020.1.3\project\Model_extractor\models\original_minist_model.h5')
        # truelabel = o_model(img.reshape(-1, 28, 28, 1))
        # truelabel = np.where(truelabel == np.max(truelabel))[1][0]
        # result = test_acc(
        #     o_model,
        #     np.asarray(temp).reshape(-1, 28, 28, 1))
        # labels = []
        # for j in range(len(result)):
        #     cur = result[j]
        #     index = np.where(cur == np.max(cur))[0][0]
        #     labels.append(index)
        # label_unique = set(labels)
        # label_times = list(map(labels.count, label_unique))
        # print('true_lable', truelabel)
        # print('label_unique', label_unique)
        # print('label_times', label_times)
        temp_return[index].extend(temp)
        dataset[index].extend(temp)
        temp = []
    return dataset, temp_return


def test_acc(original_model, testdata):
    result = original_model(testdata)
    return result


def get_result_on_other_dataset(model, dataset, batch=5000, class_num=10):
    """
    使用一个正常的模型在本不是他该预测的数据集上进行预测，获取输出，这时我们会得到一组（x,pre_y）数据，再使用这组数据去训练我们的substitute模型

    :param model:我们要使用substitute模型进行拟合的模型，即目标模型
    :param dataset:我们要让目标模型进行预测的数据集
    :param batch:由于内存限制，默认每个预测batch为5000
    :param class_num:目标模型的分类数量
    :return: 新的用来训练substitute的训练集（x,pre_y）
    """
    (data_x, _), (_, _) = dataset
    data_x = data_x / 255.0
    records = [[] for _ in range(class_num)]
    epochs = int(len(data_x) / batch)
    for i in range(epochs):
        result = model(data_x[batch * i:batch * (i + 1)])
        for j in range(len(result)):
            cur = result[j]
            index = np.where(cur == np.max(cur))[0][0]
            records[index].append(data_x[i * batch + j])
        print('已获取：%d/%d' % (i + 1, epochs))
    # 数据增强，裁减数据解决数据分布不均匀的问题
    records = augmenting_dataset(records, 500)
    # 去归一化，返回一个标准的数据集
    new_x = [item * 255.0 for record in records for item in record]
    new_y = [index for index, record in enumerate(records) for _ in range(len(record))]
    shuffle_index = np.arange(len(new_x))
    np.random.shuffle(shuffle_index)
    new_x = np.asarray(new_x)[shuffle_index]
    new_y = np.asarray(new_y)[shuffle_index]
    return new_x, new_y


def train_minist_substitute_model(pre_data=keras.datasets.fashion_mnist.load_data(), epochs=5, callback=[]):
    """
    训练替代模型

    :param pre_data: 要用来训练的数据集
    :param epochs: 训练的epoch
    :param callback: 训练时的函数回调
    :return: 训练集和替代模型
    """
    original_model = keras.models.load_model(
        r'E:\PyCharm 2020.1.3\project\Model_extractor\models\original_minist_model.h5')
    minist_substitute_m = minist_model.SubstituteModel_mid().model()
    # 使用minist数据集正常训练出来的model去识别fashion_minist
    (train_x, train_y) = get_result_on_other_dataset(original_model, pre_data)
    optimizer = keras.optimizers.Adam(lr=0.002, beta_1=0.5)
    minist_substitute_m.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    minist_substitute_m.fit(train_x, train_y, epochs=epochs, callbacks=callback)
    minist_substitute_m.save(r'E:\PyCharm 2020.1.3\project\Model_extractor\models\substitute_minist_model.h5')
    return (train_x, train_y), minist_substitute_m


# 计算x方向及y方向相邻像素差值，如果有高频花纹，这个值肯定会高，
def high_pass_x_y(image):
    x_var = image[:, 1:, :] - image[:, :-1, :]
    y_var = image[1:, :, :] - image[:-1, :, :]

    return x_var, y_var


# 计算总体变分损失
def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_mean(x_deltas ** 2) + tf.reduce_mean(y_deltas ** 2)


def evaluate(model, datas):
    for index, data in enumerate(datas):
        imgs = np.asarray(data).reshape(-1, 28, 28, 1)
        result = model(imgs)
        save_result_in_npy(imgs, result)
        simplify_result(result, class_num=3, write=False)


def combin(num):
    path = 'E:/PyCharm 2020.1.3/project/Model_extractor/substitute_mnist/pre/' + str(num) + '*.npz'
    files = glob.glob(path)
    combin_x, combin_y, combin_pro = [], [], []
    # 组合
    for f in files:
        datas = np.load(f)
        x, y, pro = datas['x'], datas['y'], datas['pro']
        combin_x.extend(x)
        combin_y.extend(y)
        combin_pro.extend(pro)
    # 保存
    np.savez('E:/PyCharm 2020.1.3/project/Model_extractor/substitute_mnist/pre/f_' + str(num),
             x=combin_x[:20000], y=combin_y[:20000], pro=combin_pro[:20000])


def fillter():
    path = 'E:\\PyCharm 2020.1.3\\project\\Model_extractor\\substitute_mnist\\*.npz'
    list = glob.glob(path)
    for l in list:
        temp_x, temp_y, temp_p = [], [], []
        datas = np.load(l)
        x, y, p = datas['x'], datas['y'], datas['pro']
        for i in range(len(p)):
            if p[i] >= 0.99:
                temp_x.append(x[i])
                temp_y.append(y[i])
                temp_p.append(p[i])
        np.savez(l, x=temp_x, y=temp_y, pro=temp_p)


def train_substitute_model(minist_substitute_m, model_name, model_size, data_path, is_weight, epoch=10):
    # minist_substitute_m = keras.models.load_model(
    #     r'E:\PyCharm 2020.1.3\project\Model_extractor\models\substitute_fashion_minist_model.h5')
    datas = np.load(data_path)
    (valide_x, valide_y), (_, _) = keras.datasets.mnist.load_data()
    valide_data = (valide_x, valide_y)
    train_x, train_y, pro = datas['x'], datas['y'], datas['pro']
    indexs = np.arange(len(train_y))
    np.random.shuffle(indexs)
    train_y = train_y[indexs]
    train_x = train_x[indexs]
    pro = pro[indexs]
    if is_weight:
        pro = 1.1 - pro
    else:
        pro = np.ones_like(pro)
    optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    minist_substitute_m.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    minist_substitute_m.fit(train_x, train_y, epochs=epoch, validation_data=valide_data,
                            callbacks=myCallback(model_size), sample_weight=pro)
    minist_substitute_m.save('E:/PyCharm 2020.1.3/project/Model_extractor/models/' + model_name + '.h5')


def combin_all():
    path = r'E:/PyCharm 2020.1.3/project/Model_extractor/substitute_fashion_mnist/*.npz'
    files = glob.glob(path)
    file_num = len(files)
    combin_x, combin_y, combin_pro = [], [], []
    for i in range(file_num):
        datas = np.load(files[i])
        x, y, pro = datas['x'], datas['y'], datas['pro']
        indexs = np.arange(20000)
        np.random.shuffle(indexs)
        combin_x.extend(x[indexs])
        combin_y.extend(y[indexs])
        combin_pro.extend(pro[indexs])
    indexs = np.arange(len(combin_y))
    np.random.shuffle(indexs)
    combin_x = np.asarray(combin_x)[indexs]
    combin_y = np.asarray(combin_y)[indexs]
    combin_pro = np.asarray(combin_pro)[indexs]
    np.savez(path[:-5] + "substitute_mnist", x=combin_x, y=combin_y, pro=combin_pro)


def combin_all_temp():
    path = r'E:/PyCharm 2020.1.3/project/Model_extractor/substitute_fashion_mnist/*.npz'
    files = glob.glob(path)
    file_num = len(files)
    combin_x, combin_y, combin_pro = [], [], []
    for i in range(file_num):
        if i == 9:
            (x, y), (x_, y_) = keras.datasets.fashion_mnist.load_data()
            x = np.append(x, x_, axis=0)
            y = np.append(y, y_, axis=0)
            index = np.where(y == 9)
            x = x[index]
            y = y[index]
            pro = np.ones_like(y)
            combin_x.extend(x.reshape(-1, 28, 28, 1))
            combin_y.extend(y)
            combin_pro.extend(pro)
        else:
            datas = np.load(files[i])
            x, y, pro = datas['x'], datas['y'], datas['pro']
            indexs = np.arange(20000)
            np.random.shuffle(indexs)
            combin_x.extend(x[indexs])
            combin_y.extend(y[indexs])
            combin_pro.extend(pro[indexs])
    indexs = np.arange(len(combin_y))
    np.random.shuffle(indexs)
    combin_x = np.asarray(combin_x)[indexs]
    combin_y = np.asarray(combin_y)[indexs]
    combin_pro = np.asarray(combin_pro)[indexs]
    np.savez(path[:-5] + "substitute_mnist", x=combin_x, y=combin_y, pro=combin_pro)


def tc():
    (x, y), (x1, y1) = keras.datasets.fashion_mnist.load_data()
    x = np.append(x, x1, axis=0)
    y = np.append(y, y1, axis=0)
    tc_index = np.where(y == 9)[0]
    y = np.delete(y, tc_index, 0)
    x = np.delete(x, tc_index, 0)
    return x, y


def split_by_num(num):
    (x, y), (_, _) = keras.datasets.mnist.load_data()
    result = []
    label = []
    # result = x
    # label = y
    for i in range(10):
        temp_index = np.where(y == i)[0]
        shuffle_index = np.arange(len(temp_index))
        np.random.shuffle(temp_index)
        temp_x = x[temp_index][:]
        result.extend(temp_x)
        label.extend([i for _ in range(len(temp_x))])
    shuffle_index = np.arange(len(label))
    np.random.shuffle(shuffle_index)
    result = np.asarray(result)[shuffle_index]
    label = np.asarray(label)[shuffle_index]
    return result, label


def test_model_on_orginal_data(model_path, data_name):
    model = keras.models.load_model(model_path)
    if data_name == 'mnist':
        (x, y), (x_, y_) = keras.datasets.mnist.load_data()
    elif data_name == 'fashion_mnist':
        (x, y), (x_, y_) = keras.datasets.fashion_mnist.load_data()
    elif data_name == 'cifar10':
        (x, y), (x_, y_) = keras.datasets.cifar10.load_data()
    model.evaluate(x, y)
    model.evaluate(x_, y_)


if __name__ == '__main__':
    # accs = [0.9, 0.8, 0.7, 0.6]
    # (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    # train_x = train_x / 255.0
    # test_x = test_x / 255.0
    # for acc in accs:
    #     model = minist_model.TinyMinistModel().model()
    #     model.compile(optimizer=keras.optimizers.Adam(lr=0.002, beta_1=0.5),
    #                   loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #     model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=100, callbacks=myCallback2(acc))
    #     model.save('E:/PyCharm 2020.1.3/project/Model_extractor/models/original_mnist_model_acc' + str(acc) + '.h5')
    # num = 2000
    # x, y = split_by_num(num)
    # train_minist_model_by_num(x, y, num, epochs=50)
    #
    # # model_path = r'E:\PyCharm 2020.1.3\project\Model_extractor\models\original_fashion_minist_model_.h5'
    # # test_model_on_orginal_data(model_path, 'fashion_mnist')
    # model_par = [
    #     [minist_model.SubstituteModel_small().model(), "substitute_mnist_model_small_probability", 'small_probability'],
    #     [minist_model.SubstituteModel_mid().model(), "substitute_mnist_model_mid_probability", 'mid_probability'],
    #     [minist_model.SubstituteModel_large().model(), "substitute_mnist_model_large_probability", 'large_probability']]
    #
    # model_par1 = [[minist_model.SubstituteModel_small().model(), "substitute_mnist_model_small_label", 'small_label'],
    #               [minist_model.SubstituteModel_mid().model(), "substitute_mnist_model_mid_label", 'mid_label'],
    #               [minist_model.SubstituteModel_large().model(), "substitute_mnist_model_large_label", 'large_label']]
    # for model_p in model_par1:
    #     # print(model_p[1])
    #     # model_path = 'E:/PyCharm 2020.1.3/project/Model_extractor/models/' + model_p[1] + '.h5'
    #     # test_model_on_orginal_data(model_path, 'mnist')
    #     train_substitute_model(model_p[0], model_p[1], model_p[2],
    #                            r'E:\PyCharm 2020.1.3\project\Model_extractor\substitute_mnist\substitute_mnist.npz',
    #                            False,
    #                            epoch=50)
    # tc()
    # combin_all_temp()
    # fillter()
    # for i in [0, 4, 9]:
    #     combin(i)
    # combin_all()
    # keras.preprocessing.image.save_img('s.png', )
    # original_model = keras.models.load_model(
    #     r'E:\PyCharm 2020.1.3\project\Model_extractor\models\original_minist_model.h5')
    # xx=np.load(r'E:\PyCharm 2020.1.3\project\Model_extractor\substitute_mnist\4.npz')
    # x=xx['x']
    # pro=xx['pro'][:10]
    # y=original_model(x)[:10]
    start = -8
    while True:
        random_search(loc=[start, 1, 1], scale=[0, 100, 1], bh=18)
        start = -50
        gc.collect()
    print('done!')
    # original_mnist_model = train_minist_model(epochs=5, training=False)
    # original_fashion_model = train_fashion_minist_model(epochs=5, training=False)
    # (train_x, train_y), minist_substitute_m = train_minist_substitute_model(epochs=20000, callback=[myCallback()])
    # train_substitute_model(epoch=30)
    # train_substitute_model()
    # train_fashion_minist_model(epochs=30)
    # train_minist_model()
    # minist_substitute_m = keras.models.load_model(
    #     r'E:\PyCharm 2020.1.3\project\Model_extractor\models\substitute_fashion_minist_model.h5')
    # datas = np.load(
    #     r'E:\PyCharm 2020.1.3\project\Model_extractor\substitute_fashion_mnist\substitute_fashion_mnist.npz')
    # train_x, train_y, pro = datas['x'], datas['y'], datas['pro']
    # # minist_substitute_m.evaluate(train_x, train_y)
    # # (x_, y_), (x1_, y1_) = keras.datasets.fashion_mnist.load_data()
    # # x_ = x_
    # # x1_ = x1_
    # x_, y_ = tc()
    # # minist_substitute_m.evaluate(x_, y_)
    # # minist_substitute_m.evaluate(x1_, y1_)
    # result = minist_substitute_m.predict(x_)
    # lables = []
    # for r in result:
    #     lable = np.where(r == np.max(r))[0][0]
    #     lables.append(lable)
    # zero_indexs = np.where(np.asarray(lables) == 2)[0]
    # zero_indexs_ = np.where(y_ ==2)[0]
    # n = 0
    # for index in zero_indexs_:
    #     if zero_indexs.__contains__(index):
    #         n += 1
    # for i in range(len(lables)):
    #     if lables[i] == y_[i]:
    #         print(lables[i], end=' ')
    # minist_substitute_m.evaluate(x_, y_)
    # minist_substitute_m.evaluate(x1_, y1_)
