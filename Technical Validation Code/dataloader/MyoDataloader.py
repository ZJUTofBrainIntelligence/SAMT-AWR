import numpy as np
import scipy.signal as signal
from scipy.io import loadmat
from scipy import interpolate
import os
from scipy.signal import butter, filtfilt
from dataloader.Feature_Extraction import *


def process_emg_data(base_dir, target_participants, num_train=28, num_test=9, duration=2000, interp_type='quadratic'):
    X_train = np.zeros((37 * 12 * num_train, 400, 8), dtype=np.float64)   #37表示37个动作，12表示单个个体12组，num_train表示个体的数目，400表示数据长度，8表示8个通道
    Y_train = np.zeros((37 * 12 * num_train,))
    X_test = np.zeros((37 * 12 * num_test, 400, 8), dtype=np.float64)
    Y_test = np.zeros((37 * 12 * num_test,))

    target_data_exists = os.path.exists(os.path.join(base_dir, 'emg_target.npy')) and os.path.exists(
        os.path.join(base_dir, 'label_target.npy'))

    if not target_data_exists:
        X_target = np.zeros((37 * 12 * len(target_participants), 400, 8), dtype=np.float64)
        Y_target = np.zeros((37 * 12 * len(target_participants),))
    else:
        X_target = None
        Y_target = None

    train_count = 0
    test_count = 0
    target_count = 0

    all_participants = np.arange(5, 51)
    remaining_participants = [p for p in all_participants if p not in target_participants]
    np.random.shuffle(remaining_participants)

    train_participants = remaining_participants[:num_train]
    test_participants = remaining_participants[num_train:num_train + num_test]

    # 定义滤波器参数
    fs = 200  # 采样频率
    cutoff = 10  # 截止频率
    order = 4  # 滤波器阶数



    for b in all_participants:
        print(f"Processing participant {b}")
        for folder in range(1, 13):
            emg_path = os.path.join(base_dir, str(b), str(folder), 'emg.mat')
            label_path = os.path.join(base_dir, str(b), str(folder), '{}EMGLabels.mat'.format(folder))
            emgmat = loadmat(emg_path)
            labelmat = loadmat(label_path)
            emg_data = emgmat['emgArray']
            labels = labelmat['Labels']


            for i in range(1, 38):
                indices = np.where(labels == i)[0]
                emg_data_i = emg_data[indices]
                emg_signal = np.zeros((duration, 8))

                if len(emg_data_i) <= duration:
                    for j in range(0, 8):
                        x = np.linspace(0, 1, len(emg_data_i))
                        y = emg_data_i[:, j]
                        f = interpolate.interp1d(x, y, kind=interp_type)
                        xnew = np.linspace(0, 1, duration)
                        emg_signal[:, j] = f(xnew)

                else:
                    emg_signal = emg_data_i[0:duration, :]


                for k in range(8):
                    window_length = int(0.025 * 200)         #滑动窗为5
                    step_size = window_length                #滑动步长为5
                    num_windows = (duration - window_length) // step_size + 1      #窗口的个数
                    features = np.zeros((num_windows,))      #初始化feature的格式为(5, ),每5个sEMG信号特征(一个窗口)产生一个feature

                    # 加入截止频率为50Hz的高通滤波，由于采样频率为200Hz，所以信号段是在50Hz~200Hz
                    # emg_signal[:,k] = butter_highpass_filter(emg_signal[:,k], cutoff, fs, order=4)
                    # print(emg_signal[:,k].shape)

                    for m in range(num_windows):
                        start = m * step_size
                        end = start + window_length
                        segment = emg_signal[start:end, k]
                        feature = RMS(segment)
                        features[m] = feature

                    if b in train_participants:
                        X_train[train_count, :, k] = features
                    elif b in test_participants:
                        X_test[test_count, :, k] = features
                    elif not target_data_exists and b in target_participants:
                        X_target[target_count, :, k] = features

                if b in train_participants:
                    Y_train[train_count] = i - 1
                    train_count += 1
                elif b in test_participants:
                    Y_test[test_count] = i - 1
                    test_count += 1
                elif not target_data_exists and b in target_participants:
                    Y_target[target_count] = i - 1
                    target_count += 1

    save_dir = '/home/dongxu/pc_tmp/MyoData/'
    np.save(os.path.join(save_dir, 'emg_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'label_train.npy'), Y_train)
    np.save(os.path.join(save_dir, 'emg_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'label_test.npy'), Y_test)

    if not target_data_exists:
        np.save(os.path.join(save_dir, 'emg_target.npy'), X_target)
        np.save(os.path.join(save_dir, 'label_target.npy'), Y_target)

    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)

    if not target_data_exists:
        print("X_target shape:", X_target.shape)
        print("Y_target shape:", Y_target.shape)




if __name__ == "__main__":
    base_dir = '/home/dongxu/MyoData/'
    # target_participants = np.arange(42, 51)  # 选择目标域
    # process_emg_data(base_dir, target_participants)
    process_emg_data(base_dir)
