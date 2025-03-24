import numpy as np
import scipy.signal as signal
from scipy.io import loadmat
from scipy import interpolate
import os
from tqdm import tqdm
from Feature_Extraction import *
import matplotlib.pyplot as plt

def process_capture_data(base_dir, num_train=32, num_test=8, duration=600, interp_type='quadratic'):
    X_train = np.zeros((37 * 12 * num_train, 298, 3), dtype=np.float64)   #37表示37个动作，12表示单个个体12组，num_train表示个体的数目，400表示数据长度，3表示3个通道
    Y_train = np.zeros((37 * 12 * num_train,))
    X_test = np.zeros((37 * 12 * num_test, 298, 3), dtype=np.float64)
    Y_test = np.zeros((37 * 12 * num_test,))

    train_count = 0
    test_count = 0
    target_count = 0

    all_participants = np.arange(1, 41)
    train_participants =np.arange(1, 33)
    test_participants = np.arange(33, 41)

    # 定义滤波器参数
    fs = 200  # 采样频率
    cutoff = 10  # 截止频率
    order = 4  # 滤波器阶数



    for b in all_participants:
        print(f"Processing participant {b}")
        for folder in range(1, 13):
            print(f"Processing folder {folder}")
            capture_path = os.path.join(base_dir, str(b), str(folder), 'Optitrack', 'RigidBody1.mat')
            label_path = os.path.join(base_dir, str(b), str(folder), 'Optitrack', '{}TrajectoryLabels.mat'.format(folder))
            capturemat = loadmat(capture_path)
            labelmat = loadmat(label_path)
            capture_data = capturemat['data']
            labels = labelmat['Labels']




            # for i in range(1, 38):
            #     indices = np.where(labels == i)[0]
            #     capture_data_i = capture_data[indices]
            #     capture_signal = np.zeros((duration, 3))
            #
            #     if len(capture_data_i) <= duration:
            #         for j in range(0, 3):
            #             x = np.linspace(0, 1, len(capture_data_i))
            #             y = capture_data_i[:, j]
            #             f = interpolate.interp1d(x, y, kind=interp_type)
            #             xnew = np.linspace(0, 1, duration)
            #             capture_signal[:, j] = f(xnew)
            #
            #     else:
            #         capture_signal = capture_data_i[0:duration, :]

                    # 在这里添加绘图代码
                    # 您可以选择特定的动作进行绘图，或者绘制所有动作的数据
                    # 下面的代码将在每个文件夹中绘制每个动作的三个通道数据

                    # 定义您想要绘制的动作编号列表，如果想绘制所有动作，可以使用 range(1, 38)
            actions_to_plot = range(1, 38)  # 您可以修改为特定的动作编号列表，例如 [1, 2, 3]

            for i in actions_to_plot:
                indices = np.where(labels == i)[0]
                capture_data_i = capture_data[indices]

                if capture_data_i.size == 0:
                    continue  # 如果没有该动作的数据，跳过

                # 如果数据长度不足，进行插值；否则，截取指定长度
                if len(capture_data_i) <= duration:
                    capture_signal = np.zeros((duration, 3))
                    for j in range(3):
                        x = np.linspace(0, 1, len(capture_data_i))
                        y = capture_data_i[:, j]
                        f = interpolate.interp1d(x, y, kind=interp_type)
                        xnew = np.linspace(0, 1, duration)
                        capture_signal[:, j] = f(xnew)
                else:
                    capture_signal = capture_data_i[0:duration, :3]  # 只取前三个通道

                # # 绘制三个通道的动作数据
                # plt.figure(figsize=(12, 6))
                # time_axis = np.arange(capture_signal.shape[0]) / fs  # 时间轴（秒）
                # for channel in range(3):
                #     plt.plot(time_axis, capture_signal[:, channel], label=f'Channel {channel + 1}')
                # plt.title(f'Participant {b}, Folder {folder}, Action {i} - Three Channels Data')
                # plt.xlabel('Time (s)')
                # plt.ylabel('Amplitude')
                # plt.legend()
                # plt.tight_layout()
                # plt.show()



                for k in range(3):
                    window_length = int(0.025 * 200)         #滑动窗为5
                    step_size = window_length // 2                #滑动步长为5
                    num_windows = (duration - window_length) // step_size + 1      #窗口的个数
                    features = np.zeros((num_windows,))      #初始化feature的格式为(5, ),每5个sEMG信号特征(一个窗口)产生一个feature

                    # 加入截止频率为50Hz的高通滤波，由于采样频率为200Hz，所以信号段是在50Hz~200Hz
                    # emg_signal[:,k] = butter_highpass_filter(emg_signal[:,k], cutoff, fs, order=4)
                    # print(emg_signal[:,k].shape)

                    for m in range(num_windows):
                        start = m * step_size
                        end = start + window_length
                        segment = capture_signal[start:end, k]
                        # FFT and PSD
                        # frequencies, power_spectral_density = FFT(segment, fs)
                        # feature = frequency_ratio(power_spectral_density, frequencies,0,200)
                        feature = RMS(segment)
                        features[m] = feature

                    if b in train_participants:
                        X_train[train_count, :, k] = features
                    elif b in test_participants:
                        X_test[test_count, :, k] = features


                if b in train_participants:
                    Y_train[train_count] = i - 1
                    train_count += 1
                elif b in test_participants:
                    Y_test[test_count] = i - 1
                    test_count += 1


    save_dir = '/home/dongxu/pc_tmp/MyoData/'
    np.save(os.path.join(save_dir, 'CaptureData_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'CaptureData_label_train.npy'), Y_train)
    np.save(os.path.join(save_dir, 'CaptureData_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'CaptureDatalabel_test.npy'), Y_test)


    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)



if __name__ == "__main__":
    base_dir = '/home/dongxu/DataSet/'
    process_capture_data(base_dir)