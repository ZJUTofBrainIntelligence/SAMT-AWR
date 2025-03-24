import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.utils.data as data
from matplotlib import pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上一级目录
parent_dir = os.path.dirname(current_dir)

# 将上一级目录添加到sys.path中
sys.path.append(parent_dir)

from networks.Optitrack_Models import *
# Function for moving tensor or model to GPU or CPU
def cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]
    else:
        return xs



def train(model, train_loader, criterion, optimizer, device):
    model.train()  # 将模型设置为训练模式

    total = 0
    correct = 0
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据和标签移动到指定的设备

        optimizer.zero_grad()  # 清空之前的梯度

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)  # 计算损失

        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

        # 记录性能指标
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

    train_acc = 100 * correct / total  # 计算训练准确率
    return running_loss / len(train_loader), train_acc  # 返回平均损失和训练准确率
# 定义测试函数
def test(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    all_pred = []
    all_true = []
    all_outputs = []
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            all_true.extend(labels.cpu().tolist())
            outputs = model(images)
            all_outputs.append(outputs)

            _, predicted = torch.max(outputs.data, 1)
            all_pred.extend(predicted.cpu().tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            running_loss += loss.item()

    all_outp = torch.cat(all_outputs).cpu().numpy()
    test_acc = 100 * (correct / total)

    # 计算混淆矩阵
    cm = confusion_matrix(all_true, all_pred)
    class_acc = cm.diagonal() / cm.sum(axis=1) * 100

    return running_loss / len(loader), test_acc, all_pred, all_true, all_outp, class_acc


class MyDataset(data.Dataset):
    def __init__(self, x1, y):
        self.x1 = x1
        self.y = y

    def __getitem__(self, index):
        input_emg = self.x1[index]
        target = self.y[index]
        return input_emg, target

    def __len__(self):
        return len(self.x1)

# 示例训练和测试
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print("Train is Starting!!!")
    train_losses = []
    source_losses = []
    target_losses = []
    meta_test_losses = []

    train_accuracies = []
    source_accuracies = []
    target_accuracies = []
    meta_test_accuracies = []

    # 用于保存训练和测试信息的DataFrame
    results_df = pd.DataFrame(columns=["Epoch", "Train Loss", "Train Accuracy", "Source Domain Loss", "Source Domain Accuracy",
                                       "Target Domain Loss", "Target Domain Accuracy", "Meta Test Loss", "Meta Test Accuracy"])

    num_epochs = 200


    # model = One_DCNN_CaptureData().to(device)
    # model = TransformerTimeSeries_Capture(feature_dim=3, n_classes=37).to(device)
    # model = LSTMModel_Capture().to(device)
    # model = BiLSTMModel_Capture().to(device)
    # model = CNN_TransformerTimeSeries().to(device)
    # model = OneD_ConvLSTMModel_Capture().to(device)
    # model = OneDCNN_BiLSTM_Capture().to(device)
    model = EMGResNet().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=6, verbose=True)



    # 加载数据
    CAPTURETRAIN = np.load('/home/dongxu/pc_tmp/MyoData/CaptureData_train.npy')
    LABELTRAIN = np.load('/home/dongxu/pc_tmp/MyoData/CaptureData_label_train.npy')
    CAPTURETEST = np.load('/home/dongxu/pc_tmp/MyoData/CaptureData_test.npy')
    LABELTEST = np.load('/home/dongxu/pc_tmp/MyoData/CaptureDatalabel_test.npy')
    # CAPTURETARGET = np.load('/home/dongxu/tmp/MyoData/CAPTURE_target.npy')
    # LABELTARGET = np.load('/home/dongxu/tmp/MyoData/label_target.npy')
    print(CAPTURETRAIN.shape)
    print(LABELTRAIN.shape)
    print(CAPTURETEST.shape)
    print(LABELTEST.shape)
    # print(CAPTURETARGET.shape)
    # print(LABELTARGET.shape)

    # 转换成tensor
    CAPTURETRAIN = torch.from_numpy(CAPTURETRAIN).float().to(device)
    CAPTURETEST = torch.from_numpy(CAPTURETEST).float().to(device)
    LABELTRAIN = torch.from_numpy(LABELTRAIN).long().to(device)
    LABELTEST = torch.from_numpy(LABELTEST).long().to(device)
    # CAPTURETARGET = torch.from_numpy(CAPTURETARGET).float().to(device)
    # LABELTARGET = torch.from_numpy(LABELTARGET).long().to(device)

    # 创建数据集和数据加载器
    traindatasets = MyDataset(CAPTURETRAIN, LABELTRAIN)  # 初始化
    testdatasets = MyDataset(CAPTURETEST, LABELTEST)
    # targetdatasets = MyDataset(CAPTURETARGET, LABELTARGET)
    train_loader = data.DataLoader(traindatasets, batch_size=32, shuffle=True, drop_last=True)
    test_loader = data.DataLoader(testdatasets, batch_size=32, shuffle=True, drop_last=True)
    # target_loader = data.DataLoader(targetdatasets, batch_size=32, shuffle=True, drop_last=True)

    best_test_acc = 0.0
    early_stop_counter = 0
    patience = 10  # 提前停止的耐心值

    # 训练模型
    for epoch in range(num_epochs):  # 训练50个epoch

        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        # 测试源域
        source_loss, source_acc,_,_,_,_= test(model, train_loader, criterion, device)
        # source_loss, source_acc = test(model, train_loader, criterion, device)
        print(f"Epoch {epoch + 1}, Source Domain Loss: {source_loss:.4f}, Source Domain Accuracy: {source_acc:.2f}%")

        # 测试元测试域
        meta_test_loss, meta_test_acc, all_pred, all_true, all_outp, class_acc = test(model, test_loader, criterion, device)
        # meta_test_loss, meta_test_acc = test(model, test_loader, criterion, device)
        print(f"Epoch {epoch + 1}, Target Domain Loss: {meta_test_loss:.4f}, Target Domain Accuracy: {meta_test_acc:.2f}%")

        # 测试目标域
        # target_loss, target_acc, _, _, _, _ = test(model, target_loader, criterion, device)
        # print(f"Epoch {epoch + 1}, Target Domain Loss: {target_loss:.4f}, Target Domain Accuracy: {target_acc:.2f}%")

        train_losses.append(train_loss)
        source_losses.append(source_loss)
        # target_losses.append(target_loss)
        meta_test_losses.append(meta_test_loss)

        train_accuracies.append(train_acc)
        source_accuracies.append(source_acc)
        # target_accuracies.append(target_acc)
        meta_test_accuracies.append(meta_test_acc)

        # 用于保存训练和测试信息的DataFrame
        # results_df = pd.DataFrame(
        #     columns=["Epoch", "Train Loss", "Train Accuracy", "Source Domain Loss", "Source Domain Accuracy",
        #              "Target Domain Loss", "Target Domain Accuracy", "Meta Test Loss", "Meta Test Accuracy"])

        # 在训练循环中，替换以下代码：
        # results_df = results_df.append({...}, ignore_index=True)

        # 使用 pd.concat 代替
        new_row = pd.DataFrame({
            "Epoch": [epoch + 1],
            "Train Loss": [train_loss],
            "Train Accuracy": [train_acc],
            "Source Domain Loss": [source_loss],
            "Source Domain Accuracy": [source_acc],
            # "Target Domain Loss": [target_loss],
            # "Target Domain Accuracy": [target_acc],
            "Target Domain Loss": [meta_test_loss],
            "Target Domain Accuracy": [meta_test_acc]
        })

        results_df = pd.concat([results_df, new_row], ignore_index=True)


        # 调度器更新
        scheduler.step(meta_test_loss)  # 使用元测试损失更新学习率
        # scheduler.step()

        # # 保存最好的模型和提前停止
        if meta_test_acc > best_test_acc:
            best_test_acc = meta_test_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), "/home/dongxu/pc_tmp/models/capture_model.pt")
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

        # 保存结果到Excel文件
    results_df.to_excel("/home/dongxu/pc_tmp/results/capture_training_results.xlsx", index=False)




    # 绘制训练和测试损失
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(source_losses, label='Source Domain Loss')
    # plt.plot(target_losses, label='Target Domain Loss')
    plt.plot(meta_test_losses, label='Target Domain Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(source_accuracies, label='Source Domain Accuracy')
    # plt.plot(target_accuracies, label='Target Domain Accuracy')
    plt.plot(meta_test_accuracies, label='Target Domain Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


    cm = confusion_matrix(all_true, all_pred)

    class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'Flip Left', 'Flip Right',
                   'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    # 创建紫黑色到棕色的自定义渐变色彩映射
    blue_orange_brown_cmap = LinearSegmentedColormap.from_list(
        "blue_orange_brown", ["#add8e6", "#FFA500", "#8B4513"], N=256)

    # 设置图形和坐标轴
    plt.figure(figsize=(20, 16))

    # 绘制热图
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap=blue_orange_brown_cmap, xticklabels=class_names, yticklabels=class_names)

    # 设置轴标签和标题
    plt.xlabel('Predicted Action', fontsize=14)
    plt.ylabel('True Action', fontsize=14)
    plt.title('Target Set Confusion Matrix', fontsize=16)

    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    # 保存图像到指定路径
    plt.savefig('/home/dongxu/pc_tmp/results/capture_confusion_matrix.jpg', dpi=300, bbox_inches='tight')

    # 显示图表
    plt.show()