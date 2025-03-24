clear;
SEAcc_Path = 'E:\研究生\DataSet\34\1\Acceleration';
rawsAccdata_Path = 'E:\研究生\DataSet\34\1\Acceleration';
AcccsvFilePath = 'E:\研究生\DataSet\34\1\Acceleration\acceleration.csv';
% 使用readtable函数读取CSV文件
acc = readtable(AcccsvFilePath,'VariableNamingRule', 'preserve');
j =15;
Acc_SP_filename = '1AccStartPoint.mat';
Acc_EP_filename = '1AccEndPoint.mat';

Acc_SP_Path = fullfile(SEAcc_Path, Acc_SP_filename);
Acc_EP_Path = fullfile(SEAcc_Path, Acc_EP_filename);
Acc_StartPoint = load(Acc_SP_Path);
Acc_EndPoint = load(Acc_EP_Path);
SPoint = Acc_StartPoint.AccSPoint;       % 由于load读取到的是一个结构体，所以可以通过这种形式读取到数据
EPoint = Acc_EndPoint.AccEPoint;

% 读出文件夹下的RigiBody1，并对其进行数据类型转换
Acc_filename = 'acceleration.mat';
Acc_Path = fullfile(rawsAccdata_Path, Acc_filename);
Acc = load(Acc_Path);
My_data = Acc.Acceleration;
data_1 = My_data(:,1);
data_2 = My_data(:,2);
data_3 = My_data(:,3);

data1 = data_1(SPoint(j):EPoint(j));
data2 = data_2(SPoint(j):EPoint(j));
data3 = data_3(SPoint(j):EPoint(j));
M = [data1 data2 data3]
stackedplot(M)

% 自定义函数对 z 轴数据进行映射
z_transform = @(z) (z > -0.5) .* (z - 0.5) * 0.5 + (z > -0.5) * 0.5 + (z <= -0.5) .* z;

% 应用映射
data1_z = z_transform(data1);
data2_z = z_transform(data2);
data3_z = z_transform(data3);

numPoints = EPoint(j) - SPoint(j) + 1;  % 实际的数据点数
totalTime = numPoints;  % 总时间，单位ms
t = linspace(0, totalTime, numPoints);  % 根据实际数据点数创建时间向量

figure;
hold on;
colors = lines(3);  % 生成3种不同的颜色

% 依次绘制每个通道的数据
plot3(t, ones(numPoints,1)*1, data1_z, 'Color', colors(1,:));
plot3(t, ones(numPoints,1)*2, data2_z, 'Color', colors(2,:));
plot3(t, ones(numPoints,1)*3, data3_z, 'Color', colors(3,:));

grid on;
xlabel('Time/ms');
ylabel('Channels');
zlabel('Amplitude');

% 设置离散的 x 轴刻度
%xticks([0, totalTime * 0.1, totalTime * 0.24, totalTime * 0.36, totalTime * 0.48, totalTime * 0.6, totalTime * 0.72, totalTime * 0.84, totalTime]);

% 设置 z 轴刻度
zticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.6, 1.2]);
zlim([-1.2, 1.2]);

% 调整视角
view(-48, 27);
campos([-1052.3, -8.6, 9.436]);
hold off;
