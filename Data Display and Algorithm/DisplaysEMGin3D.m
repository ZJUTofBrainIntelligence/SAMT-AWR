
clear;
SEsEMG_Path = 'E:\研究生\DataSet\34\2\sEMG';
rawsEMGdata_Path = 'E:\研究生\DataSet\34\2\sEMG';
EMGcsvFilePath = 'E:\研究生\DataSet\34\2\sEMG\emg.csv'
%使用readtable函数读取CSV文件
emg = readtable(EMGcsvFilePath,'VariableNamingRule', 'preserve');
j =25;
EMG_SP_filename = '2sEMGStartPoint.mat';
EMG_EP_filename = '2sEMGEndPoint.mat';

EMG_SP_Path = fullfile(SEsEMG_Path, EMG_SP_filename);
EMG_EP_Path = fullfile(SEsEMG_Path, EMG_EP_filename);
EMG_StartPoint = load(EMG_SP_Path);
EMG_EndPoint = load(EMG_EP_Path);
SPoint = EMG_StartPoint.ESPoint;       %由于load读取到的是一个结构体，所以可以通过这种形式读取到数据
EPoint = EMG_EndPoint.EEPoint;

 % 读出文件夹下的RigiBody1，并对其进行数据类型转换
    EMG_filename = 'emg.mat';
    EMG_Path = fullfile(rawsEMGdata_Path, EMG_filename);
    EMG = load(EMG_Path);
    My_data = EMG.emgArray;
    %My_data = table2array(EMG);
    data_1 = My_data(:,1);
    data_2 = My_data(:,2);
    data_3 = My_data(:,3);
    data_4 = My_data(:,4);
    data_5 = My_data(:,5);
    data_6 = My_data(:,6);
    data_7 = My_data(:,7);
    data_8 = My_data(:,8);
% 
%     data_1 = data_1';
%     data_2 = data_2';
%     data_3 = data_3';
%     data_4 = data_4';
%     data_5 = data_5';
%     data_6 = data_6';
%     data_7 = data_7';
%     data_8 = data_8';

    data1 = data_1(SPoint(j):EPoint(j));
    data2 = data_2(SPoint(j):EPoint(j));
    data3 = data_3(SPoint(j):EPoint(j));
    data4 = data_4(SPoint(j):EPoint(j));
    data5 = data_5(SPoint(j):EPoint(j));
    data6 = data_6(SPoint(j):EPoint(j));
    data7 = data_7(SPoint(j):EPoint(j));
    data8 = data_8(SPoint(j):EPoint(j));
    M = [data1 data2 data3 data4 data5 data6 data7 data8];
    stackedplot(M);
numPoints = EPoint(j)-SPoint(j)+1;  % 实际的数据点数
totalTime = 1000;  % 总时间，单位ms
t = linspace(0, numPoints, numPoints);  % 使用实际的数据点数来创建时间向量

figure('Position', [100, 100, 750, 416]);  % 设置图形窗口的位置和大小;
hold on;
colors = lines(8);  % 生成8种不同的颜色

% 创建一个空的cell数组来存储图例文本
legendInfo = cell(1, 8);

% 依次绘制每个通道的数据并添加到图例信息
plot3(t, ones(numPoints,1)*1, data1, 'Color', colors(1,:));
legendInfo{1} = 'Channel 1';

plot3(t, ones(numPoints,1)*2, data2, 'Color', colors(2,:));
legendInfo{2} = 'Channel 2';

plot3(t, ones(numPoints,1)*3, data3, 'Color', colors(3,:));
legendInfo{3} = 'Channel 3';

plot3(t, ones(numPoints,1)*4, data4, 'Color', colors(4,:));
legendInfo{4} = 'Channel 4';

plot3(t, ones(numPoints,1)*5, data5, 'Color', colors(5,:));
legendInfo{5} = 'Channel 5';

plot3(t, ones(numPoints,1)*6, data6, 'Color', colors(6,:));
legendInfo{6} = 'Channel 6';

plot3(t, ones(numPoints,1)*7, data7, 'Color', colors(7,:));
legendInfo{7} = 'Channel 7';

plot3(t, ones(numPoints,1)*8, data8, 'Color', colors(8,:));
legendInfo{8} = 'Channel 8';

% 添加图例
legend(legendInfo, 'Location', 'northeastoutside');  % 将图例放在图形的右上角外侧

grid on;
xlabel('Time/ms');
ylabel('Channels');
zlabel('Amplitude');
% title('8-Channel sEMG Data');
view(-45, 45);  % 调整视角以更好地观看图形
campos([-2000 -30 2000]);
hold off;

  %CameraPosition (-2000,-30,2000 )
  %view(-45, 45)
  %Position中的width为750,height为416
      



