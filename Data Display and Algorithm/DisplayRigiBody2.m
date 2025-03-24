clear;
Path = 'E:\研究生\DataSet\34\1\Optitrack\';
SEPath = 'E:\研究生\DataSet\34\1\Optitrack';
i = 35;%i表示该组的第i个动作

fileName1 = fullfile(SEPath, '1StartPoint.mat');
fileName2 = fullfile(SEPath, '1EndPoint.mat');
csvFilePath = fullfile(Path, 'RigidBody1.csv');

% fileName1 = fullfile(Path, 'StartPoint.mat');
% fileName2 = fullfile(Path, 'EndPoint.mat');
% csvFilePath = fullfile(Path, 'RigidBody1.csv');

% 使用readtable函数读取CSV文件
RigidBody1 = readtable(csvFilePath,'VariableNamingRule', 'preserve');

% 定义MAT文件保存路径和文件名
matFilePath = fullfile(Path, 'RigidBody1.mat');


% 将数据保存为MAT文件的
% save(matFilePath, 'RigidBody1');

%% 数据格式转换
My_data = table2array(RigidBody1);
X_data = My_data(:,4);
Y_data = My_data(:,5);
Z_data = My_data(:,6);
X_data = X_data';
Y_data = Y_data';
Z_data = Z_data';

% 加载文件中的数据
SPointData = load(fileName1);
EPointData = load(fileName2);

% 假设文件中变量名分别为'SPoint'和'EPoint'
SPoint = SPointData.SPoint;
EPoint = EPointData.EPoint;



figure;
hold on;

plot(X_data(SPoint(i):EPoint(i)));
hold off;
legend('X');

figure;
hold on;
plot(Y_data(SPoint(i):EPoint(i)));
hold off;
legend('Y');

figure;
hold on;
plot(Z_data(SPoint(i):EPoint(i)));
hold off;
legend('Z');
clear;
