clear;
Path = 'E:\研究生\DataSet\1\1\Optitrack';
SEPath = 'E:\研究生\DataSet\1\1\Optitrack';
i = 32;%i表示该组的第i个动作

fileName1 = fullfile(SEPath, '1StartPoint.mat');
fileName2 = fullfile(SEPath, '1EndPoint.mat');
csvFilePath = fullfile(Path, 'RigidBody1.csv');

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


figure('Position',[680,558,560,420]); % 创建新图形窗口
hold on; % 保持图形，允许添加新的图形元素

% 获取特定动作的数据区间
x_segment = X_data(SPoint(i):EPoint(i));
y_segment = Y_data(SPoint(i):EPoint(i));
z_segment = Z_data(SPoint(i):EPoint(i));

% 绘制三维折线图
plot3(x_segment, y_segment, z_segment, 'LineWidth', 2);
%title('3D Trajectory of Motion Data');
xlabel('X Axis');
ylabel('Y Axis');
zlabel('Z Axis');
grid on; % 开启网格

view(-158, 5);  % 调整视角以更好地观看图形
campos([-1.812924677525146 1.601 1.2174])

hold off; % 结束图形保持状态
