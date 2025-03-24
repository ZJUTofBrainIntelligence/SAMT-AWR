clc;
clear;
Path = 'E:\研究生\DataSet\1\1\Optitrack';
SEPath = 'E:\研究生\DataSet\1\1\Optitrack';

fileName1 = fullfile(SEPath, '1StartPoint.mat');
fileName2 = fullfile(SEPath, '1EndPoint.mat');
csvFilePath = fullfile(Path, 'RigidBody1.csv');

% 使用readtable函数读取CSV文件
RigidBody1 = readtable(csvFilePath,'VariableNamingRule', 'preserve');

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
for i = 1 : 37
    if(EPoint(i)-SPoint(i) > 400)
       fprintf('动捕第[%d]动作时间过长！\n', i);        
    end
    if((0 <= EPoint(i)-SPoint(i))&& (EPoint(i)-SPoint(i)<= 180))
       fprintf('动捕第[%d]动作太短\n', i);        
    end
    if( EPoint(i)-SPoint(i) <= 10)
       fprintf('动捕第[%d]动作error!!!\n', i);        
    end
    plot(X_data(SPoint(i):EPoint(i)));
end
hold off;
legend('X');

figure;
hold on;
for i = 1 : 37
    plot(Y_data(SPoint(i):EPoint(i)));
end
hold off;
legend('Y');

figure;
hold on;
for i = 1 : 37
    plot(Z_data(SPoint(i):EPoint(i)));
end
hold off;
legend('Z');

clear;

% figure;
% hold on;
% i = 13;
% plot(X_data(SPoint(i):EPoint(i)));
% hold off;
% legend('X');
% 
% figure;
% hold on;
% plot(Y_data(SPoint(i):EPoint(i)));
% hold off;
% legend('Y');
% 
% figure;
% hold on;
% plot(Z_data(SPoint(i):EPoint(i)));
% hold off;
% legend('Z');