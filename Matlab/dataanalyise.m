% dataanalyise.m
% 分析 ele434_team10_2026 机器人的 trace.csv 运行日志数据

clear; clc; close all;

% 默认的单个数据路径 (你也可以将其改成包含 '.tmp_logs' 的路径)
default_file = '/home/kevin/Documents/task/tmp_logs/work_20260512_172917/trace.csv';

% 检查文件是否存在，如果不存在则允许用户通过 UI 窗口选择
if exist(default_file, 'file')
    trace_file = default_file;
else
    disp('未找到指定的默认日志，请手动选择 trace.csv');
    % 指向 tmp_logs 目录让用户选择
    search_dir = '/home/kevin/Documents/task/tmp_logs/';
    [file, path] = uigetfile('*.csv', '选择 trace.csv 文件', search_dir);
    if isequal(file, 0)
        disp('已取消文件选择');
        return;
    end
    trace_file = fullfile(path, file);
end

fprintf('正在加载日志文件: %s\n', trace_file);

% 使用 readtable 读取 csv 数据
try
    data = readtable(trace_file);
catch ME
    error('读取文件失败，请检查文件格式: %s', ME.message);
end

% 创建可视化窗口
figure('Name', '机器人运行状态综合分析', 'Position', [100, 100, 1200, 800]);

% 1. 轨迹与地图路径规划分析
subplot(2, 2, 1);
hold on; box on; grid on;
plot(data.x, data.y, 'b-', 'LineWidth', 1.5, 'DisplayName', '实际轨迹');
plot(data.tgt_x, data.tgt_y, 'ro', 'MarkerSize', 4, 'DisplayName', '追踪目标 (tgt)');
plot(data.lh_x, data.lh_y, 'g.', 'MarkerSize', 6, 'DisplayName', '前瞻点 (lh)');
% 标注起点和终点
plot(data.x(1), data.y(1), 'ks', 'MarkerSize', 8, 'MarkerFaceColor', 'y', 'DisplayName', '起点');
plot(data.x(end), data.y(end), 'kh', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', '终点');
xlabel('X 坐标 (m)');
ylabel('Y 坐标 (m)');
title('机器人全局 XY 轨迹');
axis equal;
legend('Location', 'best');

% 2. 速度指令分析
subplot(2, 2, 2);
hold on; box on; grid on;
plot(data.t, data.v_cmd, 'b-', 'LineWidth', 1.2, 'DisplayName', '线速度 v\_cmd (m/s)');
plot(data.t, data.w_cmd, 'r-', 'LineWidth', 1.2, 'DisplayName', '角速度 w\_cmd (rad/s)');
xlabel('时间 t (s)');
ylabel('指令值');
title('控制层下发速度指令');
legend('Location', 'best');

% 3. 传感器测距数据 (避障与沿墙参考)
subplot(2, 2, 3);
hold on; box on; grid on;
plot(data.t, data.d_front, 'k-', 'LineWidth', 1.2, 'DisplayName', '正前方 d\_front');
plot(data.t, data.d_fl, 'g-', 'LineWidth', 1.2, 'DisplayName', '左前方 d\_fl');
plot(data.t, data.d_fr, 'm-', 'LineWidth', 1.2, 'DisplayName', '右前方 d\_fr');
xlabel('时间 t (s)');
ylabel('距离 (m)');
title('前方 LiDAR 测距数据');
legend('Location', 'best');

% 4. 路径跟踪误差
subplot(2, 2, 4);
yyaxis left;
plot(data.t, data.yaw_err_deg, '-', 'LineWidth', 1.2);
ylabel('偏航角误差 yaw\_err (°)');
yyaxis right;
plot(data.t, data.d_goal, '-', 'LineWidth', 1.2);
ylabel('距目标距离 d\_goal (m)');
xlabel('时间 t (s)');
title('追踪误差与目标距离');
grid on;