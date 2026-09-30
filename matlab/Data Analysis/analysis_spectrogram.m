clc;
clear;
close all;
debug = 0;
ToF_debug = 0;

%filename='D:\PhD_projects\Current_Projects\fitness\data\past\gesture1_cong.dat';
%filename='D:\PhD_projects\Current_Projects\IoT_project\Data\02152019\zhenzhe_walking_3people_interference.dat';
%filename='D:\PhD_projects\Current_Projects\IoT_project\Data\0715\8.5m_5m_channel_48_36_walkcut_mcs_2_6.dat'
filename = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data/0340/E02_U06_A09.dat');
% filename='D:\Dropbox\conformal_prediction\data\deep_l\changming_drawing_Z.dat';
N = 10;
energy = 100;
Scut_off = [1, 45];

sample_rate = 100;
sample_win = 4;
index = 22;
subcarriers = 30;
d = 0.0573 / 2; % half wavelength

Threshold_amplitude = -35;
Threshold_amplitude_ceil = 10;
Threshold_phase = -45; % Threshold for spectrogram
% Sander
Tx = 1;

for Tx = 1:3
    % Process data from the first antenna
    [~, H_complex_antenna1, Time1, H1, ~] = step02_get_allkinds_scaled_csi(filename, Tx, 1);
    disp(['First Antenna for Tx ', num2str(Tx)]);
    [resample_H1, resample_Time1] = resampleValues(Time1, H1);

    % Process data from the second antenna
    [~, H_complex_antenna2, Time2, H2, ~] = step02_get_allkinds_scaled_csi(filename, Tx, 2);
    disp(['Second Antenna for Tx ', num2str(Tx)]);
    [resample_H2, resample_Time2] = resampleValues(Time2, H2);

    % Process data from the third antenna
    [~, H_complex_antenna3, Time3, H3, ~] = step02_get_allkinds_scaled_csi(filename, Tx, 3);
    disp(['Third Antenna for Tx ', num2str(Tx)]);
    [resample_H3, resample_Time3] = resampleValues(Time3, H3);

    % Downsample the data
    D_H1 = downsample_function(resample_H1);
    D_H2 = downsample_function(resample_H2);
    D_H3 = downsample_function(resample_H3);

    % Concatenate the data for all three antennas for this Tx
    magnitude = cat(3, D_H1, D_H2, D_H3);

    % Store the magnitude in the cell array
    all_magnitudes{Tx} = magnitude;
end
% Concatenate all magnitudes from each Tx into a single 4D array
final_magnitude = cat(3, all_magnitudes{:});
save(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data/0340/E02_U06_A09.mat'), 'final_magnitude');
%save file

stack = vertcat(D_H1, D_H2, D_H3);
k = 10;
movingVariances = movvar(stack, k, 1, 2);
averageMovingVariances = mean(movingVariances, 1);
figure; % 创建一个新的图形窗口
plot(averageMovingVariances, 'LineWidth', 2); % 绘制平均移动方差，设置线宽为2
grid on; % 开启网格
xlabel('Point Index'); % X轴标签
ylabel('Average Moving Variance'); % Y轴标签
title('Average Moving Variance of all Subcarrier Over Time');

% Test 15 subcarrier
relative_phase1 = angle(H_complex_antenna1 .* conj(H_complex_antenna2));
relative_phase2 = angle(H_complex_antenna1 .* conj(H_complex_antenna3));
relative_phase3 = angle(H_complex_antenna2 .* conj(H_complex_antenna3));

% csi_start = 3*sample_rate;
csi_start = 1;
csi_end = min(size(H1, 2), size(H2, 2));
csi_end = min(csi_end, size(H3, 2)) - 0 * sample_rate;
len = csi_end - csi_start + 1;

relative_phase1 = relative_phase1(:, csi_start:csi_end);
relative_phase2 = relative_phase2(:, csi_start:csi_end);
relative_phase3 = relative_phase3(:, csi_start:csi_end);
Time_phase = Time1(:, csi_start:csi_end);
%relative_phase = relative_phase1(:,find(Time>=time_start & Time<=time_end));
%relative_phase = unwrap(relative_phase1(:,find(Time>=time_start & Time<=time_end)),[],2);
H1 = H1(:, csi_start:csi_end);
H2 = H2(:, csi_start:csi_end);
H3 = H3(:, csi_start:csi_end);
Time_amp = Time1(:, csi_start:csi_end);
Time = Time_amp;

stackedArray = cat(3, H1, H2, H3);
%save('H:\0340\E03_U02_A10.mat', 'stackedArray');
clear H_complex_antenna1;
clear H_complex_antenna2;
clear H_complex_antenna3;

%% Filter the low frequency noise ?amplitude?
stack2 = [];

% Scut_off=[3,sample_rate/2-2];

%cut_off = 0.3;

Norm_freq = Scut_off / (sample_rate) * 2;
[b, a] = butter(2, Norm_freq, 'bandpass');
temp = freqz(b, a);
% dataOut = filter(b,a,H1(index,:));
% figure;
% plot(H1(index,:));

signal = D_H1;
for i = 1:N
    sub_signal = signal(i, :);

    % Butter worth filter
    % ================
    sub_signal = filter(b, a, sub_signal);
    [s_amplitude, ~, ~, ps_amplitude] = spectrogram(sub_signal, round(sample_rate / sample_win), round(sample_rate / sample_win / 2), sample_rate, sample_rate);
    if isempty(stack2)
        stack2 = zeros (size(s_amplitude, 1), size(s_amplitude, 2));
    end
    stack2 = stack2 + abs(s_amplitude);

end

fig4 = figure;
hold on;
set(fig4, 'Position', [500, 500, 1200, 300]);
x_time = 0:max(Time) / size(stack2, 2):max(Time) / size(stack2, 2) * (size(stack2, 2) - 1);
surf(x_time, 1:size(stack2, 1), stack2, 'EdgeColor', 'none');
caxis([0 energy]);
colorbar;
colormap(jet);
ylim([2 100]);
xlim([min(x_time) max(x_time)]);
ylabel('Frequency (Hz)');
xlabel('Time');
title('Accumulated Amplitude Spectrogram');
%% Filter the low frequency noise ?amplitude?
%close all;
stack2 = [];

Norm_freq = Scut_off / (sample_rate) * 2;
[b, a] = butter(2, Norm_freq, 'bandpass');
temp = freqz(b, a);

signal = H1;
range = 6:10;

% for i = range
%     sub_signal = signal(i,:);
%     sub_signal = filter(b,a, sub_signal);
%
%     [s_amplitude,~,~,ps_amplitude] = spectrogram(sub_signal,round(sample_rate/sample_win),round(sample_rate/sample_win/2),sample_rate,sample_rate);
%     s_amplitude=abs(s_amplitude);
%     figure
%     hold on
%     x_time = 0:max(Time)/size(s_amplitude,2):max(Time)/size(s_amplitude,2)*(size(s_amplitude,2)-1);
%     surf(x_time,1:size(s_amplitude,1),s_amplitude,'EdgeColor','none');
%     caxis([0 100]);
%     colorbar;
%     colormap(jet);
%     ylim([2 100]);
%     xlim([min(x_time) max(x_time)]);
%     ylabel('Frequency (Hz)');
%     xlabel('Time');
%     title(['Relative Phase Spectrogram' int2str(i)]);
%
% end

%  %% CSI measurements from 30 subcarrier
% figure
% hold on;
% mesh(H1);
% colorbar;
% colormap(jet);
%
% figure
% for i=1:5
%     hold on;
%     plot(H1(i,:));
% end
%
function downsampled = downsample_function(H1)
    downsampled = zeros(30, size(H1, 2) / 10);
    for i = 1:30
        downsampled(i, :) = downsample(H1(i, :), 10);
    end
end

function [final_values, final_timestamps] = resampleValues(Time1, H1)
    timestamps = Time1 / 1000000;

    dt_expected = median(diff(timestamps));  % 期望的时间间隔，可以根据实际情况调整

    timestamps_corr = timestamps;  % 复制原始数据
    offset = 0;  % 初始补偿量为0

    for i = 2:length(timestamps)
        % 当前值加上之前的补偿后应大于前一个值
        if (timestamps_corr(i) + offset) < timestamps_corr(i - 1)
            % 计算需要补偿的差值，使当前值接近前一个值加上期望的间隔
            delta = (timestamps_corr(i - 1) + dt_expected) - (timestamps_corr(i) + offset);
            offset = offset + delta;  % 更新补偿量
        end
        % 更新当前值
        timestamps_corr(i) = timestamps_corr(i) + offset;
    end

    startTime = timestamps_corr(1);
    endTime = timestamps_corr(end) - 1;

    values = H1;
    values_upsampled = {};
    timestamps_upsampled = {}; % 初始化一个cell数组来存储处理后的时间戳

    current_time = startTime;
    while current_time < endTime
        idx = find(timestamps >= current_time & timestamps < current_time + 1);

        if ~isempty(idx)
            currentTimestamps = timestamps(idx);
            currentValues = values(:, idx);
            newTimestamps = linspace(current_time, current_time + 1, 1000);

            processedValues = zeros(size(values, 1), 1000);

            if length(idx) < 1000
                % 上采样
                for i = 1:size(values, 1)
                    processedValues(i, :) = interp1(currentTimestamps, currentValues(i, :), newTimestamps, 'linear', 'extrap');
                end
                timestamps_upsampled{end + 1} = newTimestamps;
            elseif length(idx) > 1000
                % 下采样
                downsampleIndices = round(linspace(1, length(idx), 1000));
                processedValues = currentValues(:, downsampleIndices);
                timestamps_upsampled{end + 1} = currentTimestamps(downsampleIndices);
            else
                % 数据点数量已经为1000，不需要处理
                processedValues = currentValues;
                timestamps_upsampled{end + 1} = currentTimestamps;
            end

            values_upsampled{end + 1} = processedValues;
        end
        current_time = current_time + 1;
    end

    % 将上采样或下采样后的值合并为最终矩阵
    final_values = horzcat(values_upsampled{:});
    % 合并处理后的时间戳
    final_timestamps = horzcat(timestamps_upsampled{:});
end
