clc;
clear;
close all;
debug = 0;
ToF_debug = 0;

%filename='D:\PhD_projects\Current_Projects\fitness\data\past\gesture1_cong.dat';
%filename='D:\PhD_projects\Current_Projects\IoT_project\Data\02152019\zhenzhe_walking_3people_interference.dat';
%filename='D:\PhD_projects\Current_Projects\IoT_project\Data\0715\8.5m_5m_channel_48_36_walkcut_mcs_2_6.dat'
% filename='D:\Dropbox\Dropbox\Siemens_Futuremarker\data\raw_data\0220\song_0220_document_loc4.dat';
filename = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data/deep_l/recordings/tianfang_drawing_Z.dat');
% filename='D:\Dropbox\conformal_prediction\data\crossroom\CSI\test\runtian\runtian_walk_103.dat';
N = 10;
energy = 1000;
Scut_off = [1, 100];

sample_rate = 1000;
sample_win = 4;
index = 22;
subcarriers = 30;
d = 0.0573 / 2; % half wavelength

Threshold_amplitude = -35;
Threshold_amplitude_ceil = 10;
Threshold_phase = -45; % Threshold for spectrogram
% Sander
Tx = 1;

[~, H_complex_antenna1, Time1, H1, ~] = step02_get_allkinds_scaled_csi(filename, Tx, 1);
disp('First Antenna');
[~, H_complex_antenna2, Time2, H2, ~] = step02_get_allkinds_scaled_csi(filename, Tx, 2);
disp('Second Antenna');
[~, H_complex_antenna3, Time3, H3, ~] = step02_get_allkinds_scaled_csi(filename, Tx, 3);
disp('Third Antenna');
%Compute the relative phase from CSI measurements from two receiver
%antennas12 13 23

% Test 15 subcarrier
x_min = 30;
y_min = size(H1, 2);

H1 = imresize(H1, [x_min y_min]);
H2 = imresize(H2, [x_min y_min]);
H3 = imresize(H3, [x_min y_min]);

relative_amplitude1 = abs(H_complex_antenna1 .* conj(H_complex_antenna2));
relative_amplitude2 = abs(H_complex_antenna1 .* conj(H_complex_antenna3));
relative_amplitude3 = abs(H_complex_antenna2 .* conj(H_complex_antenna3));

relative_amplitude1 = imresize(relative_amplitude1, [x_min y_min]);
relative_amplitude2 = imresize(relative_amplitude2, [x_min y_min]);
relative_amplitude3 = imresize(relative_amplitude3, [x_min y_min]);

H = [H1; H2; H3; relative_amplitude1; relative_amplitude2; relative_amplitude3];

csi_start = 0.1 * sample_rate;
csi_end = min(size(H1, 2), size(H2, 2));
csi_end = min(csi_end, size(H3, 2)) - 3 * sample_rate;
H = H(:, csi_start:csi_end);

Time_amp = Time1(:, csi_start:csi_end);
Time = Time_amp;

stackedArray = H;
output_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'stack_arary');
if ~isfolder(output_dir),
    mkdir(output_dir);
end
save(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'stack_arary/tianfang_6.mat'), 'stackedArray');
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

signal = H;
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
