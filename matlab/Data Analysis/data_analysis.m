clc;
close all;
clear;

subcarrier = 30;
samples = 1;
channel = 3;

filename1 = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data/appleman_0713/0713_xiangyu_sitting_B1_raw.mat');
load(filename1)
data = segment_data_stft(samples, subcarrier, :, channel);
data1 = squeeze(data);

filename = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data/appleman_0713/0713_xiangyu_raisingarm_B1_raw.mat');
load(filename)
data = segment_data_stft(samples, subcarrier, :, channel);
data2 = squeeze(data);

filename = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data/appleman_0713/0713_xiangyu_walking_traj2_B1_raw.mat');
load(filename)
data = segment_data_stft(samples, subcarrier, :, channel);
data3 = squeeze(data);

filename1 = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data/appleman_0713/0713_xiangyu_sitting_B2_raw.mat');
load(filename1)
data = segment_data_stft(samples, subcarrier, :, channel);
data4 = squeeze(data);

filename = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data/appleman_0713/0713_xiangyu_raisingarm_B2_raw.mat');
load(filename)
data = segment_data_stft(samples, subcarrier, :, channel);
data5 = squeeze(data);

filename = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data/appleman_0713/0713_xiangyu_walking_traj2_B2_raw.mat');
load(filename)
data = segment_data_stft(samples, subcarrier, :, channel);
data6 = squeeze(data);

% filename1 = 'C:\local\Siemens\processed_data\appleman_0713\0713_xin_sitting_B1_raw.mat';
% load(filename1)
% data = segment_data_stft(samples, subcarrier, :, channel);
% data4 = squeeze(data);
%
% filename = 'C:\local\Siemens\processed_data\appleman_0713\0713_xin_raisingarm_B1_raw.mat';
% load(filename)
% data = segment_data_stft(samples, subcarrier, :, channel);
% data5 = squeeze(data);
%
% filename = 'C:\local\Siemens\processed_data\appleman_0713\0713_xin_walking_traj2_B1_raw.mat';
% load(filename)
% data = segment_data_stft(samples, subcarrier, :, channel);
% data6 = squeeze(data);

%% CSI amplitude
% figure
% subplot(2, 3, 1)
% histogram(data1, 40, 'Normalization', 'probability', 'BinEdges', -20:10)
% xlabel('CSI Amplitude')
% ylabel('Probability')
% ylim([0 0.6])
% set(gca, 'FontSize', 16)
%
% subplot(2, 3, 2)
% histogram(data2, 40, 'Normalization', 'probability', 'BinEdges', -20:10)
% xlabel('CSI Amplitude')
% ylabel('Probability')
% ylim([0 0.6])
% set(gca, 'FontSize', 16)
%
% subplot(2, 3, 3)
% histogram(data3, 'Normalization', 'probability', 'BinEdges', -20:10)
% xlabel('CSI Amplitude')
% ylabel('Probability')
% ylim([0 0.6])
% set(gca, 'FontSize', 16)
%
% subplot(2, 3, 4)
% histogram(data4, 20, 'Normalization', 'probability', 'BinEdges', -20:10)
% xlabel('CSI Amplitude')
% ylabel('Probability')
% ylim([0 0.6])
% set(gca, 'FontSize', 16)
%
% subplot(2, 3, 5)
% histogram(data5, 20, 'Normalization', 'probability', 'BinEdges', -20:10)
% xlabel('CSI Amplitude')
% ylabel('Probability')
% ylim([0 0.6])
% set(gca, 'FontSize', 16)
%
% subplot(2, 3, 6)
% histogram(data6, 20, 'Normalization', 'probability', 'BinEdges', -20:10)
% xlabel('CSI Amplitude')
% ylabel('Probability')
% ylim([0 0.6])
% set(gca, 'FontSize', 16)

%% CSI amplitude (spectrogram)
% figure
% subplot(2, 3, 1)
% histogram(data1, 40, 'Normalization', 'probability', 'BinEdges', 0:20:600)
% xlabel('CSI Amplitude')
% ylabel('Probability')
% ylim([0 0.9])
% set(gca, 'FontSize', 16)
%
% subplot(2, 3, 2)
% histogram(data2, 40, 'Normalization', 'probability', 'BinEdges', 0:20:600)
% xlabel('CSI Amplitude')
% ylabel('Probability')
% ylim([0 0.9])
% set(gca, 'FontSize', 16)
%
% subplot(2, 3, 3)
% histogram(data3, 'Normalization', 'probability', 'BinEdges', 0:20:600)
% xlabel('CSI Amplitude')
% ylabel('Probability')
% ylim([0 0.9])
% set(gca, 'FontSize', 16)
%
% subplot(2, 3, 4)
% histogram(data4, 20, 'Normalization', 'probability', 'BinEdges', 0:20:600)
% xlabel('CSI Amplitude')
% ylabel('Probability')
% ylim([0 0.9])
% set(gca, 'FontSize', 16)
%
% subplot(2, 3, 5)
% histogram(data5, 20, 'Normalization', 'probability', 'BinEdges', 0:20:600)
% xlabel('CSI Amplitude')
% ylabel('Probability')
% ylim([0 0.9])
% set(gca, 'FontSize', 16)
%
% subplot(2, 3, 6)
% histogram(data6, 20, 'Normalization', 'probability', 'BinEdges', 0:20:600)
% xlabel('CSI Amplitude')
% ylabel('Probability')
% ylim([0 0.9])
% set(gca, 'FontSize', 16)

%% CSI relative amplitude (spectrogram)
figure
subplot(2, 3, 1)
histogram(data1, 'Normalization', 'probability', 'BinEdges', 0:400:20000)
xlabel('CSI Amplitude')
ylabel('Probability')
ylim([0 0.9])
set(gca, 'FontSize', 16)

subplot(2, 3, 2)
histogram(data2, 'Normalization', 'probability', 'BinEdges', 0:400:20000)
xlabel('CSI Amplitude')
ylabel('Probability')
ylim([0 0.9])
set(gca, 'FontSize', 16)

subplot(2, 3, 3)
histogram(data3, 'Normalization', 'probability', 'BinEdges', 0:400:20000)
xlabel('CSI Amplitude')
ylabel('Probability')
ylim([0 0.9])
set(gca, 'FontSize', 16)

subplot(2, 3, 4)
histogram(data4, 'Normalization', 'probability', 'BinEdges', 0:400:20000)
xlabel('CSI Amplitude')
ylabel('Probability')
ylim([0 0.9])
set(gca, 'FontSize', 16)

subplot(2, 3, 5)
histogram(data5, 'Normalization', 'probability', 'BinEdges', 0:400:20000)
xlabel('CSI Amplitude')
ylabel('Probability')
ylim([0 0.9])
set(gca, 'FontSize', 16)

subplot(2, 3, 6)
histogram(data6, 'Normalization', 'probability', 'BinEdges', 0:400:20000)
xlabel('CSI Amplitude')
ylabel('Probability')
ylim([0 0.9])
set(gca, 'FontSize', 16)

%% CSI relative amplitude
%
% figure
% subplot(2, 3, 1)
% histogram(data1, 40, 'Normalization', 'probability', 'BinEdges', -400:10:200)
% xlabel('CSI Relative Amplitude')
% ylabel('Probability')
% ylim([0 0.3])
% set(gca, 'FontSize', 16)
%
% subplot(2, 3, 2)
% histogram(data2, 40, 'Normalization', 'probability', 'BinEdges', -400:10:200)
% xlabel('CSI Relative Amplitude')
% ylabel('Probability')
% ylim([0 0.3])
% set(gca, 'FontSize', 16)
%
% subplot(2, 3, 3)
% histogram(data3, 'Normalization', 'probability', 'BinEdges', -400:10:200)
% xlabel('CSI Relative Amplitude')
% ylabel('Probability')
% ylim([0 0.3])
% set(gca, 'FontSize', 16)
%
% subplot(2, 3, 4)
% histogram(data4, 20, 'Normalization', 'probability', 'BinEdges',-400:10:200)
% xlabel('CSI Relative Amplitude')
% ylabel('Probability')
% ylim([0 0.3])
% set(gca, 'FontSize', 16)
%
% subplot(2, 3, 5)
% histogram(data5, 20, 'Normalization', 'probability', 'BinEdges',-400:10:200)
% xlabel('CSI Relative Amplitude')
% ylabel('Probability')
% ylim([0 0.3])
% set(gca, 'FontSize', 16)
%
% subplot(2, 3, 6)
% histogram(data6, 20, 'Normalization', 'probability', 'BinEdges', -400:10:200)
% xlabel('CSI Relative Amplitude')
% ylabel('Probability')
% ylim([0 0.3])
% set(gca, 'FontSize', 16)

set(gcf, 'Position', [200 100 1100 500]);
