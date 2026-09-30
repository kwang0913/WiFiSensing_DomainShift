clc;
clear;
close all;
debug = 0;
ToF_debug = 0;
addpath(genpath([fullfile(fileparts(fileparts(mfilename('fullpath'))), 'matlab/'), filesep]));

location = {'1'; '2'; '3'; '4'};
% users = {'cong';'song';'xiangyu';'baiyang';'yilin'}; %0303
% users = {'cong';'song';'xiangyu';'baiyang';'zhenzhe'}; %0310
users = {'cong'; 'song'; 'xiangyu'; 'xin'; 'yilin'}; %0312

figure
for loc = 1:size(location, 1)
    for user = 1:size(users, 1)

        filename = [[fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data/raw_data/0312_afternoon/data/'), filesep] char(users(user)) '_0312_squat_loc_a' char(location(loc)) '.dat'];
        disp(filename)
        Scut_off = [5, 100];
        sample_rate = 1000;
        subcarriers = 30;

        Threshold_amplitude = -35;
        Threshold_amplitude_ceil = 10;
        Threshold_phase = -45;

        Tx = 1; % Transmitting antenna
        [~, H_complex_antenna1, Time1, H1, ~] = step02_get_allkinds_scaled_csi(filename, Tx, 1);
        [~, H_complex_antenna2, Time2, H2, ~] = step02_get_allkinds_scaled_csi(filename, Tx, 2);
        [~, H_complex_antenna3, Time3, H3, ~] = step02_get_allkinds_scaled_csi(filename, Tx, 3);

        relative_phase1 = angle(H_complex_antenna1 .* conj(H_complex_antenna2));
        relative_phase2 = angle(H_complex_antenna1 .* conj(H_complex_antenna3));
        relative_phase3 = angle(H_complex_antenna2 .* conj(H_complex_antenna3));

        clear H_complex_antenna1;
        clear H_complex_antenna2;
        clear H_complex_antenna3;

        %% CSI measurements from 30 subcarrier

        subplot(size(location, 1), size(users, 1), (loc - 1) * size(users, 1) + user)
        disp([size(location, 1), size(users, 1), (loc - 1) * size(users, 1) + user])
        hold on;
        %surf(x_1time,1:size(stack2,1),stack2,'EdgeColor','none');
        Time = 0:(1 / sample_rate):(1 / sample_rate) * size(H1, 2);
        Time = Time - 4;
        t_max = min(size(Time, 2), size(H1, 2));
        mesh(Time(:, 1:t_max), 1:30, H1(:, 1:t_max));
        xlim([0, max(Time)]);
        ylim([1, 30])
        caxis([0, 40])
        xlabel('Time (seconds)')
        ylabel('Subcarrier index')
        colorbar;
        colormap(jet);
        %set(gca,'FontSize', 16);

    end
end
