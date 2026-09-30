function [segment_data, segment_data_stft, info] = step04_func_segmentation_var_slot_freq(filename, dist, seg_dur, thres, bandpass_filter, read_from_file)
% Profile form: step04_func_segmentation_var_slot_freq(filename, config, read_from_file).
% Existing six-argument callers retain the previous implementation.
info = struct();
if isstruct(dist)
    [segment_data, segment_data_stft, info] = step04_extract_configured(filename, dist, seg_dur);
    return;
end
% addpath(genpath('C:\Users\mason\Dropbox\CSI_Authentication\Code\matlab\'));

% Always return both outputs, including records with no usable segments.
segment_data = [];
segment_data_stft = [];

debug = 0;
relative_phase = 0;
ToF_debug = 0;

freq_featre = 1;

% Selected principal components
pca_denoise = 0;
nFirst = 2;
nLast = 45;

% Apply bandpass filter
% bandpass_filter = 1;
Scut_off = [0.8, 200];

N = 30;
energy = 3000;

sample_rate = 1000;
%sample_win=4;
sample_win = 8;

% Extracting raw CSI from .dat files
filename_save = [filename(1, 1:end - 4), '.mat'];
if read_from_file == 1
    [~, H_complex_antenna1, ~, H1, ~] = step02_get_allkinds_scaled_csi(filename, 1, 1);
    % disp('1-1 Antenna Pair');
    [~, H_complex_antenna2, ~, H2, ~] = step02_get_allkinds_scaled_csi(filename, 1, 2);
    % disp('1-2 Antenna Pair');
    [~, H_complex_antenna3, ~, H3, ~] = step02_get_allkinds_scaled_csi(filename, 1, 3);
    % disp('1-3 Antenna Pair');
    %[~,H_complex_antenna4,~,H4,~] = step02_get_allkinds_scaled_csi(filename,2,1);
    % disp('2-1 Antenna Pair');
    %[~,H_complex_antenna5,~,H5,~] = step02_get_allkinds_scaled_csi(filename,2,2);
    % disp('2-2 Antenna Pair');
    %[~,H_complex_antenna6,~,H6,~] = step02_get_allkinds_scaled_csi(filename,2,3);
    % disp('2-3 Antenna Pair');
    %[~,H_complex_antenna7,~,H7,~] = step02_get_allkinds_scaled_csi(filename,3,1);
    % disp('3-1 Antenna Pair');
    %[~,H_complex_antenna8,~,H8,~] = step02_get_allkinds_scaled_csi(filename,3,2);
    % disp('3-2 Antenna Pair');
    %[~,H_complex_antenna9,~,H9,~] = step02_get_allkinds_scaled_csi(filename,3,3);
    % disp('3-3 Antenna Pair');
    % save(filename_save,'H_complex_antenna1','H1','H_complex_antenna2','H2','H_complex_antenna3','H3','H_complex_antenna4','H4','H_complex_antenna5','H5','H_complex_antenna6','H6','H_complex_antenna7','H7','H_complex_antenna8','H8','H_complex_antenna9','H9')
    save(filename_save, 'H_complex_antenna1', 'H1', 'H_complex_antenna2', 'H2', 'H_complex_antenna3', 'H3')
else
    load(filename_save);
end

if isempty(H1) || isempty(H2) || isempty(H3)
    warning('wifi:EmptyRecord', 'No CSI samples: %s', filename);
    return;
end

% Resize all measurements to make the dimensions consistent
x_min = 30;
y_min = size(H1, 2);

H1 = imresize(H1, [x_min y_min]);
H2 = imresize(H2, [x_min y_min]);
H3 = imresize(H3, [x_min y_min]);
%H4 = imresize(H4,[x_min y_min]);
%H5 = imresize(H5,[x_min y_min]);
%H6 = imresize(H6,[x_min y_min]);
%H7 = imresize(H7,[x_min y_min]);
%H8 = imresize(H8,[x_min y_min]);
%H9 = imresize(H9,[x_min y_min]);

relative_amplitude1 = abs(H_complex_antenna1 .* conj(H_complex_antenna2));
relative_amplitude2 = abs(H_complex_antenna1 .* conj(H_complex_antenna3));
relative_amplitude3 = abs(H_complex_antenna2 .* conj(H_complex_antenna3));

relative_amplitude1 = imresize(relative_amplitude1, [x_min y_min]);
relative_amplitude2 = imresize(relative_amplitude2, [x_min y_min]);
relative_amplitude3 = imresize(relative_amplitude3, [x_min y_min]);

if relative_phase == 1
    relative_phase1 = unwrap(angle(H_complex_antenna1 .* conj(H_complex_antenna2)), [], 2);
    relative_phase2 = unwrap(angle(H_complex_antenna1 .* conj(H_complex_antenna3)), [], 2);
    relative_phase3 = unwrap(angle(H_complex_antenna2 .* conj(H_complex_antenna3)), [], 2);

    %     relative_phase4 = unwrap(angle(H_complex_antenna4.*conj(H_complex_antenna5)), [], 2);
    %     relative_phase5 = unwrap(angle(H_complex_antenna4.*conj(H_complex_antenna6)), [], 2);
    %     relative_phase6 = unwrap(angle(H_complex_antenna5.*conj(H_complex_antenna6)), [], 2);
    %
    %     relative_phase7 = unwrap(angle(H_complex_antenna7.*conj(H_complex_antenna8)), [], 2);
    %     relative_phase8 = unwrap(angle(H_complex_antenna7.*conj(H_complex_antenna9)), [], 2);
    %     relative_phase9 = unwrap(angle(H_complex_antenna8.*conj(H_complex_antenna9)), [], 2);
    %

    %
    %     relative_amplitude4 = abs(H_complex_antenna4.*conj(H_complex_antenna5));
    %     relative_amplitude5 = abs(H_complex_antenna4.*conj(H_complex_antenna6));
    %     relative_amplitude6 = abs(H_complex_antenna5.*conj(H_complex_antenna6));
    %
    %     relative_amplitude7 = abs(H_complex_antenna7.*conj(H_complex_antenna8));
    %     relative_amplitude8 = abs(H_complex_antenna7.*conj(H_complex_antenna9));
    %     relative_amplitude9 = abs(H_complex_antenna8.*conj(H_complex_antenna9));
    %     relative_phase1 = relative_phase1(:, 3*sample_rate:(end-3*sample_rate));
    %     relative_phase2 = relative_phase2(:, 3*sample_rate:(end-3*sample_rate));
    %     relative_phase3 = relative_phase3(:, 3*sample_rate:(end-3*sample_rate));
    relative_phase1 = imresize(relative_phase1, [x_min y_min]);
    relative_phase2 = imresize(relative_phase2, [x_min y_min]);
    relative_phase3 = imresize(relative_phase3, [x_min y_min]);
    %     relative_phase4 = imresize(relative_phase4,[x_min y_min]);
    %     relative_phase5 = imresize(relative_phase5,[x_min y_min]);
    %     relative_phase6 = imresize(relative_phase6,[x_min y_min]);
    %     relative_phase7 = imresize(relative_phase7,[x_min y_min]);
    %     relative_phase8 = imresize(relative_phase8,[x_min y_min]);
    %     relative_phase9 = imresize(relative_phase9,[x_min y_min]);

    %     relative_amplitude4 = imresize(relative_amplitude4,[x_min y_min]);
    %     relative_amplitude5 = imresize(relative_amplitude5,[x_min y_min]);
    %     relative_amplitude6 = imresize(relative_amplitude6,[x_min y_min]);
    %     relative_amplitude7 = imresize(relative_amplitude7,[x_min y_min]);
    %     relative_amplitude8 = imresize(relative_amplitude8,[x_min y_min]);
    %     relative_amplitude9 = imresize(relative_amplitude9,[x_min y_min]);
end

% Combine all CSI streams
if relative_phase == 1
    % H = [H1; H2; H3; H4; H5; H6; H7; H8; H9; relative_phase1; relative_phase2; relative_phase3; relative_phase4; relative_phase5; relative_phase6; relative_phase7; relative_phase8; relative_phase9; relative_amplitude1; relative_amplitude2; relative_amplitude3; relative_amplitude4; relative_amplitude5; relative_amplitude6; relative_amplitude7; relative_amplitude8; relative_amplitude9];
    H = [H1; H2; H3; relative_phase1; relative_phase2; relative_phase3; relative_amplitude1; relative_amplitude2; relative_amplitude3];
else
    % H = [H1; H2; H3; H4; H5; H6; H7; H8; H9];
    H = [H1; H2; H3; relative_amplitude1; relative_amplitude2; relative_amplitude3];
end
% Remove some samples corresponding to truning on and closing off the WiFi
if strcmp(filename, fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data/raw_data/appleman_0514/data/0514_hanyi_sitting_a1.dat'))
    H = H(:, 4.0 * sample_rate:(end - 8.0 * sample_rate));
elseif strcmp(filename, fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data/raw_data/appleman_0521/data/0521_yalei_raisingarm_a1.dat'))
    H = H(:, 4.0 * sample_rate:(end - 48.0 * sample_rate));
elseif strcmp(filename, fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data/raw_data/0305_morning/data/yilin_0305_document_loc_d3.dat'))
    H = H(:, 4.0 * sample_rate:(end - 20.0 * sample_rate));
elseif strcmp(filename, fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data/raw_data/0305_morning/data/xiangyu_0305_keyboard_loc_d4.dat'))
    H = H(:, 4.0 * sample_rate:(end - 12.0 * sample_rate));
elseif strcmp(filename, fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data/raw_data/0305_morning/data/yilin_0305_keyboard_loc_d1.dat'))
    H = H(:, 4.0 * sample_rate:(end - 12.0 * sample_rate));
elseif strcmp(filename, fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data/raw_data/appleman_0623/data/0623_yaleibi_raisingarm_B3.dat'))
    H = H(:, 4000:50000);
else
    % H = H(:, 4.0*sample_rate:(end-8.0*sample_rate));
    H = H(:, 0.1 * sample_rate:(end - 4 * sample_rate));
end
% Reject unusable records before filtering, PCA, and spectrogram calls.
if size(H, 2) < round(sample_rate / sample_win) || ...
        (size(H, 2) - 1) / sample_rate <= dist || any(~isfinite(H(:)))
    warning('wifi:UnusableRecord', 'Record is too short or nonfinite: %s', filename);
    return;
end

stackedArray = H;
output_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'stack_arary');
if ~isfolder(output_dir),
    mkdir(output_dir);
end
save(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'stack_arary/test2.mat'), 'stackedArray');
if debug == 1
    fig_tmp = figure;
    hold on
    surf(H(1:180, :), 'EdgeColor', 'none');
    xlim([1 size(H, 2)])
    ylim([90 120])
    xlabel('WiFi Packets')
    ylabel('Subcarriers')
    colormap(jet);
    % caxis([25 40]);
    set(gca, 'FontSize', 16)
    colorbar
    % close all;
end

% H = H(:, 1:(size(H,2)-mod(size(H,2), 1000))+1);

% Remove signals of frequency bands irrevalent to human activities
H_seg = step03_func_bandpass_filter(H(1:90, :), sample_rate, [2.5, 100]);

if bandpass_filter == 1
    H = step03_func_bandpass_filter(H, sample_rate, Scut_off);
end

if debug == 1
    fig_tmp = figure;
    hold on
    surf(H(1:180, :), 'EdgeColor', 'none');
    xlim([1 size(H, 2)])
    ylim([1 270])
    xlabel('WiFi Packets')
    ylabel('Subcarriers')
    caxis([-1.5 5]);
    colormap(jet);
    set(gca, 'FontSize', 16)
    colorbar
    close all;
end

% Time for each WiFi packets
Time = 0:1 / sample_rate:1 / sample_rate * (size(H_seg, 2) - 1);

if debug == 1
    fig1 = figure;
    set(fig1, 'Position', [300, 300, 1200, 300]);
    hold on;
    surf(Time, 1:90, H_seg, 'EdgeColor', 'none');
    ylim([1, 90])
    xlim([0, max(Time)])
    caxis([-1 5]);
    colorbar;
    colormap(jet);
    %ylim([2 100]);
    ylabel('Subcarriers');
    xlabel('WiFi packets');
    title('Bandpass filter')
    close all
end

if pca_denoise == 1
    % Denoise H for feature extraction
    idx = 1:100:size(H, 2);
    H_hat = H;
    for i = idx(1, 1:end - 1)
        X = H_hat(:, i:i + 100)';
        mu = mean(X);
        [eigenvectors, scores] = pca(X);
        Xhat = scores(:, nFirst:nLast) * eigenvectors(:, nFirst:nLast)';
        Xhat = bsxfun(@plus, Xhat, mu);
        Xhat = Xhat + mu;
        H_hat(:, i:i + 100) = Xhat';
    end
end

% Denoise H for data segmentation, using a fixed set of principal
% components
idx = 1:1000:size(H_seg, 2);
H_hat_seg = H_seg;
for i = idx(1, 1:end - 1)
    X = H_hat_seg(:, i:i + 1000)';
    mu = mean(X);
    [eigenvectors, scores] = pca(X);
    Xhat = scores(:, 8:90) * eigenvectors(:, 8:90)';
    Xhat = bsxfun(@plus, Xhat, mu);
    Xhat = Xhat + mu;
    H_hat_seg(:, i:i + 1000) = Xhat';
end

if debug == 1
    fig1 = figure;
    set(fig1, 'Position', [300, 300, 1200, 300]);
    hold on;
    surf(Time, 1:90, H_hat_seg, 'EdgeColor', 'none');
    ylim([1, 90])
    xlim([0, max(Time)])
    caxis([-1 5]);
    colorbar;
    colormap(jet);
    %ylim([2 100]);
    ylabel('Subcarriers');
    xlabel('WiFi packets');
    title('PCA denoising + Bandpass filter')
    % close all
end

% H_seg = H_hat_seg;
% if pca_denoise == 1
%     H = H_hat;
% end
%
% stackedArray = H_seg;
% save('D:\Dropbox\conformal_prediction\stack_arary\test.mat', 'stackedArray');
% stack2 = [];
% Norm_freq=Scut_off/(sample_rate)*2;
% [b,a] = butter(2,Norm_freq,'bandpass');
% temp = freqz(b,a);
% signal_2 = H_seg;
% for i = 1:N
%     sub_signal = signal_2(i,:);
%
%     % Butter worth filter
%     % ================
%     sub_signal = filter(b,a, sub_signal);
%     [s_amplitude,~,~,ps_amplitude] = spectrogram(sub_signal,round(sample_rate/sample_win),round(sample_rate/sample_win/2),sample_rate,sample_rate);
%      if isempty(stack2)
%             stack2 = zeros (size(s_amplitude,1), size(s_amplitude,2));
%     end
%     stack2 = stack2 + abs(s_amplitude);
%
% end

% fig5 = figure;
% hold on;
% set(fig5, 'Position', [500,500,1200,300]);
% x_time = 0:max(Time)/size(stack2,2):max(Time)/size(stack2,2)*(size(stack2,2)-1);
% surf(x_time,1:size(stack2,1),stack2,'EdgeColor','none');
% caxis([0 energy]);
% colorbar;
% colormap(jet);
% ylim([2 100]);
% xlim([min(x_time) max(x_time)]);
% ylabel('Frequency (Hz)');
% xlabel('Time');
% title('Accumulated Amplitude Spectrogram');

% Compute spectrogram of N subcarriers
stack2 = [];

for i = 1:90
    sub_signal = H_seg(i, :);
    [s_amplitude, ~, ~, ~] = spectrogram(sub_signal, round(sample_rate / sample_win), round(sample_rate / sample_win / 2), sample_rate);
    if isempty(stack2)
        stack2 = zeros(size(s_amplitude, 1), size(s_amplitude, 2));
    end
    stack2 = stack2 + abs(s_amplitude);
end

if freq_featre == 1
    % Compute spectrogram of each individual subcarriers
    for i = 1:size(H, 1)
        sub_signal = H(i, :);
        [s_amplitude, ~, ~, ~] = spectrogram(sub_signal, round(sample_rate / sample_win), round(sample_rate / sample_win / 2), sample_rate);
        spec_all(i, :, :) = abs(s_amplitude(1:64, :));
    end
end

if isempty(stack2) || any(~isfinite(stack2(:))) || ~any(stack2(:) > 0)
    warning('wifi:NoSignalEnergy', 'No finite spectral energy: %s', filename);
    return;
end

freq_contour = sum(stack2 / mean(mean(stack2)), 1);
freq_contour = smoothdata(freq_contour, 'movmedian', 5);
if any(~isfinite(freq_contour)) || max(freq_contour) <= 0
    warning('wifi:NoSignalEnergy', 'No usable motion contour: %s', filename);
    return;
end
freq_contour = freq_contour / max(freq_contour);
x_time = 0:max(Time) / size(stack2, 2):max(Time) / size(stack2, 2) * (size(stack2, 2) - 1);

if debug == 1
    fig4 = figure;
    hold on;
    set(fig4, 'Position', [500, 500, 1200, 300]);
    % surf(x_time,1:size(stack2,1),stack2,'EdgeColor','none');
    surf(stack2, 'EdgeColor', 'none');
    %caxis([0 3000]);
    colorbar;
    colormap(jet);
    ylim([2 100]);
    xlim([1 size(stack2, 2)])
    %xlim([min(x_time) max(x_time)]);
    ylabel('Frequency (Hz)');
    xlabel('Frame index');
    title('Spectrogram of CSI Amplitude ');
end

% Metrics for data segmentation
mov_metric = freq_contour;

if debug == 1
    figure
    findpeaks(mov_metric, 'MinPeakDistance', size(mov_metric, 2) / max(Time) * dist, 'MinPeakHeight', thres, 'MinPeakProminence', 0.01);
    %xlim([5000, 80000])
    ylabel('Magnitude of Stacked Spectrograms')
    xlabel('Frame index')
    set(gca, 'FontSize', 16);
    close all
end

findpeaks(mov_metric, 'MinPeakDistance', size(mov_metric, 2) / max(Time) * dist, 'MinPeakHeight', thres, 'MinPeakProminence', 0.01)
[~, locs_amplitude] = findpeaks(mov_metric, 'MinPeakDistance', size(mov_metric, 2) / max(Time) * dist, 'MinPeakHeight', thres, 'MinPeakProminence', 0.01);

% Calculate dynamic threshold and segment CSI of each subcarrier

% Segment 2s data c each peak
% segments = zeros(2, size(locs_amplitude,2));
% segment_data = cell(1, size(locs_amplitude,2));
%segment_data = zeros(size(locs_amplitude,2), 90, sample_rate*(seg_dur)+1);
segment_data = [];
freq_features = [];

num = 1;
for i = 1:size(locs_amplitude, 2)

    peak_time = x_time(locs_amplitude(1, i));
    [~, peak_time_index] = min(abs(Time - peak_time));

    % Determine the start and end of each segment on Hs
    start = peak_time_index - round(sample_rate * seg_dur);
    %start = start + round(sample_rate/sample_win/2);
    ed = peak_time_index + round(sample_rate * seg_dur) - 1;
    %ed = ed + round(sample_rate/sample_win/2);

    % Determine the start and end of each segment on spectrogram
    start_spec = locs_amplitude(1, i) - round(seg_dur / (max(Time) / size(stack2, 2)));
    ed_spec = locs_amplitude(1, i) + round(seg_dur / (max(Time) / size(stack2, 2)));
    %     start_spec = locs_amplitude(1,i) - 10;
    %     ed_spec = locs_amplitude(1,i) + 10;

    % Exclude segmentations with ending index larger than the data length
    if ed > size(Time, 2)
        continue;
    elseif start < 1
        continue;
    elseif ed_spec > size(stack2, 2)
        continue;
    elseif start_spec < 1
        continue;
    end

    % Plot spectrogram

    segments(1:2, i) = [start, ed];
    % segment_data(1,i) = {[H1(:, start:ed); H2(:, start:ed); H3(:, start:ed); relative_phase1(:, start:ed); relative_phase2(:, start:ed); relative_phase3(:, start:ed)]};

    temp = H(:, start:ed);

    % Filter out the outlier segments using variance across subcarriers and
    % select only segments with maximum variance over 1
    % disp([var(var(temp, 0, 1)), mean(var(temp, 0, 1))])
    if debug == 1
        % Mesh plot of the CSI amplitude of all subcarriers
        fig_tmp = figure;
        hold on
        surf(temp(541:810, :), 'EdgeColor', 'none');
        xlim([1 size(temp, 2)])
        ylim([1 270])
        xlabel('WiFi Packets')
        ylabel('Subcarriers')
        %caxis([10 40]);
        caxis([-0.15 0.5]);
        colormap(jet);
        set(gca, 'FontSize', 16)
        title('Relative Amplitude')
        colorbar

        fig_tmp = figure;
        hold on
        surf(temp(271:540, :), 'EdgeColor', 'none');
        xlim([1 size(temp, 2)])
        ylim([1 270])
        xlabel('WiFi Packets')
        ylabel('Subcarriers')
        %caxis([10 40]);
        caxis([-1 3.1]);
        colormap(jet);
        set(gca, 'FontSize', 16)
        title('Relative Phase')
        colorbar

        fig_tmp = figure;
        hold on
        surf(temp(1:270, :), 'EdgeColor', 'none');
        xlim([1 size(temp, 2)])
        ylim([1 270])
        xlabel('WiFi Packets')
        ylabel('Subcarriers')
        % caxis([10 40]);
        % caxis([-1 5]);
        colormap(jet);
        set(gca, 'FontSize', 16)
        colorbar
        pause(1)

        if freq_featre == 1
            % Spectrogram
            seg_spec = squeeze(spec_all(1, 1:100, start_spec:ed_spec));
            figure;
            hold on;
            surf(seg_spec, 'EdgeColor', 'none');
            %caxis([0 energy]);
            colorbar;
            colormap(jet);
            ylim([2 100]);
            xlim([1 21])
            set(gca, 'FontSize', 16)
            ylabel('Frequency (Hz)');
            xlabel('Frame Index');
        end

        close all;
    end

    %if var(var(temp, 0, 1))>0.001
    segment_data(num, :, :) = temp(:, :);
    for car = 1:size(spec_all, 1)
        segment_data_stft(num, :, :, car) = squeeze(spec_all(car, :, start_spec:ed_spec));
    end

    % Extract frequency domain features
    %         temp_freq_features = [];
    %
    %         if freq_featre == 1
    %             for k = 1:size(spec_all,1)
    %                     seg_spec = squeeze(spec_all(k, :, start_spec:ed_spec));
    %                     seg_spec_sum = sum(seg_spec,2)/sum(sum(seg_spec,2));
    %                     temp_freq_features = [temp_freq_features, seg_spec_sum(1:50)];
    % %                     figure
    % %                     plot(seg_spec_sum)
    % %                     xlabel('Frequency (Hz)')
    % %                     ylabel('Energy')
    % %                     set(gca,'FontSize',16)
    % %                     close all;
    %             end
    %             temp_freq_features = temp_freq_features';
    %             t_freq_f = reshape(temp_freq_features, [], 1);
    %             freq_features(num,:) = t_freq_f(:);
    %         end

    num = num + 1;
    %end
end
% close all
end
