function [segment_data, segment_data_stft, info] = step04_segment_features(H, config)
% Segment one continuous, uniformly sampled feature block.
segment_data = [];
segment_data_stft = [];
info = struct('raw_starts', [], 'stft_starts', [], 'peak_frames', [], 'config', config);
fs = config.sample_rate;
raw_length = round(fs * config.segment_duration);
nlinks = config.tx_count * config.rx_count;
ngroups = 2 + config.relative_phase;
filename = 'continuous feature block';
first = config.trim_start_samples;
last = size(H, 2) - config.trim_end_samples;
if last < first
    warning('wifi:UnusableRecord', 'Record too short after trimming: %s', filename);
    return;
end
H = H(:, first:last);
if size(H, 2) < config.stft_window || (size(H, 2) - 1) / fs <= config.peak_distance || any(~isfinite(H(:)))
    warning('wifi:UnusableRecord', 'Record too short or nonfinite: %s', filename);
    return;
end
[contour, actual_frame_times] = step03_detector_curve(H(1:config.detector_rows, :), config);
if isempty(contour)
    warning('wifi:NoSignalEnergy', 'No finite spectral energy: %s', filename);
    return;
end
if config.filter_features
    H = step03_func_bandpass_filter(H, fs, config.filter_cutoffs);
end
Time = (0:size(H, 2) - 1) / fs;
if isfield(config, 'time_mapping') && strcmp(config.time_mapping, 'stft_centers')
    frame_spacing = (config.stft_window - config.stft_overlap) / fs;
    frame_times = actual_frame_times;
    distance_frames = config.peak_distance / frame_spacing;
else
    frame_spacing = max(Time) / numel(contour);
    frame_times = (0:numel(contour) - 1) * frame_spacing;
    distance_frames = size(contour, 2) / max(Time) * config.peak_distance;
end
if distance_frames >= numel(contour),
    return;
end
[~, peaks] = findpeaks(contour, 'MinPeakDistance', distance_frames, ...
                      'MinPeakHeight', config.peak_height, 'MinPeakProminence', config.peak_prominence);
if isempty(config.stft_half_frames)
    half_frames = round((config.segment_duration / 2) / frame_spacing);
else
    half_frames = config.stft_half_frames;
end
for peak = peaks(:)'
    [~, center] = min(abs(Time - frame_times(peak)));
    start = center - raw_length / 2;
    finish = center + raw_length / 2 - 1;
    start_spec = peak - half_frames;
    finish_spec = peak + half_frames;
    if start < 1 || finish > size(H, 2) || start_spec < 1 || finish_spec > numel(contour),
        continue;
    end
    info.raw_starts(end + 1) = start;
    info.stft_starts(end + 1) = start_spec;
    info.peak_frames(end + 1) = peak;
end
if isempty(info.raw_starts),
    return;
end
count = numel(info.raw_starts);
rows_per_group = 30 * nlinks;
segment_data = zeros(count, rows_per_group, raw_length, ngroups);
frames = 2 * half_frames + 1;
if strcmp(config.stft_layout, 'flattened_groups')
    segment_data_stft = zeros(count, rows_per_group, config.stft_bins * frames, ngroups);
elseif strcmp(config.stft_layout, 'frequency_time_stream')
    segment_data_stft = zeros(count, config.stft_bins, frames, size(H, 1));
else
    error('wifi:Layout', 'Unknown STFT layout: %s', config.stft_layout);
end
for group = 1:ngroups
    rows = (group - 1) * rows_per_group + (1:rows_per_group);
    for j = 1:count
        start = info.raw_starts(j);
        segment_data(j, :, :, group) = H(rows, start:start + raw_length - 1);
    end
end
for row = 1:size(H, 1)
    S = abs(spectrogram(H(row, :), config.stft_window, config.stft_overlap, config.stft_nfft));
    for j = 1:count
        start = info.stft_starts(j);
        piece = S(1:config.stft_bins, start:start + frames - 1);
        if strcmp(config.stft_layout, 'flattened_groups')
            group = floor((row - 1) / rows_per_group) + 1;
            group_row = mod(row - 1, rows_per_group) + 1;
            segment_data_stft(j, group_row, :, group) = piece(:);
        else
            segment_data_stft(j, :, :, row) = piece;
        end
    end
end
end
