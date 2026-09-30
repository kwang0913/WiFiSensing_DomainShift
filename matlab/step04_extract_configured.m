function [segment_data, segment_data_stft, info] = step04_extract_configured(filename, config, read_from_file)
% Explicit feature pipeline. Never writes CSI caches or debug files.
% Timestamp mode is default; legacy_resize explicitly preserves old cache behavior.
segment_data = [];
segment_data_stft = [];
info = struct('raw_starts', [], 'stft_starts', [], 'peak_frames', [], 'config', config);
fs = config.sample_rate;
validateattributes(config.filter_features, {'logical'}, {'scalar'});
validateattributes(config.segment_duration, {'numeric'}, {'scalar', 'positive', 'finite'});
raw_length = round(fs * config.segment_duration);
if mod(raw_length, 2) ~= 0
    error('wifi:SegmentLength', 'Segment duration must yield an even number of samples.');
end
if isfield(config, 'alignment_mode') && strcmp(config.alignment_mode, 'timestamp')
    if ~read_from_file
        error('wifi:UnalignedCache', 'Timestamp alignment requires the original DAT; old H caches lack packet masks and time axes. Set read_from_file=true or explicitly choose legacy_resize.');
    end
    [chunks, timing] = step02_read_aligned_features(filename, config);
    info.alignment = timing;
    info.block_ids = [];
    info.raw_start_seconds = [];
    info.stft_start_seconds = [];
    local = config;
    local.trim_start_samples = 1;
    local.trim_end_samples = 0;
    for block = 1:numel(chunks)
        [a, b, details] = step04_segment_features(chunks(block).features, local);
        if isempty(a),
            continue;
        end
        if isempty(segment_data),
            segment_data = a;
            segment_data_stft = b;
        else,
            segment_data = cat(1, segment_data, a);
            segment_data_stft = cat(1, segment_data_stft, b);
        end
        info.raw_starts = [info.raw_starts details.raw_starts];
        info.stft_starts = [info.stft_starts details.stft_starts];
        info.peak_frames = [info.peak_frames details.peak_frames];
        info.block_ids = [info.block_ids repmat(block, 1, numel(details.raw_starts))];
        info.raw_start_seconds = [info.raw_start_seconds chunks(block).start_seconds + (details.raw_starts - 1) / fs];
        hop = (config.stft_window - config.stft_overlap) / fs;
        info.stft_start_seconds = [info.stft_start_seconds chunks(block).start_seconds + (details.stft_starts - 1) * hop + config.stft_window / (2 * fs)];
    end
    return;
end
nlinks = config.tx_count * config.rx_count;
amplitude = cell(1, nlinks);
complex_csi = cell(1, nlinks);
if read_from_file
    for tx = 1:config.tx_count
        for rx = 1:config.rx_count
            k = (tx - 1) * config.rx_count + rx;
            [~, complex_csi{k}, ~, amplitude{k}, ~] = step02_get_allkinds_scaled_csi(filename, tx, rx);
        end
    end
else
    [folder, stem] = fileparts(filename);
    cached = load(fullfile(folder, [stem '.mat']));
    for k = 1:nlinks
        hn = sprintf('H%d', k);
        zn = sprintf('H_complex_antenna%d', k);
        if ~isfield(cached, hn) || ~isfield(cached, zn)
            error('wifi:CacheStreams', 'Profile needs %d links; cache lacks %s or %s.', nlinks, hn, zn);
        end
        amplitude{k} = cached.(hn);
        complex_csi{k} = cached.(zn);
    end
end
if any(cellfun(@isempty, amplitude)) || any(cellfun(@isempty, complex_csi))
    warning('wifi:EmptyRecord', 'No CSI samples: %s', filename);
    return;
end
n = size(amplitude{1}, 2);
pairs = [1 2; 1 3; 2 3];
relative_amplitude = cell(1, nlinks);
relative_phase = cell(1, nlinks);
for k = 1:nlinks
    if size(amplitude{k}, 1) ~= 30 || size(complex_csi{k}, 1) ~= 30
        error('wifi:Subcarriers', 'Expected 30 subcarriers per link.');
    end
    amplitude{k} = imresize(amplitude{k}, [30 n]);
    pair = pairs(mod(k - 1, 3) + 1, :) + floor((k - 1) / 3) * 3;
    product = complex_csi{pair(1)} .* conj(complex_csi{pair(2)});
    relative_amplitude{k} = imresize(abs(product), [30 n]);
    if config.relative_phase
        relative_phase{k} = imresize(unwrap(angle(product), [], 2), [30 n]);
    end
end
if config.relative_phase
    H = vertcat(amplitude{:}, relative_phase{:}, relative_amplitude{:});
    ngroups = 3;
else
    H = vertcat(amplitude{:}, relative_amplitude{:});
    ngroups = 2;
end
[segment_data, segment_data_stft, info] = step04_segment_features(H, config);
end
