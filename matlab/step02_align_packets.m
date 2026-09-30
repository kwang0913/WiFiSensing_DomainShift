function [chunks, info] = step02_align_packets(z, timestamps, packet_ids, config)
% z: [30*tx*rx, packets], scaled complex CSI in common packet order.
% Build real features per packet before interpolation, avoiding common-phase
% rotation artifacts from interpolating raw complex CSI across packets.
chunks = struct('features', {}, 'start_seconds', {}, 'epoch', {});
count = numel(timestamps);
rows = config.tx_count * config.rx_count * 30;
assert(size(z, 1) == rows && size(z, 2) == count && numel(packet_ids) == count);
validateattributes(config.max_interpolation_gap, {'numeric'}, {'scalar', 'positive', 'finite'});
info = struct('input_packets', count, 'valid_packet_ids', [], 'invalid_packet_ids', [], ...
            'duplicate_packet_ids', [], 'reset_packet_ids', [], 'wrap_count', 0, 'blocks', []);
if count == 0,
    return;
end
% Reject nonfinite or zero complex coefficients jointly; phase at zero is undefined.
valid = all(isfinite(z) & abs(z) > 0, 1) & isfinite(timestamps) & timestamps >= 0 & timestamps < 2^32;
epoch = ones(1, count);
time = zeros(1, count);
offset = 0;
epoch_no = 1;
origin = NaN;
previous = NaN;
for i = 1:count
    low = timestamps(i);
    if ~isfinite(low) || low < 0 || low >= 2^32,
        valid(i) = false;
        continue;
    end
    if isfinite(previous)
        delta = low - previous;
        if delta < -2^31
            offset = offset + 2^32;
            info.wrap_count = info.wrap_count + 1;
        elseif delta < 0
            epoch_no = epoch_no + 1;
            offset = 0;
            origin = low;
            info.reset_packet_ids(end + 1) = packet_ids(i);
        elseif delta == 0
            valid(i) = false;
            info.duplicate_packet_ids(end + 1) = packet_ids(i);
        end
    elseif ~isfinite(origin),
        origin = low;
    end
    time(i) = (low + offset - origin) / 1e6;
    epoch(i) = epoch_no;
    previous = low;
end
info.valid_packet_ids = packet_ids(valid);
info.invalid_packet_ids = packet_ids(~valid);
idx = find(valid);
if numel(idx) < 2,
    return;
end
cuts = [1 find(diff(epoch(idx)) ~= 0 | diff(time(idx)) > config.max_interpolation_gap | diff(time(idx)) <= 0) + 1 numel(idx) + 1];
first_epoch = epoch(idx(1));
last_epoch = epoch(idx(end));
last_time = time(idx(end));
block_info = struct('epoch', {}, 'start_seconds', {}, 'end_seconds', {}, 'source_packet_ids', {}, 'max_source_gap', {});
for b = 1:numel(cuts) - 1
    chosen = idx(cuts(b):cuts(b + 1) - 1);
    if numel(chosen) < 2,
        continue;
    end
    t = time(chosen);
    ep = epoch(chosen(1));
    lo = t(1);
    hi = t(end);
    if ep == first_epoch,
        lo = max(lo, (config.trim_start_samples - 1) / config.sample_rate);
    end
    if ep == last_epoch,
        hi = min(hi, last_time - config.trim_end_samples / config.sample_rate);
    end
    first_tick = ceil(lo * config.sample_rate - 1e-8);
    last_tick = floor(hi * config.sample_rate + 1e-8);
    if last_tick - first_tick + 1 < max(config.stft_window, round(config.segment_duration * config.sample_rate)),
        continue;
    end
    grid = (first_tick:last_tick) / config.sample_rate;
    % Tolerate only rounding at an endpoint; interp1 never extrapolates.
    grid = min(max(grid, t(1)), t(end));
    zz = z(:, chosen);
    amplitude = 20 * log10(abs(zz));
    pairamp = zeros(size(amplitude));
    phase = zeros(size(amplitude));
    pairs = [1 2; 1 3; 2 3];
    for tx = 1:config.tx_count
        for p = 1:3
            dest = ((tx - 1) * 3 + p - 1) * 30 + (1:30);
            ra = ((tx - 1) * 3 + pairs(p, 1) - 1) * 30 + (1:30);
            rb = ((tx - 1) * 3 + pairs(p, 2) - 1) * 30 + (1:30);
            product = zz(ra, :) .* conj(zz(rb, :));
            pairamp(dest, :) = abs(product);
            if config.relative_phase,
                phase(dest, :) = unwrap(angle(product), [], 2);
            end
        end
    end
    if config.relative_phase,
        H = [amplitude; phase; pairamp];
    else,
        H = [amplitude; pairamp];
    end
    H = interp1(t, H', grid, 'linear')';
    if any(~isfinite(H(:))),
        error('wifi:Interpolation', 'Nonfinite aligned features.');
    end
    k = numel(chunks) + 1;
    chunks(k) = struct('features', H, 'start_seconds', grid(1), 'epoch', ep);
    block_info(k) = struct('epoch', ep, 'start_seconds', grid(1), 'end_seconds', grid(end), ...
                         'source_packet_ids', packet_ids(chosen), 'max_source_gap', max(diff(t)));
end
info.blocks = block_info;
end
