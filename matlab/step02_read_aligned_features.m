function [chunks, info] = step02_read_aligned_features(filename, config)
% Decode once and retain a common packet index across all selected links.
[trace, decoder] = step01_read_bf_file(filename);
n = numel(trace);
rows = 30 * config.tx_count * config.rx_count;
z = complex(nan(rows, n));
timestamps = nan(1, n);
bfee_count = nan(1, n);
for i = 1:n
    entry = trace{i};
    timestamps(i) = entry.timestamp_low;
    bfee_count(i) = entry.bfee_count;
    if entry.Ntx < config.tx_count || entry.Nrx < config.rx_count,
        continue;
    end
    scaled = step02_scale_packet(entry);
    for tx = 1:config.tx_count
        for rx = 1:config.rx_count
            r = ((tx - 1) * config.rx_count + rx - 1) * 30 + (1:30);
            z(r, i) = reshape(scaled(tx, rx, :), 30, 1);
        end
    end
end
[chunks, info] = step02_align_packets(z, timestamps, decoder.record_indices, config);
info.decoder = decoder;
info.timestamp_low = timestamps;
info.bfee_count = bfee_count;
info.packet_record_indices = decoder.record_indices;
end
