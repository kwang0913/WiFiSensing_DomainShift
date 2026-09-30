function [ret, info] = step01_read_bf_file(filename)
% Read Intel CSI records safely, retaining complete packets before a truncated tail.
% Decoder/scaling conventions originate in Daniel Halperin's CSI Tool.
f = fopen(filename, 'rb');
if f < 0,
    error('wifi:OpenDAT', 'Cannot open %s', filename);
end
cleanup = onCleanup(@() fclose(f));
fseek(f, 0, 'eof');
bytes_total = ftell(f);
fseek(f, 0, 'bof');
ret = cell(ceil(bytes_total / 95), 1);
count = 0;
record = 0;
info = struct('truncated', false, 'malformed_records', 0, 'record_indices', [], ...
            'tail_reason', '', 'discarded_tail_bytes', 0);
while ftell(f) < bytes_total
    record_start = ftell(f);
    field_len = fread(f, 1, 'uint16', 0, 'ieee-be');
    if isempty(field_len) || field_len < 1
        info.truncated = true;
        info.discarded_tail_bytes = bytes_total - record_start;
        info.tail_reason = 'invalid_or_incomplete_length';
        if ~isempty(field_len) && field_len == 0
            all_zero = true;
            while ftell(f) < bytes_total
                tail = fread(f, 1024 * 1024, 'uint8=>uint8');
                if any(tail ~= 0),
                    all_zero = false;
                    break;
                end
            end
            if all_zero,
                info.tail_reason = 'zero_filled_tail';
            end
        end
        break;
    end
    payload = fread(f, field_len, 'uint8=>uint8');
    if numel(payload) ~= field_len
        info.truncated = true;
        info.tail_reason = 'incomplete_payload';
        info.discarded_tail_bytes = bytes_total - record_start;
        break;
    end
    if payload(1) ~= 187,
        continue;
    end
    record = record + 1;
    bytes = payload(2:end);
    if numel(bytes) < 20,
        info.malformed_records = info.malformed_records + 1;
        continue;
    end
    nr = double(bytes(9));
    nt = double(bytes(10));
    if nr < 1 || nr > 3 || nt < 1 || nt > 3,
        info.malformed_records = info.malformed_records + 1;
        continue;
    end
    expected = ceil(30 * (nr * nt * 16 + 3) / 8);
    declared = double(bytes(17)) + 256 * double(bytes(18));
    if declared ~= expected || numel(bytes) < 20 + expected
        info.malformed_records = info.malformed_records + 1;
        continue;
    end
    entry = step01_read_bfee(bytes);
    % Compute uint32 time in double to avoid platform-dependent signed shifts.
    entry.timestamp_low = double(bytes(1:4))' * [1; 256; 65536; 16777216];
    perm = entry.perm(1:nr);
    if nr > 1
        if ~isequal(sort(perm(:))', (1:nr))
            info.malformed_records = info.malformed_records + 1;
            continue;
        end
        entry.csi(:, perm, :) = entry.csi(:, 1:nr, :);
    end
    count = count + 1;
    ret{count} = entry;
    info.record_indices(count) = record;
end
ret = ret(1:count);
if strcmp(info.tail_reason, 'zero_filled_tail')
    warning('wifi:ZeroFilledTail', 'Ignored %d trailing zero bytes in %s; retained %d complete CSI packets. Cause of zero-filled tail is unknown.', info.discarded_tail_bytes, filename, count);
elseif info.truncated
    warning('wifi:TruncatedDAT', 'Invalid or incomplete tail (%d bytes) ignored in %s; retained %d complete CSI packets.', info.discarded_tail_bytes, filename, count);
end
if info.malformed_records > 0,
    warning('wifi:MalformedDAT', 'Skipped %d malformed CSI records in %s.', info.malformed_records, filename);
end
end
