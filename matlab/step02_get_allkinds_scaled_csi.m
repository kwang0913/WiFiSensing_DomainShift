% GET_ALLKINDS_SCALED_CSI obtains available csi and filters out unusable(-inf) csi.
% function [csi_trace,H_complex,Time,H,Rate] = step02_get_allkinds_scaled_csi(data_file,tx,rx,low,high)
function [csi_trace, H_complex, Time, H, Rate] = step02_get_allkinds_scaled_csi(data_file, tx, rx)
%tx: trasmitting antenna no.
%rx: receiving antenna no. (tx-rx is a antenna pair you want to extract)
%low: MCS_rate lowest rate for filter purpose
%high: MCS_rate highest rate for filter purpose
%Rate: MCS rate for each packets
%H: CSI data

% Read the data sample
csi_trace = step01_read_bf_file(data_file);
pkt_length = length(csi_trace);
% Restore all CSI data
%Rssi = zeros(1,pkt_length);
Time = zeros(1, pkt_length);
count = 0;
%initialization all the variable with all value equal to zero
H_raw = zeros(30, pkt_length);
Rate = zeros(1, pkt_length);
H = zeros(30, pkt_length);
H_complex = zeros(30, pkt_length);

% for each received packet, extract CSI stuct
for i = 1:pkt_length
    count = count + 1;
    csi_entry = csi_trace{i};
    % extract received signal strength
    %Rssi(1,i) = step02_get_total_rss(csi_entry);
    % extract MCS rate
    Rate(1, i) = csi_entry.rate;
    % extract time
    Time(1, i) = csi_entry.timestamp_low;
    csi = step02_scale_packet(csi_entry);
    % Extract the 1st transmitter antenna data
    T1 = csi(tx, :, :);
    H_temp = squeeze(T1).';
    % Extract the CSI in complex form
    H_complex(:, i) = H_temp(:, rx);
    W = db(abs(squeeze(T1).')); % 30*3 CSI matrix
    % Extract the 1st receiver antenna data
    R1 = W(:, rx);
    H_raw(:, i) = R1;
    H(:, i) = R1;
    % eliminate CSI that corresponding to unstable MCS rate
    %     if Rate(1,i) >= low && Rate(1,i) <= high
    %         H(:,i) = R1;
    %     end
end
%Time = (Time - Time(1))./1000000;
% Filter out the unusable csi
[x, y] = find(H_raw == -inf);
H_raw(:, y) = [];
[x, y] = find(H == -inf);
H(:, y) = [];
Time(y) = [];
[x, y] = find(H == 0);
H(:, y) = [];
end
