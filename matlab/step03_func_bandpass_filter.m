function [input] = step03_func_bandpass_filter(input, sample_rate, Scut_off)
% Define bandpass filter
Norm_freq = Scut_off / (sample_rate) * 2;
[b, a] = butter(2, Norm_freq, 'bandpass');

for i = 1:size(input, 1)
    input(i, :) = filter(b, a, input(i, :));
end

end
