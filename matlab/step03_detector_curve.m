function [contour, frame_times] = step03_detector_curve(H, config)
% Detection only: initialize the causal high-pass as a constant prehistory.
% Subtracting each channel's first sample prevents its DC level from producing
% a zero-state startup pulse. Saved feature filtering is independent.
contour = [];
frame_times = [];
if isempty(H) || any(~isfinite(H(:))),
    return;
end
mode = 'constant';
if isfield(config, 'detector_initialization'),
    mode = config.detector_initialization;
end
switch mode
    case 'constant',
        H = H - H(:, 1);
    case 'zero' % Explicit historical detector compatibility.
    otherwise,
        error('wifi:DetectorInitialization', 'Unknown detector initialization: %s', mode);
end
H = step03_func_bandpass_filter(H, config.sample_rate, config.detector_cutoffs);
stack = [];
for k = 1:size(H, 1)
    [s, ~, frame_times] = spectrogram(H(k, :), config.stft_window, config.stft_overlap, config.stft_nfft, config.sample_rate);
    if isempty(stack),
        stack = zeros(size(s));
    end
    stack = stack + abs(s);
end
if isempty(stack) || any(~isfinite(stack(:))) || ~any(stack(:) > 0),
    return;
end
contour = smoothdata(sum(stack / mean(mean(stack)), 1), 'movmedian', 5);
if any(~isfinite(contour)) || max(contour) <= 0,
    contour = [];
    return;
end
contour = contour / max(contour);
end
