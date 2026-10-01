function config = step00_processing_profile(dataset)
% Feature representation settings; detector settings are not historical provenance.
config.dataset = char(dataset);
config.sample_rate = 1000;
config.alignment_mode = 'timestamp';
config.time_mapping = 'stft_centers';
config.max_interpolation_gap = 0.010; % Seconds; longer gaps split the recording.
config.rx_count = 3;
config.filter_cutoffs = [0.8 200];
config.detector_cutoffs = [2.5 100];
config.detector_rows = 90;
config.detector_initialization = 'constant'; % Constant prehistory removes DC startup pulse.
config.peak_prominence = 0.01;
config.skip_keywords = {};
config.overwrite = false;
switch config.dataset
    case {'self_time', 'appleman_0713', 'setting_dataset'}
        config.tx_count = 3;
        config.relative_phase = true;
        config.filter_features = true;
        config.segment_duration = 1.2; % TOTAL duration, not a half-window
        config.trim_start_samples = 4000; % Inclusive MATLAB index
        config.trim_end_samples = 8000;
        config.stft_window = 250;
        config.stft_overlap = 125;
        config.stft_nfft = 1000;
        config.stft_bins = 32;
        config.stft_half_frames = 10; % 21 frames; independent of raw duration
        config.stft_layout = 'flattened_groups';
        % Detector starting values, not recovered historical parameters.
        % Retained starting value; historical segment-count fitting is not pursued.
        config.peak_distance = 2.8;
        config.peak_height = 0.1;
        config.skip_keywords = {'keyboard'};
        config.provenance = 'Three-group historical representation; new DAT timestamp alignment is not historical segment replay.';
    case 'crossroom'
        config.tx_count = 3;
        config.relative_phase = true;
        config.filter_features = true; % Requested dynamic-feature bandpass
        config.segment_duration = 4;
        config.trim_start_samples = 100;
        config.trim_end_samples = 4000;
        config.stft_window = 125;
        config.stft_overlap = 63;
        config.stft_nfft = 1000;
        config.stft_bins = 64;
        config.stft_half_frames = 32; % Fixed 65 frames, even across split blocks
        config.stft_layout = 'flattened_groups';
        config.peak_distance = 3;
        config.peak_height = 0.4; % Post-startup-fix crossroom audit
        config.provenance = '3x3 confirmed in raw packet headers; feature filtering enabled. Relative phase and flattened STFT enabled; timestamp alignment splits long gaps.';
    otherwise
        error('wifi:UnknownDataset', 'Unknown dataset: %s', config.dataset);
end
end
