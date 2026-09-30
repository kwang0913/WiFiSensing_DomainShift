function step00_run_extraction(input_root, output_root, config, read_from_file)
% Keep input subdirectories, save provenance, and protect existing outputs.
if ~isfolder(input_root)
    error('wifi:MissingRecordings', 'Raw recordings directory is missing: %s', input_root);
end
files = dir(fullfile(input_root, '**', '*.dat'));
if isempty(files)
    error('wifi:MissingRecordings', 'No DAT recordings found under: %s', input_root);
end
% Resolve all names first so conflicting inputs cannot overwrite each other.
destinations = cell(numel(files), 1);
metadata = cell(numel(files), 1);
for k = 1:numel(files)
    filename = fullfile(files(k).folder, files(k).name);
    [~, stem] = fileparts(filename);
    if any(cellfun(@(word) contains(stem, word), config.skip_keywords))
        continue;
    end
    relative_folder = files(k).folder(numel(input_root) + 1:end);
    if startsWith(relative_folder, filesep)
        relative_folder = relative_folder(2:end);
    end
    [output_name, metadata{k}] = step00_feature_filename(filename, config.dataset);
    destinations{k} = fullfile(output_root, relative_folder, output_name);
end
active = ~cellfun(@isempty, destinations);
normalized = cellfun(@lower, destinations(active), 'UniformOutput', false);
if numel(unique(normalized)) ~= nnz(active)
    error('wifi:DuplicateOutput', ...
        'Multiple inputs map to one output. Give repeated recordings distinct _rNN suffixes.');
end
for k = 1:numel(files)
    if ~active(k)
        continue;
    end
    filename = fullfile(files(k).folder, files(k).name);
    destination = destinations{k};
    if isfile(destination) && ~config.overwrite
        previous = load(destination, 'extraction_info');
        if ~isfield(previous, 'extraction_info') || ~isfield(previous.extraction_info, 'config') || ~isequaln(previous.extraction_info.config, config)
            warning('wifi:ExistingConfigMismatch', 'Existing output has different or missing settings; left untouched: %s', destination);
        end
        fprintf('Skipping existing output: %s\n', destination);
        continue;
    end
    fprintf('Extracting: %s\n', filename);
    [segment_data, segment_data_stft, extraction_info] = ...
        step04_func_segmentation_var_slot_freq(filename, config, read_from_file);
    if isempty(segment_data) || isempty(segment_data_stft)
        warning('wifi:NoValidSegments', 'No valid segments: %s', filename);
        continue;
    end
    extraction_info.source_file = filename;
    extraction_info.labels = metadata{k};
    extraction_info.read_from_file = logical(read_from_file);
    extraction_info.created_at = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    parent = fileparts(destination);
    if ~isfolder(parent),
        mkdir(parent);
    end
    save(destination, 'segment_data', 'segment_data_stft', 'extraction_info');
    fprintf('Saved %d segments: %s\n', size(segment_data, 1), destination);
end
end
