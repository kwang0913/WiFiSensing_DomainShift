function [output_name, metadata] = step00_feature_filename(filename, dataset)
% Normalize DAT names to user_activity_position[_rNN]_raw.mat.
% Position tokens are numeric, cN, or pN; rNN is a recording identifier.
[~, stem] = fileparts(filename);
parts = regexp(stem, '^([^_]+)_(.+)$', 'tokens', 'once');
if isempty(parts)
    error('wifi:InvalidRecordingName', 'Expected user_activity name: %s', filename);
end
user = lower(parts{1});
rest = parts{2};
recording = '';
% Historical self_time names include a date and literal loc marker.
if strcmp(dataset, 'self_time')
    legacy = regexp(rest, '^[0-9]{4}_(.+)_loc_([^_]+)$', 'tokens', 'once');
    if ~isempty(legacy)
        rest = [legacy{1} '_' legacy{2}];
    end
end
% Historical crossroom repeats use position_1, position_2, etc.
if strcmp(dataset, 'crossroom')
    legacy = regexp(rest, '^(.+_[0-9]+)_([0-9]+)$', 'tokens', 'once');
    if ~isempty(legacy)
        rest = [legacy{1} '_r' sprintf('%02d', str2double(legacy{2}))];
    end
end
repeat = regexp(rest, '^(.*)_(r[0-9]+)$', 'tokens', 'once');
if ~isempty(repeat)
    rest = repeat{1};
    recording = repeat{2};
end
parts = regexp(rest, '^(.+)_((?:p|c)[0-9]+|[0-9]+)$', 'tokens', 'once');
if isempty(parts)
    activity = rest;
    position = 'p1';
else
    activity = parts{1};
    position = parts{2};
end
activity = strrep(activity, '_', '');
switch activity
    case 'document'
        activity = 'doc';
    case {'walking', 'walkingtraj2'}
        activity = 'walk';
    case 'sitting'
        activity = 'sit';
    case 'wiping'
        activity = 'wipe';
end
if isempty(activity)
    error('wifi:InvalidRecordingName', 'Activity is missing: %s', filename);
end
output_stem = [user '_' activity '_' position];
if ~isempty(recording)
    output_stem = [output_stem '_' recording];
end
output_name = [output_stem '_raw.mat'];
metadata = struct('user', user, 'activity', activity, ...
    'position', position, 'recording_id', recording);
end
