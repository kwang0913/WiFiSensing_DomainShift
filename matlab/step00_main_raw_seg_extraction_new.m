% Crossroom: [N,270,4000,3], [N,270,4160,3].
clear;
clc;
project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(project_root, 'matlab'));
% User subfolder; use '' to process all available users.
name = '';
read_from_file = true;
config = step00_processing_profile('crossroom');
% All three feature groups are enabled; STFT is flattened frequency-first.
% See README for explicit legacy-cache compatibility settings.
input_root = fullfile(project_root, 'data', 'crossroom', 'recordings', name);
output_root = fullfile(project_root, 'data', 'crossroom', 'generated_features', name);
step00_run_extraction(input_root, output_root, config, read_from_file);
