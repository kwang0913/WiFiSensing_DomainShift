% Three-group historical representation: [N,270,1200,3], [N,270,672,3].
clear;
clc;
project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(project_root, 'matlab'));
% Select: self_time, appleman_0713, or setting_dataset.
dataset = 'self_time';
% Select a recordings subfolder, or '' for all available recordings.
subset = ''; % e.g. '0310_afternoon' for self_time
read_from_file = true;
config = step00_processing_profile(dataset);
% Detector defaults are configurable; historical peak selection is not recovered.
% config.peak_distance = 2.8;
% config.peak_height = 0.1;
input_root = fullfile(project_root, 'data', dataset, 'recordings', subset);
output_root = fullfile(project_root, 'data', dataset, 'generated_features', subset);
step00_run_extraction(input_root, output_root, config, read_from_file);
