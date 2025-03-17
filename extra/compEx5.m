%load('compEx5data.mat');  % Contains calibration matrix K and image points
%whos
%structure_from_motion(K, x, 100);  % Test with 20 iterations for bundle adjustment


%% Trying to use more images to reconstruct
load('compEx5data.mat');
incremental_sfm(K, x, 100);  % Run with 30 iterations for bundle adjustment
%% %% Trying to use more images to reconstruct
%load('compEx5data.mat');
%incremental_sfm_adjacent(K, x);