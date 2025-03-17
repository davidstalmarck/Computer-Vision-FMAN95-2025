function [P_new, U_new] = resectioning(U, x_new, K, P1)
% RESECTIONING: Register a new image using 2D-3D correspondences.
% Inputs:
%   U     - Existing 3D points
%   x_new - 2D points in the new image
%   K     - Calibration matrix
%   P1    - Camera matrix for the first image (needed for triangulation)
% Outputs:
%   P_new - New camera matrix
%   U_new - Newly triangulated points

% Identify 2D-3D matches
matched_points = matchPoints(U, x_new);

% Estimate camera pose using PnP
U_inh = U(1:3, matched_points) ./ U(4, matched_points);  
x_inh = x_new(1:2, matched_points) ./ x_new(3, matched_points);

[R_new, t_new, inliers] = pnpRansac(U_inh, x_inh, K);
disp(['New Camera Pose Translation Vector (t_new): ', num2str(t_new')]);
disp(['New Camera Pose Rotation Matrix (R_new): ']);
disp(R_new);

% Check depth consistency (ensure points are in front of camera)
if all(U(3, :) > 0)
    disp('Valid Pose: All points are in front of the camera.');
else
    disp('Warning: Some points may be behind the camera. Possible pose error.');
end

% Compute New Camera Matrix
P_new = K * [R_new, t_new];

% Triangulate new points
U_new = triangulate_new_points(P1, P_new, x_new(:, inliers), x_new(:, inliers));
end
