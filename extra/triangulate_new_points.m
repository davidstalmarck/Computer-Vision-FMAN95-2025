function U_new = triangulate_new_points(P1, P2, x1, x2)
% TRIANGULATE_NEW_POINTS: Triangulate new 3D points
% Inputs:
%   P1, P2 - Camera matrices
%   x1, x2 - Matched 2D points in two views
% Output:
%   U_new  - Triangulated 3D points

% 1. Triangulate
U_new = triangulateDLT(P1, P2, x1, x2);

% 2. Compute per-point reprojection errors
[~, per_point_res] = ComputeReprojectionError({P1, P2}, U_new, {x1, x2});
% per_point_res should be the same length as the # of points in U_new.

% 3. Filter out points with large reprojection errors
threshold = 5;   % Adjust as needed
valid_points_logical = (per_point_res < threshold);

% Make sure valid_points_logical matches the # of columns in U_new
% If 'per_point_res' is 1×N, then 'valid_points_logical' is also 1×N
if length(valid_points_logical) ~= size(U_new, 2)
    warning('Mismatch in residual vector size vs. U_new columns.');
    valid_points_logical = valid_points_logical(1:size(U_new,2)); 
end

U_new = U_new(:, valid_points_logical);
end
