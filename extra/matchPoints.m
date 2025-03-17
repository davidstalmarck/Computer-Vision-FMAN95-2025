function matched_points = matchPoints(U, xi_h)
% MATCHPOINTS: Matches 3D points to 2D points in new image
% Inputs:
%   U     - Existing 3D points (4xN in homogeneous coordinates)
%   xi_h  - New 2D image points in homogeneous coordinates (3xM)
% Output:
%   matched_points - Indices of matched points in xi_h

% Project 3D points into 2D
projected_points = U(1:2,:) ./ U(3,:);

% Compute Euclidean distances
distances = zeros(size(xi_h, 2), size(projected_points, 2));

for i = 1:size(xi_h, 2)
    dists = sqrt(sum((projected_points - xi_h(1:2, i)).^2, 1));
    [~, matched_points(i)] = min(dists);
end

% Filter matches based on threshold
threshold = 0.01;
valid_matches = vecnorm(projected_points(:, matched_points) - xi_h(1:2, :)) < threshold;

% Return valid matches
matched_points = matched_points(valid_matches);
end
