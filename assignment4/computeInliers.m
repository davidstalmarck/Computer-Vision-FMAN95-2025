function inliers = computeInliers(E, x1_h, x2_h, K, threshold)
% computeInliers - Identifies inliers based on epipolar geometry.
%
% Inputs:
%   E         - Candidate essential matrix (3x3)
%   x1_h       - 3xN matrix of homogeneous points in image 1 (pixel coordinates)
%   x2_h       - 3xN matrix of homogeneous points in image 2 (pixel coordinates)
%   K         - Calibration matrix (3x3)
%   threshold - Distance threshold for inliers (e.g., 5 pixels)
%
% Output:
%   inliers   - Indices of points satisfying the epipolar constraint.

% Step 1: Compute Fundamental Matrix
F = inv(K') * E * inv(K);

% Step 2: Compute epipolar lines
l2 = F * x1_h;  % Lines in image 2 for points in image 1
l1 = F' * x2_h; % Lines in image 1 for points in image 2

% Step 3: Normalize the lines
l2 = l2 ./ sqrt(l2(1,:).^2 + l2(2,:).^2);  % Normalize epipolar lines in image 2
l1 = l1 ./ sqrt(l1(1,:).^2 + l1(2,:).^2);  % Normalize epipolar lines in image 1

% Step 4: Compute distances
d1 = abs(sum(x1_h .* l1, 1));  % Distance in image 1
d2 = abs(sum(x2_h .* l2, 1));  % Distance in image 2

% Step 5: Identify inliers
inliers = find((d1 < threshold) & (d2 < threshold));
end
