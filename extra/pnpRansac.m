function [R, t, inliers] = pnpRansac(U, x, K)
% PNP_RANSAC: Estimate camera pose using PnP with RANSAC.
% Inputs:
%   U - 3D points in Euclidean coordinates (3xN)
%   x - 2D points in Euclidean coordinates (2xN)
%   K - Calibration matrix
% Outputs:
%   R - Estimated rotation matrix
%   t - Estimated translation vector
%   inliers - Indices of the inliers

% Parameters
numIters = 1000;
threshold = 3;
bestInliers = [];

% Check if enough points are available
if size(U, 2) < 6
    warning('Insufficient points for PnP (less than 6 matches). Skipping this image.');
    R = eye(3);
    t = zeros(3, 1);
    inliers = [];
    return;
end

% Iterate RANSAC
for i = 1:numIters
    % Randomly select 6 points for PnP (minimum required for PnP)
    sampleIdx = randperm(size(U, 2), 6);

    % Solve PnP using Direct Linear Transform (DLT)
    [R_tmp, t_tmp] = pnpDLT(U(:, sampleIdx), x(:, sampleIdx), K);

    % Reproject points
    P_tmp = K * [R_tmp, t_tmp];
    x_proj = P_tmp * [U; ones(1, size(U, 2))];
    x_proj = x_proj(1:2, :) ./ x_proj(3, :);  % Normalize to 2D

    % Compute errors
    errors = vecnorm(x - x_proj, 2, 1);

    % Identify inliers
    inliers = find(errors < threshold);

    % Keep the best model
    if length(inliers) > length(bestInliers)
        bestInliers = inliers;
        R = R_tmp;
        t = t_tmp;
    end
end

% Return best inliers
inliers = bestInliers;
end
