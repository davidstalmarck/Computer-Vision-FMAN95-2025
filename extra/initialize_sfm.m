function [P1, P2, U, inlier_x1, inlier_x2] = initialize_sfm(K, x1, x2)
% INITIALIZE_SFM: Initialize the structure-from-motion system
% Inputs:
%   K  - Calibration matrix
%   x1 - 2D points in the first image
%   x2 - 2D points in the second image
% Outputs:
%   P1, P2 - Camera matrices for the initial pair
%   U       - Initial set of 3D points
%   inlier_x1, inlier_x2 - Inlier points after RANSAC

% Normalize points
min_points = min(size(x1, 2), size(x2, 2));
x1_h = [x1(:, 1:min_points); ones(1, min_points)];
x2_h = [x2(:, 1:min_points); ones(1, min_points)];

x1_n = inv(K) * x1_h;
x2_n = inv(K) * x2_h;

% Estimate Essential Matrix using RANSAC
numIters = 2000;
threshold = 5;
bestE = [];
bestInliers = [];

for i = 1:numIters
    sampleIdx = randperm(min_points, 5);
    E_candidates = fivepoint_solver(x1_n(:, sampleIdx), x2_n(:, sampleIdx));

    for c = 1:length(E_candidates)
        E_cand = E_candidates{c};
        inliers = computeInliers(E_cand, x1_h, x2_h, K, threshold);

        if length(inliers) > length(bestInliers)
            bestInliers = inliers;
            bestE = E_cand;
        end
    end
end

disp(['Best E found with ', num2str(length(bestInliers)), ' inliers.']);

% Extract Camera Matrices
[R1, t1] = decomposeE(bestE);
P1 = K * [eye(3), zeros(3, 1)];
P2 = K * [R1, t1];

% Triangulate Initial 3D Points
inlier_x1 = x1_h(:, bestInliers);
inlier_x2 = x2_h(:, bestInliers);

U = triangulateDLT(P1, P2, inlier_x1, inlier_x2);

disp(['Initialized with ', num2str(size(U, 2)), ' points.']);
end
