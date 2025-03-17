function incremental_sfm_adjacent(K, x)
% INCREMENTAL_SFM_ADJACENT: Builds a 3D structure using adjacent image pairs.

% Initialize cumulative 3D point cloud
U_all = [];

% Iterate through all adjacent pairs of images
for i = 1:(length(x) - 1)
    disp(['Processing Image Pair: ', num2str(i), ' & ', num2str(i + 1)]);

    % Load image pairs
    x1 = x{i};
    x2 = x{i + 1};

    % Normalize points
    min_points = min(size(x1, 2), size(x2, 2));
    x1_h = [x1(:, 1:min_points); ones(1, min_points)];
    x2_h = [x2(:, 1:min_points); ones(1, min_points)];

    % Estimate Essential Matrix with RANSAC
    numIters = 1000;
    threshold = 2;
    bestE = [];
    bestInliers = [];

    for j = 1:numIters
        sampleIdx = randperm(min_points, 5);
        E_candidates = fivepoint_solver(x1_h(:, sampleIdx), x2_h(:, sampleIdx));

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

    % Step 2: Extract Initial Camera Matrices
    [R1, t1] = decomposeE(bestE);
    P1 = K * [eye(3), zeros(3, 1)];
    P2 = K * [R1, t1];

    % Step 3: Triangulate Initial 3D Points
    inlier_x1 = x1_h(:, bestInliers);
    inlier_x2 = x2_h(:, bestInliers);

    U = triangulateDLT(P1, P2, inlier_x1, inlier_x2);

    % Combine triangulated points into the cumulative point cloud
    U_all = [U_all, U];
end

% Step 4: Visualization of Combined Structure
figure;
scatter3(U_all(1,:), U_all(2,:), U_all(3,:), 10, 'b', 'filled');
axis equal;
xlabel('X (world units)');
ylabel('Y (world units)');
zlabel('Z (world units)');
title('Combined 3D Reconstruction from Adjacent Image Pairs');
grid on;

disp('Incremental SfM Pipeline Completed Successfully!');
end
