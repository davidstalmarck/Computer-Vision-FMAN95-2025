function structure_from_motion(K, x, max_iterations)
% STRUCTURE_FROM_MOTION: Builds an incremental SfM system.

% Step 1: Initialization - Estimate E, Extract Cameras, Triangulate Points
disp('Step 1: Data Loading and Normalization');

% Data Loading
x1 = x{7};
x2 = x{8};

% Check Dimensions
disp(['Size of x1: ', mat2str(size(x1))]);
disp(['Size of x2: ', mat2str(size(x2))]);

% Normalize points
% Ensure the two sets have matched points by trimming to minimum size
min_points = min(size(x1, 2), size(x2, 2));

x1_h = [x1(:, 1:min_points); ones(1, min_points)];
x2_h = [x2(:, 1:min_points); ones(1, min_points)];

x1_n = inv(K) * x1_h;
x2_n = inv(K) * x2_h;


disp('Step 1 Passed - Normalization Complete');

% Step 2: Estimate Essential Matrix with RANSAC
disp('Step 2: Estimating Essential Matrix with RANSAC...');

numIters = 1000;
threshold = 10;
bestE = [];
bestInliers = [];



for i = 1:numIters
    sampleIdx = randperm(min_points, 5);
    E_candidates = fivepoint_solver(x1_n(:, sampleIdx), x2_n(:, sampleIdx));

    for c = 1:length(E_candidates)
        E_cand = E_candidates{c};
        inliers = computeInliers(E_cand, x1_h, x2_h, K, threshold);
        
        % Display Candidate Information
        disp(['Iteration ', num2str(i), ', Candidate ', num2str(c), ...
            ': Number of Inliers = ', num2str(length(inliers))]);

        if length(inliers) > length(bestInliers)
            bestInliers = inliers;
            bestE = E_cand;
        end
    end
end

if isempty(bestE)
    error('Essential Matrix Estimation Failed. Check data or threshold.');
end
disp(['Best E found with ', num2str(length(bestInliers)), ' inliers.']);

% Step 3: Extract Camera Matrices
disp('Step 3: Extracting Camera Matrices...');
[R1, t1] = decomposeE(bestE);
[R2, t2] = decomposeE(bestE);

P1 = K * [eye(3), zeros(3, 1)];
P2 = K * [R1, t1];

disp('Step 3 Passed - Camera Matrices Extracted');

% Step 4: Triangulate Initial 3D Points
disp('Step 4: Triangulating 3D Points...');
inlier_x1 = x1_h(:, bestInliers);
inlier_x2 = x2_h(:, bestInliers);

U = triangulateDLT(P1, P2, inlier_x1, inlier_x2);

% Verify Triangulated 3D Points
disp(['Size of Triangulated Points U: ', mat2str(size(U))]);

% Step 5: Visualize Initial Reconstruction
figure;
scatter3(U(1,:), U(2,:), U(3,:), 10, 'b', 'filled');
axis equal;
grid on;
title('Initial 3D Reconstruction');
xlabel('X (world units)');
ylabel('Y (world units)');
zlabel('Z (world units)');

disp('Step 4 Passed - 3D Points Visualized');

% Step 6: Bundle Adjustment
disp('Step 5: Running Bundle Adjustment...');
levenberg_marquardt_optimization({P1, P2}, U, {inlier_x1, inlier_x2}, max_iterations, 1e-2);

disp('Step 5 Passed - Bundle Adjustment Complete');
disp('Structure-from-Motion Pipeline Completed Successfully!');
end
