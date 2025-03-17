function incremental_sfm(K, x, max_iterations)
% INCREMENTAL_SFM - Performs incremental structure-from-motion.

% Step 1: Initialization
P = cell(1, 2);  % Initialize camera poses
U = [];           % Initialize 3D points

% Step 2: Iterate through image pairs
for i = 2:length(x)
    fprintf('\n==================== Reconstruction Step %d ====================\n', i);

    % Extract image points
    x1 = x{i-1};
    x2 = x{i};

    % Step 3: Essential Matrix Estimation
    threshold = 1.0; % Refined RANSAC threshold
    [E, inliers] = estimateEssentialMatrix(x1, x2, K, threshold);
    fprintf('Step %d: Found %d inliers out of %d points.\n', i, length(inliers), size(x1, 2));

    % Step 4: Decompose Essential Matrix
    [R, t] = decomposeE(E);
    fprintf('Step %d: Estimated Rotation Matrix:\n'), disp(R);
    fprintf('Step %d: Estimated Translation Vector:\n'), disp(t');

    % Step 5: Triangulation
    [P1, P2] = camera_poses(K, R, t);
    U = triangulateDLT(P1, P2, x1(:, inliers), x2(:, inliers));

    % Scale Normalization
    if mean(vecnorm(U, 2, 1)) > 1000  
        U = U / 100;  % Normalize large-scale reconstructions
    end

    fprintf('Step %d: Triangulated %d new points.\n', i, size(U, 2));
    disp(['Scale range of reconstructed points: [', ...
        num2str(min(vecnorm(U(1:3, :), 2, 1))), ' to ', ...
        num2str(max(vecnorm(U(1:3, :), 2, 1))), ']']);

    % Step 6: Visualizing Progress
    figure(1); clf;
    scatter3(U(1,:), U(2,:), U(3,:), 10, 'b', 'filled');
    hold on;

    % Display camera centers
    camera_center1 = -R' * t;
    plot3(camera_center1(1), camera_center1(2), camera_center1(3), 'go', 'MarkerSize', 10, 'LineWidth', 2);

    % Final Display
    axis equal;
    title(sprintf('Incremental 3D Reconstruction - Step %d', i));
    xlabel('X (world units)'); ylabel('Y (world units)'); zlabel('Z (world units)');
    grid on;
    drawnow;

    % Step 7: Bundle Adjustment Progress Tracking
    fprintf('Step %d: Running Bundle Adjustment...\n', i);
    bundle_adjustment({P1, P2}, U, {x1, x2}, max_iterations);

    % Compute Final RMS Error
    [error, ~] = ComputeReprojectionError({P1, P2}, U, {x1, x2});
    fprintf('Step %d: Final RMS Error after Bundle Adjustment: %.4f\n', i, sqrt(error / numel(x1)));
end

disp('Reconstruction Complete!');
end
