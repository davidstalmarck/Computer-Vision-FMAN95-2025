function bundle_adjustment(P, U, x, max_iterations)
% BUNDLE_ADJUSTMENT: Non-linear refinement of all cameras and 3D points

% Step 1: Filter Out Invalid Data Before Starting
valid_cameras = {};
valid_x = {};
filtered_U = U;

for i = 1:length(P)
    uu = x{i};
    vis = isfinite(uu(1,:));

    % Filter valid points
    valid_vis = find(vis & (vis <= size(U, 2)));

    % Correct out-of-bound entries
    valid_vis(valid_vis > size(filtered_U, 2)) = [];

    % Remove points that are outside valid bounds
    if isempty(valid_vis)
        warning(['Skipping Image ', num2str(i), ' due to insufficient valid points.']);
        continue;  % Skip this image
    end

    % Filter U to retain only valid points
    filtered_U = filtered_U(:, valid_vis);

    % Add valid data for Bundle Adjustment
    valid_cameras{end + 1} = P{i};
    valid_x{end + 1} = uu(:, valid_vis);
end

% Step 2: Ensure Data is Sufficient for Optimization
if isempty(valid_cameras) || isempty(filtered_U)
    warning('No valid data for Bundle Adjustment. Skipping optimization.');
    return;
end

% Step 3: Compute Initial Error
[initial_error, ~] = ComputeReprojectionError(valid_cameras, filtered_U, valid_x);
fprintf('Initial RMS Error: %.4f\n', sqrt(initial_error / numel(valid_x{1})));

% Step 4: Optimization Process
for k = 1:max_iterations
    [r, J] = LinearizeReprojErr(valid_cameras, filtered_U, valid_x);

    lambda = 1e-2;

    % LM Step
    C = J' * J + lambda * speye(size(J, 2));
    c = J' * r;
    delta_v = -C \ c;

    % Update solution
    [valid_cameras, filtered_U] = update_solution(delta_v, valid_cameras, filtered_U);

    % Step 5: Compute New Error
    [new_error, ~] = ComputeReprojectionError(valid_cameras, filtered_U, valid_x);
    fprintf('Iteration %d - RMS Error: %.4f\n', k, sqrt(new_error / numel(valid_x{1})));

    % Adaptive Lambda Control
    if new_error < initial_error
        lambda = lambda * 0.1;  % Reduce lambda if improved
        initial_error = new_error;  % Update best error
    else
        lambda = lambda * 10;   % Increase lambda to stabilize
    end
end

% Step 6: Final Error Report
[final_error, ~] = ComputeReprojectionError(valid_cameras, filtered_U, valid_x);
fprintf('Final RMS Error: %.4f\n', sqrt(final_error / numel(valid_x{1})));
end
