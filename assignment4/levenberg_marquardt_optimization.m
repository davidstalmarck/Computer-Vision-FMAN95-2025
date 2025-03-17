function levenberg_marquardt_optimization(P, U, u, max_iter, lambda)
% LEVENBERG_MARQUARDT_OPTIMIZATION: Minimizes reprojection error using LM method.
% Inputs:
%   P - Cell array containing camera matrices
%   U - 4xN matrix containing 3D points
%   u - Cell array containing image points
%   max_iter - Number of iterations to run
%   lambda - Initial damping factor
%
% Outputs:
%   Final RMS error and plot of objective value vs. iteration count

% Track objective values for plotting
obj_values = zeros(max_iter, 1);

% Compute initial error to assess starting point
[error, ~] = ComputeReprojectionError(P, U, u);
fprintf('Initial RMS Error: %.4f\n', sqrt(error / numel(u{1})));


% Iterative Optimization Process
for k = 1:max_iter
    % Step 1: Compute residuals and Jacobian
    [r, J] = LinearizeReprojErr(P, U, u);

    % Step 2: Compute LM update step
    C = J' * J + lambda * speye(size(J, 2));
    c = J' * r;
    delta_v = -C \ c;

    % Step 3: Update solution
    [P_tmp, U_tmp] = update_solution(delta_v, P, U);

    % Step 4: Compute the new error
    [new_error, ~] = ComputeReprojectionError(P_tmp, U_tmp, u);

    % Step 5: Adaptive Damping Factor Control
    if new_error < error  % Improvement: Keep the step and reduce lambda
        P = P_tmp;
        U = U_tmp;
        error = new_error;
        lambda = lambda * 0.1;  % Reduce lambda for faster convergence
    else  % Step increased error: Increase lambda to stabilize steps
        lambda = lambda * 10;
    end

    % Store error for plotting
    obj_values(k) = sqrt(error / numel(u{1}));

    % Display progress
    fprintf('Iteration %d - RMS Error: %.4f - Lambda: %.4f', k, sqrt(error / numel(u{1})), lambda);
end

% Final RMS Error
[final_error, ~] = ComputeReprojectionError(P, U, u);
fprintf('Final RMS Error: %.4f', sqrt(final_error / numel(u{1})));

% Plot the Objective Value for each Iteration
figure;
plot(1:max_iter, obj_values, '-o', 'LineWidth', 2);
title('Levenberg-Marquardt Optimization');
xlabel('Iteration Number');
ylabel('Objective Value (Reprojection Error)');
grid on;
end
