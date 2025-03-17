function steepest_descent_optimization(P, U, u, max_iter, gamma)
% STEEPEST_DESCENT_OPTIMIZATION: Minimizes reprojection error using steepest descent.
% Inputs:
%   P - Cell array containing camera matrices
%   U - 4xN matrix containing 3D points
%   u - Cell array containing image points
%   max_iter - Number of iterations to run
%   gamma - Step size (learning rate)
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

    % Step 2: Compute the update step (steepest descent formula)
    delta_v = -gamma * J' * r;
    
    % Track the norm of delta_v to check if updates are too large
    fprintf('Iteration %d - Norm of delta_v: %.4f\n', k, norm(delta_v));

    % Adaptive step-size search
    gamma = 0.01;  % Start with a conservative value
    max_attempts = 100;  % Prevent infinite loop
    attempt = 0;
    
    while true
        [P_tmp, U_tmp] = update_solution(gamma * delta_v, P, U);
        [new_error, ~] = ComputeReprojectionError(P_tmp, U_tmp, u);
    
        if new_error < error  % Found a better solution
            P = P_tmp;
            U = U_tmp;
            error = new_error;  % Corrected missing semicolon
            break;
        else
            gamma = gamma * 0.5;  % Halve the step size and retry
        end
    
        % Failsafe: Stop after too many attempts
        attempt = attempt + 1;
        if attempt >= max_attempts
            warning('Adaptive step-size search failed to improve error.');
            break;
        end
    end

    % Store error for plotting
    obj_values(k) = sqrt(error / numel(u{1}));

    % Display progress
    fprintf('Iteration %d - RMS Error: %.4f\n', k, sqrt(error / numel(u{1})));
end

% Final RMS Error
[final_error, ~] = ComputeReprojectionError(P, U, u);
fprintf('Final RMS Error: %.4f\n', sqrt(final_error / numel(u{1})));

% Plot the Objective Value for each Iteration
figure;
plot(1:max_iter, obj_values, '-o', 'LineWidth', 2);
title('Steepest Descent Optimization');
xlabel('Iteration Number');
ylabel('Objective Value (Reprojection Error)');
grid on;
end
