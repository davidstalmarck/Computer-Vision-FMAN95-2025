% Load Initial Solution
load('initial_solution.mat');

% Parameters
max_iterations = 8;      % Number of iterations
lambda = 1e-2;            % Start with a small lambda for balanced stability

% Run the optimization
levenberg_marquardt_optimization(P, U, u, max_iterations, lambda);

