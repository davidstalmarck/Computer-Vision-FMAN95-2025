% Load Data
load('compEx2data.mat');  % Contains points and cameras
load('initial_solution.mat');  % Your initial P and U
whos
% Inspect Variables
disp('Checking variable dimensions...');

% Display dimensions of key variables
disp(['Size of U: ', mat2str(size(U))]);
disp(['Size of P{1}: ', mat2str(size(P{1}))]);
disp(['Size of P{2}: ', mat2str(size(P{2}))]);

% Check the content of 'u'
disp('Checking contents of u...');
for i = 1:length(u)
    disp(['Size of u{' num2str(i) '}: ', mat2str(size(u{i}))]);
end

% Parameters
max_iterations = 100;       % Number of iterations
gamma = 1e-6;              % Step size

% Run the optimization
steepest_descent_optimization(P, U, u, max_iterations, gamma);
