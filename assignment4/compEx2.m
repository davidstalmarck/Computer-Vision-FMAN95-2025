load('compEx2data.mat');
% Suppose x{1} and x{2} hold 2D matches in images 1 and 2, respectively.
% K is the camera calibration matrix.
% x1 and x2 are 2×N
x1 = x{1};
x2 = x{2};
N  = size(x1, 2);

% Convert to homogeneous pixel coordinates
x1_h = [x1; ones(1, N)];   % 3×N
x2_h = [x2; ones(1, N)];

% Normalize with K
x1_n = inv(K) * x1_h;      % 3×N normalized coords
x2_n = inv(K) * x2_h;

%%
numIters    = 100;      % or more, depending on outlier ratio
threshold   = 5;         % pixel threshold for inlier test
bestInliers = [];
bestE       = [];

for iter = 1:numIters
    
    % 1) Randomly pick 5 correspondences out of the total N
    sampleIdx = randperm(N, 5);
    
    % 2) Extract the 5 correspondences in normalized coordinates
    x1_5 = x1_n(:, sampleIdx);
    x2_5 = x2_n(:, sampleIdx);
    
    % 3) Solve for E using fivepoint_solver
    E_candidates = fivepoint_solver(x1_5, x2_5);
    
    % 4) For each candidate E, compute inliers
    for c = 1:length(E_candidates)
        E_cand = E_candidates{c};
        
        % 5) Count how many points fit the epipolar constraint
        inliers = computeInliers(E_cand, x1_h, x2_h, K, threshold);
        if length(inliers) > length(bestInliers)
            bestInliers = inliers;
            bestE       = E_cand;
        end
    end
end
% Number of inliers
numInliers = length(bestInliers);
fprintf('Number of inliers: %d\n', numInliers);

% Output:
% bestE = the essential matrix with the largest set of inliers
% bestInliers = index set of the inliers

%%
% Improved Plotting
figure;

% Image 1
subplot(1, 2, 1);
imshow('im1.jpg'); hold on;
scatter(x1_h(1, bestInliers), x1_h(2, bestInliers), 30, 'g', 'filled', 'MarkerFaceAlpha', 0.5); % Inliers
scatter(x1_h(1, setdiff(1:N, bestInliers)), x1_h(2, setdiff(1:N, bestInliers)), 30, 'r', 'filled', 'MarkerFaceAlpha', 0.5); % Outliers
legend('Inliers', 'Outliers');
title('Inliers (green) and Outliers (red) in Image 1');

% Image 2
subplot(1, 2, 2);
imshow('im2.jpg'); hold on;
scatter(x2_h(1, bestInliers), x2_h(2, bestInliers), 30, 'g', 'filled', 'MarkerFaceAlpha', 0.5); % Inliers
scatter(x2_h(1, setdiff(1:N, bestInliers)), x2_h(2, setdiff(1:N, bestInliers)), 30, 'r', 'filled', 'MarkerFaceAlpha', 0.5); % Outliers
legend('Inliers', 'Outliers');
title('Inliers (green) and Outliers (red) in Image 2');


%%
% --- Decompose Essential Matrix ---
[R1, t1] = decomposeE(bestE);  % First solution
[R2, t2] = decomposeE(bestE);  % Second solution (since E can produce 4 solutions)

% Possible camera matrices
P1 = K * [eye(3), zeros(3,1)];      % First camera (identity)
P2_1 = K * [R1, t1];                % First solution
P2_2 = K * [R1, -t1];               % Second solution
P2_3 = K * [R2, t2];                % Third solution
P2_4 = K * [R2, -t2];               % Fourth solution

% --- Triangulate points to find the valid solution ---
% Use your triangulation function and check for positive depth
inlier_x1 = x1_h(:, bestInliers);
inlier_x2 = x2_h(:, bestInliers);

% Triangulate each pair and check valid depth
U_1 = triangulateDLT(P1, P2_1, inlier_x1, inlier_x2);
U_2 = triangulateDLT(P1, P2_2, inlier_x1, inlier_x2);
U_3 = triangulateDLT(P1, P2_3, inlier_x1, inlier_x2);
U_4 = triangulateDLT(P1, P2_4, inlier_x1, inlier_x2);

% Check positive depth condition
validPoints_1 = sum(U_1(3,:) > 0);
validPoints_2 = sum(U_2(3,:) > 0);
validPoints_3 = sum(U_3(3,:) > 0);
validPoints_4 = sum(U_4(3,:) > 0);

% Choose the solution with the most valid points
[~, idx] = max([validPoints_1, validPoints_2, validPoints_3, validPoints_4]);

% Final chosen solution
if idx == 1
    P2 = P2_1; U = U_1;
elseif idx == 2
    P2 = P2_2; U = U_2;
elseif idx == 3
    P2 = P2_3; U = U_3;
else
    P2 = P2_4; U = U_4;
end

fprintf('Chosen Solution: %d\n', idx);
%%

% 3D Reconstruction Plot
figure;
scatter3(U(1,:), U(2,:), U(3,:), 10, 'b', 'filled');
axis equal;
grid on;
title('3D Reconstruction of Scene');
xlabel('X (world units)');
ylabel('Y (world units)');
zlabel('Z (world units)');

%%
% Step 1: Project the 3D points back into both images
x1_proj = P1 * U;
x2_proj = P2 * U;

% Normalize homogeneous coordinates
x1_proj = x1_proj ./ x1_proj(3,:);
x2_proj = x2_proj ./ x2_proj(3,:);

% Step 2: Compute the Euclidean distances (errors)
error1 = sqrt(sum((x1_h(1:2, bestInliers) - x1_proj(1:2, :)).^2, 1));
error2 = sqrt(sum((x2_h(1:2, bestInliers) - x2_proj(1:2, :)).^2, 1));

% Combine errors from both images
allErrors = [error1, error2];

% Step 3: Compute RMS error
RMS_error = sqrt(mean(allErrors .^ 2));

% Step 4: Display RMS error
fprintf('RMS Reprojection Error: %.2f pixels\n', RMS_error);

% Step 5: Plot Histogram
figure;
histogram(allErrors, 30); 
title('Histogram of Reprojection Errors');
xlabel('Reprojection Error (pixels)');
ylabel('Frequency');


% Trim u to only include inlier points
u{1} = x1_h(:, bestInliers);
u{2} = x2_h(:, bestInliers);

% Save Initial Solution for Steepest Descent Optimization
P = {P1, P2};
save('initial_solution.mat', 'P', 'U', 'u');

fprintf('Initial solution saved successfully in initial_solution.mat\n');
