function [R, t] = decomposeE(E)
% decomposeE - Decomposes the Essential matrix into R and t.
%
% Inputs:
%   E - Essential matrix (3x3)
%
% Outputs:
%   R - Rotation matrix (3x3)
%   t - Translation vector (3x1)

% Step 1: Singular Value Decomposition
[U, ~, V] = svd(E);

% Step 2: Correct determinant (enforce proper rotation matrix)
if det(U) < 0, U = -U; end
if det(V) < 0, V = -V; end

% Step 3: Possible Rotation and Translation Matrices
W = [0 -1 0; 1 0 0; 0 0 1];

% Possible rotation matrices
R1 = U * W * V';
R2 = U * W' * V';

% Possible translation vectors (up to scale)
t1 = U(:, 3);
t2 = -U(:, 3);

% Step 4: Select Valid Rotation and Translation
% Ensure R1 and R2 are valid rotation matrices
if det(R1) < 0, R1 = -R1; end
if det(R2) < 0, R2 = -R2; end

% Return one valid solution — you must test combinations later
R = R1;
t = t1;

% Alternatively, you can return all combinations:
% R_set = {R1, R2};
% t_set = {t1, t2};

end
