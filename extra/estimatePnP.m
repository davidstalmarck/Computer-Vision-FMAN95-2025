function [R_new, t_new] = estimatePnP(U, xi_h, K)
% ESTIMATEPNP: Estimate camera pose using PnP (Perspective-n-Point)
% Inputs:
%   U     - Existing 3D points (4xN in homogeneous coordinates)
%   xi_h  - Corresponding 2D points in homogeneous coordinates (3xN)
%   K     - Camera calibration matrix
% Outputs:
%   R_new - Estimated rotation matrix (3x3)
%   t_new - Estimated translation vector (3x1)

% Convert points to inhomogeneous coordinates
U_inh = U(1:3, :) ./ U(4, :);  % Convert 3D points to Euclidean coordinates
xi_inh = xi_h(1:2, :) ./ xi_h(3, :);  % Convert 2D points to Euclidean coordinates

% Solve PnP using Direct Linear Transform (DLT)
% Construct the matrix A for DLT
n = size(U_inh, 2);
A = zeros(2 * n, 12);

for i = 1:n
    X = U_inh(:, i)';
    u = xi_inh(1, i);
    v = xi_inh(2, i);
    
    A(2 * i - 1, :) = [X, 1, 0, 0, 0, 0, -u * X, -u];
    A(2 * i, :) = [0, 0, 0, 0, X, 1, -v * X, -v];
end

% Solve using SVD
[~, ~, V] = svd(A);
P = reshape(V(:, end), 4, 3)';

% Decompose P into R and t
R = P(:, 1:3);
t = P(:, 4);

% Ensure R is a valid rotation matrix
[U, ~, V] = svd(R);
R_new = U * V';  % Enforce orthogonality (det(R) = 1)

% Ensure positive determinant (correct rotation direction)
if det(R_new) < 0
    R_new = -R_new;
    t = -t;
end

% Normalize t
t_new = t ./ norm(R_new);

end
