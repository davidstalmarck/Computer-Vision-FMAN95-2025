function [R, t] = pnpDLT(U, x, K)
% PNP_DLT: Direct Linear Transform for PnP
% Inputs:
%   U - 3D points in Euclidean coordinates (3xN)
%   x - 2D points in Euclidean coordinates (2xN)
%   K - Calibration matrix
% Outputs:
%   R - Estimated rotation matrix
%   t - Estimated translation vector

% Normalize points
U_h = [U; ones(1, size(U, 2))];
x_h = [x; ones(1, size(x, 2))];

% Build the A matrix
A = [];
for i = 1:size(U, 2)
    X = U_h(:, i)';
    u = x_h(1, i);
    v = x_h(2, i);

    A = [A;
        X, zeros(1, 4), -u*X;
        zeros(1, 4), X, -v*X];
end

% Solve using SVD
[~, ~, V] = svd(A);
P = reshape(V(:, end), 4, 3)';

% Extract R and t
R = P(:, 1:3);
t = P(:, 4);

% Ensure R is a valid rotation matrix
[U, ~, V] = svd(R);
R = U * V';

% Ensure positive determinant (correct rotation direction)
if det(R) < 0
    R = -R;
    t = -t;
end
end
