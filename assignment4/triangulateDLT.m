function U = triangulateDLT(P1, P2, x1, x2)
% triangulateDLT - Triangulates 3D points using DLT.
%
% Inputs:
%   P1 - 3x4 Camera matrix for image 1
%   P2 - 3x4 Camera matrix for image 2
%   x1 - 3xN Homogeneous points in image 1 (pixel coordinates)
%   x2 - 3xN Homogeneous points in image 2 (pixel coordinates)
%
% Output:
%   U  - 4xN Triangulated 3D points in homogeneous coordinates

% Step 1: Initialize 3D points
N = size(x1, 2);      % Number of points
U = zeros(4, N);      % Allocate space for 3D points

% Step 2: DLT Triangulation for each point
for i = 1:N
    % Build matrix A for Ax = 0
    A = [
        x1(1,i) * P1(3,:) - P1(1,:);
        x1(2,i) * P1(3,:) - P1(2,:);
        x2(1,i) * P2(3,:) - P2(1,:);
        x2(2,i) * P2(3,:) - P2(2,:)
    ];

    % Step 3: Solve using SVD
    [~, ~, V] = svd(A);
    X = V(:, end);   % Solution is the last column of V
    U(:, i) = X / X(4);  % Normalize to homogeneous coordinates
end

end
