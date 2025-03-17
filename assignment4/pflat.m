function [x_flat] = pflat(x)
% x = x(:, 1:3) % for verifying with a smaller example
w = x(end, :); % exstracting w
x_flat = x./w; % element wise division
end