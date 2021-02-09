function [pixel_up,pixel_left] = compute_pairwise(img)

% This code for computing pairwise potential on 4 grid
% pixel_up: upper neighbour pixel difference
%pixel_left: left neighbour pixel difference

addpath('Bk_matlab');

%convert image to double
img = double(img);

% Compute the pairwise potentials of the potts model
n1 = sum(((img(2:end,:,:) - img(1:end-1,:,:)).^2),3);
b_wgt = 1./(2*mean(n1(:)));
pixel_up = 45*exp(-b_wgt*n1)+0;

n2 = sum(((img(:,2:end,:) - img(:,1:end-1,:)).^2),3);
b_wgt = 1./(2*mean(n2(:)));
pixel_left = 45*exp(-b_wgt*n2)+0;

end

