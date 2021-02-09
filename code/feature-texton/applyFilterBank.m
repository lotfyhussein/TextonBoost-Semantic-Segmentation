function img_convolved = applyFilterBank(img, fsize, down_sample)
% From the paper
% images are convolved with a 17-dimensional filter-bank at
% scale k.7 The 17D responses for all training pixels
% are then whitened (to give zero mean and unit covariance),
% and an unsupervised clustering is performed

    filter = cell(11, 1);
    sigma = [1,2,4,8];
    [Xdx, Ydy] = meshgrid(-2, 2); % for fsize = 5

    % Gaussian at scales k, 2k, 4k
    for i = 1:3
        filter{i} = fspecial('gaussian', [fsize, fsize], sigma(i));
    end

    % Derivative of Gaussian in x and y at 2k, 4k
    for i = 4:5
       if i == 4
           sigma_val = sigma(2);
       elseif i == 5
           sigma_val = sigma(3);
       end
       filter{i} = fspecial('gaussian', [fsize, fsize], sigma_val);
       filter{i} = filter{i} .* (-2 * Xdx/ (2 * sigma_val^2));
       filter{i} = filter{i} / sqrt(sum(filter{i}.^2, 'all'));

       filter{i+2} = fspecial('gaussian', [fsize, fsize], sigma_val);
       filter{i+2} = filter{i+2} .* (-2 * Ydy/ (2 * sigma_val^2));
       filter{i+2} = filter{i+2} / sqrt(sum(filter{i+4}.^2, 'all')); 
    end

    % Laplacian
    for i=8:11
        filter{i} = fspecial('log', [fsize, fsize], sigma(i-7));
        filter{i} = filter{i} / sqrt(sum(filter{i}.^2, 'all'));
    end

    img_convolved = zeros(size(img, 1), size(img, 2), 17);
    for i=1:3
       img_convolved(:, :, (i-1)*3 + 1) = imfilter(img(:, :, 1), filter{i}, 'same');
       img_convolved(:, :, (i-1)*3 + 2) = imfilter(img(:, :, 2), filter{i}, 'same');
       img_convolved(:, :, (i-1)*3 + 3) = imfilter(img(:, :, 3), filter{i}, 'same');
    end

    for i=10:17
        img_convolved(:, :, i) = imfilter(img(:, :, 1), filter{i-6}, 'same');
    end
    img_convolved = img_convolved(1:down_sample:size(img,1),1:down_sample:size(img,2),:);
    img_convolved = permute(img_convolved, [3,2,1]);
    img_convolved = reshape(img_convolved, 17, size(img_convolved, 2) * size(img_convolved, 3))';
end



