function [color_prob] = getColorPotential(VOCopts, cmap, cm)

    num_cluster = 10;

    test_list = VOCopts.testList;
    testing_set_size = VOCopts.numTestList;
    color_prob = [];
    cform = makecform('srgb2lab');
    for i = 1:testing_set_size
        img = imread(sprintf(VOCopts.imgpath, test_list{i}));
        img = applycform(img, cform);
        img = imresize(img,[300 500]);
        mask = cm(:,:,i)>0;
        
        % color models are represented as Gaussian
        % Mixture Models (GMMs) in CIELab color space
        gmm = colorEM(img, num_cluster, 5, mask); 
        
        [size_1, size_2, ~] = size(img);
        img_reshaped = single(reshape(img, size_1*size_2, 3));
        eps = 1e-8;
        
        prob = zeros(size_1*size_2, num_cluster);
        if gmm.bool
            for k = 1:num_cluster
                tmp = gmm.cov(:,:,k);
                tmp = (tmp + tmp.') / 2;
                prob(:,k) = (gmm.pi(k))  .* mvnpdf(img_reshaped, gmm.mu(k,:), tmp) + eps;
            end
        end
        
        prob = sum(prob, 2);
%         prob = reshape(prob, 300, 500);
        color_prob = vertcat(color_prob, prob);
    end
end