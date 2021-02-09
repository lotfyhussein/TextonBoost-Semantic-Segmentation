function [texton_map, pass_var] = getTextonPotential(VOCopts, cmap)
    
    cform = makecform('srgb2lab');
    downsample_for_knn = 2;
    num_k_cluster = 100;
    
    train_list = VOCopts.trainList;
    total_training_images = VOCopts.numTrainList;
    

    img_convolved = [];
    for i = 1:total_training_images
        img_rgb = imread(sprintf(VOCopts.imgpath, train_list{i}));
        img_lab = applycform(img_rgb, cform);
        img_lab = imresize(img_lab,[300 500]);
        img_convolved = [img_convolved; applyFilterBank(img_lab, 5, downsample_for_knn)];
    end

    fprintf('Starting K-means for Train Images...\n');
    opt = statset('Display', 'iter');
    % unsupervised clustering is performed
    % employ the Euclidean-distance K-means clustering
    [C_idx, C] = kmeans(img_convolved, num_k_cluster, 'start', 'cluster', 'MaxIter', 200, 'Options', opt);
    texton_map = zeros(300,500,total_training_images);
    for i = 1:total_training_images
        img_rgb = imread(sprintf(VOCopts.imgpath, train_list{i}));
        img_lab = applycform(img_rgb, cform);
        img_lab = imresize(img_lab,[300 500]);
        img_conv = applyFilterBank(img_lab, 5, 1);   
        % each pixel in each image is assigned to the nearest cluster center,
        % producing the texton map.
        [IDX, D] = knnsearch(C, img_conv);
        texton_map(:, :, i) = reshape(IDX, 500, 300)';
    end
    
    fprintf('K-means finished! \n');
    pass_var.cluster = C;
end