function class_map = getClassMap(VOCopts, cmap)
    train_list = VOCopts.trainList;
    total_training_images = VOCopts.numTrainList;
    class_map = cell(VOCopts.nclasses, 1);
    fprintf('Starting Classes Map for Train Images...\n');
    for i=1:VOCopts.nclasses
        fprintf(' Class = %d\n', i);
        class_map_i = zeros(300,500, total_training_images);
        for t=1:total_training_images
            [img_GT, img_GT_map] = imread(sprintf(VOCopts.imgGTpath, train_list{t}));
            img_GT_RGB = ind2rgb(img_GT, img_GT_map);
            img_GT_RGB = imresize(img_GT_RGB,[300 500], 'nearest');
            class_map_i(:, :, t) = fitCmap(VOCopts, cmap, img_GT_RGB, i);
        end
        class_map{i} = class_map_i;
    end
    fprintf('Class Map finished! \n');
end