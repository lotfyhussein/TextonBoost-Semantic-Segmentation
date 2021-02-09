function [] = labeling(VOCopts, cmap, confidence_map, model_trained_i, mean_influence)
    
    addpath('feature-texton/Bk_matlab');
    mex -O feature-texton/Bk_matlab/bk_matlab.cpp

    cform = makecform('srgb2lab');
    testing_set_size = VOCopts.numTestList;
    size_1 = 300;
    size_2 = 500;

    test_list = VOCopts.testList;
    for i = 1:testing_set_size
        img = imread(sprintf(VOCopts.imgpath, test_list{i}));
        img = imresize(img,[300 500]);
        img_rgb(:,:,:,i) = img;
        
        [img_GT, img_GT_map] = imread(sprintf(VOCopts.imgGTpath, test_list{i}));
        img_GT_RGB = ind2rgb(img_GT, img_GT_map);
        img_GT_RGB = imresize(img_GT_RGB,[300 500]);
        img_GT_RGB = img_GT_RGB * 255;
        img_GT_all(:, :, :, i) = img_GT_RGB;
    end
    
    acc = [];
    for i = 1:testing_set_size
        fprintf("Labeling test set: %d", i);
        img_segm = img_rgb(:,:,:,i);
        img_blank = zeros(size(img_segm, 1), size(img_segm, 2), 3);
        for class_i = model_trained_i
            confidence_map_i = confidence_map{class_i};
            cm = confidence_map_i(:,:,i) - mean_influence * mean(confidence_map_i(:, :, i), 'all');
            
            conf_map_i = cm>0;
            tmpmat = cm;
            label_map(:,:,i) = alphaExpansion(conf_map_i(:), [0 1], [tmpmat(:), -tmpmat(:)], img_rgb(:,:,:,i));
            
            R = img_segm(:, :, 1);
            G = img_segm(:, :, 2);
            B = img_segm(:, :, 3);

            clr = cmap(class_i, :);
            
            R(find(label_map(:,:,i)==1)) = clr(1) * 255;
            G(find(label_map(:,:,i)==1)) = clr(2) * 255;
            B(find(label_map(:,:,i)==1)) = clr(3) * 255;
            
            img_segm(:, :, 1) = R;
            img_segm(:, :, 2) = G;
            img_segm(:, :, 3) = B;
            
            R = img_blank(:, :, 1);
            G = img_blank(:, :, 2);
            B = img_blank(:, :, 3);
            
            R(find(label_map(:,:,i)==1)) = clr(1);
            G(find(label_map(:,:,i)==1)) = clr(2);
            B(find(label_map(:,:,i)==1)) = clr(3);
            
            img_blank(:, :, 1) = R;
            img_blank(:, :, 2) = G;
            img_blank(:, :, 3) = B;
        end
        
        figure(4), imagesc(cm);
        
        tmp1 = img_GT_all(:, :, :, i);
        figure(3), imshow(tmp1);
        R_same = tmp1(:, :, 1) == img_blank(:, :, 1);
        G_same = tmp1(:, :, 2) == img_blank(:, :, 2);
        B_same = tmp1(:, :, 3) == img_blank(:, :, 3);
        
        tmp = (R_same == G_same);
        tmp = (tmp == B_same);
        acc_i = sum(tmp, 'all') / (size(tmp1, 1) * size(tmp1,2));
        fprintf('       accuracy: %f\n', acc_i);
        acc = [acc acc_i];
        figure(2),imshow(uint8(img_segm));
        figure(1),imshow(img_blank);
        pause
        
    end
    
    mu_acc = mean(acc_i);
    mean(mu_acc, 'all');
    fprintf('Mean Accuracy: %f\n', mu_acc);
end