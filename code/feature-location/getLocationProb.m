function [loc_prob] = getLocationProb(VOCopts, cmap, model)

    test_list = VOCopts.testList;
    testing_set_size = VOCopts.numTestList;
    loc_prob = [];
    model = reshape(model, 300*500, 1);
    for i = 1:testing_set_size
        img = imread(sprintf(VOCopts.imgpath, test_list{i}));
        img = imresize(img,[300 500]);
        
        [size_1, size_2, ~] = size(img);
        img_reshaped = single(reshape(img, size_1*size_2, 3));
        eps = 1e-8;
        
        prob = model .* mvnpdf(img_reshaped) + eps;
        loc_prob = vertcat(loc_prob, prob);
    end
end