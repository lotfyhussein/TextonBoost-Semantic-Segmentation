function [confidence_map] = classifyTexton(VOCopts, cmap, pass_var)

    FL = pass_var.FL;
    FL_size = pass_var.FL_size;
    boost_fidx = pass_var.boost_fidx;
    boost_tex = pass_var.boost_tex;
    boost_theta = pass_var.boost_theta;
    boost_a = pass_var.boost_a;
    boost_b = pass_var.boost_b;
    model_trained_i = pass_var.model_trained_i;
    C = pass_var.cluster;
    
    trained_loc = getAndTrainLocationPotential(VOCopts, cmap);

    cform = makecform('srgb2lab');
    testing_set_size = VOCopts.numTestList;
    size_1 = 300;
    size_2 = 500;

    test_list = VOCopts.testList;
    
    % Get Texton Map of the test images
    texton_map = zeros(300, 500, testing_set_size);
    for i = 1:testing_set_size
        img = imread(sprintf(VOCopts.imgpath, test_list{i}));
        img = imresize(img,[300 500]);
        img_rgb(:,:,:,i) = img;
        img = applycform(img_rgb(:,:,:,i), cform);
        img = imresize(img,[300 500]);
        img_conv = applyFilterBank(img, 5, 1);   
        [IDX, D] = knnsearch(C, img_conv);
        texton_map(:,:,i) = reshape(IDX, 500, 300)';
    end

    confidence_map = cell(VOCopts.nclasses, 1);
    [Ycod Xcod Zcod] = meshgrid(1:1:size_2,1:1:size_1,1:testing_set_size);
    CONF = 0;
    for i=model_trained_i%1:VOCopts.nclasses
        fprintf('Starting Classify of Class %d... \n', i);
        boost_idx_i = boost_fidx(:, i);
        boost_tex_i = boost_tex(:, i);
        boost_theta_i = boost_theta(:, i);
        boost_a_i = boost_a(:, i);
        boost_b_i = boost_b(:, i);
        trained_loc_i = trained_loc(:, :, i);
        for iter = 1:length(boost_theta_i)
            % The texton map is separated into K channels 
            % (one for each texton) and then, for each channel, 
            % a separate integral image is calculated.
            if rem(iter, 10) == 0
            fprintf(' Round = %d\n', iter);    
            end
            box_pos = boost_idx_i(iter);
            texton_pos = boost_tex_i(iter);
            theta = boost_theta_i(iter);
            a = boost_a_i(iter);
            b = boost_b_i(iter);
            
            % br, bl, tr and tl denote the bottom right,
            % bottom left, top right and top left corners of
            % box/rectangle
            % Refer to (Figure 10) in paper
            TL_x = min(max(FL(box_pos,1) + Xcod,1), size_1) ;
            TL_y = min(max(FL(box_pos,2) + Ycod,1), size_2) ;
            TR_x = min(max(FL(box_pos,1) + Xcod,1), size_1);
            TR_y = min(max(FL(box_pos,2) + FL_size(box_pos, 2) + Ycod, 1), size_2);
            BL_x = min(max(FL(box_pos,1) + FL_size(box_pos, 1) + Xcod, 1), size_1);
            BL_y = min(max(FL(box_pos,2) + Ycod, 1), size_2);
            BR_x = min(max(FL(box_pos,1) + FL_size(box_pos, 1) + Xcod, 1), size_1);
            BR_y = min(max(FL(box_pos,2) + FL_size(box_pos, 2) + Ycod, 1), size_2);
            integral_tm = cumsum(cumsum(texton_map == texton_pos,1),2);
            
            % get T^(t) - the integral image of T 
            % for texton channel ftr_i
            TL_int = integral_tm(sub2ind([size_1 size_2, testing_set_size], TL_x(:), TL_y(:), Zcod(:)));
            TR_int = integral_tm(sub2ind([size_1 size_2, testing_set_size], TR_x(:), TR_y(:), Zcod(:)));
            BL_int = integral_tm(sub2ind([size_1 size_2, testing_set_size], BL_x(:), BL_y(:), Zcod(:)));
            BR_int = integral_tm(sub2ind([size_1 size_2, testing_set_size], BR_x(:), BR_y(:), Zcod(:)));
            texton_ratio = (BR_int - TR_int - BL_int + TL_int)/...
                (FL_size(box_pos,1)*FL_size(box_pos,2));
            % Get confidence map
            if iter > 1
                color_prob = getColorPotential(VOCopts, cmap, confidence_map_i);
                loc_prob = getLocationProb(VOCopts, cmap, trained_loc_i);
                CONF = CONF + a*(texton_ratio>theta) + b + color_prob + loc_prob; %
            else
                CONF = CONF + a*(texton_ratio>theta) + b;
            end
            confidence_map_i = reshape(CONF,size_1,size_2,testing_set_size);
        end

        confidence_map_i = reshape(CONF,size_1,size_2,testing_set_size);
        confidence_map{i} = confidence_map_i;
    end
    fprintf("Finish Classifying of all class!\n");
end