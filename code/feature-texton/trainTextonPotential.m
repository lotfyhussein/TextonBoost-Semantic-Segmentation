function [pass_var] = trainTextonPotential(VOCopts, cmap, texton_map, class_map, pass_var)

    num_boost = 1000;
    downsample_for_training = 5;
    num_k_cluster = 100;
    num_texton_filters = 100;
    
    train_list = VOCopts.trainList;
    total_training_images = VOCopts.numTrainList;
    size_1 = 300;
    size_2 = 500;
    
    pass_var.boost_fidx = zeros(num_boost, VOCopts.nclasses);
    pass_var.boost_tex = zeros(num_boost, VOCopts.nclasses);
    pass_var.boost_theta = zeros(num_boost, VOCopts.nclasses);
    pass_var.boost_a = zeros(num_boost, VOCopts.nclasses);
    pass_var.boost_b = zeros(num_boost, VOCopts.nclasses);
    model_trained_i = pass_var.model_trained_i;
    
    % Random Filter Pos and Size
    FL = round(random('uniform', -30, 30, num_texton_filters, 2));
    FL_size = round(random('uniform', 5, 10, num_texton_filters, 2));
    
    % Get the downsampled indices
    [sampled_Y sampled_X sampled_dim] = meshgrid(...
        1:downsample_for_training:size_2,...
        1:downsample_for_training:size_1,...
        1:total_training_images);
    
    for i=model_trained_i%1:VOCopts.nclasses
        % Iterate for Each Class
        class_map_i = class_map{i};
        cm_sampled = class_map_i(sub2ind([size_1 size_2 total_training_images],...
        sampled_X(:), sampled_Y(:), sampled_dim(:)));
        p = ones(size(sampled_Y,1)*size(sampled_Y,2)*total_training_images,1);

        min_fidx = 0; min_tex = 0; min_theta = 0; min_a = 0; min_b = 0;

        fprintf('Starting Boost of Class %d... \n', i);
        for iter = 1:num_boost
            num_ftr = 100;
            box_pos_rand = round(random('uniform', 1, num_texton_filters, num_ftr, 1));
            texton_pos_rand = round(random('uniform', 1, num_k_cluster, num_ftr, 1));
            min_J = inf;
            if rem(iter, 10) == 0
            fprintf(' Round = %d\n', iter);    
            end
            for ftr_i = 1:num_ftr
                % The texton map is separated into K channels 
                % (one for each texton) and then, for each channel, 
                % a separate integral image is calculated.
                box_i = box_pos_rand(ftr_i);
                texton = texton_pos_rand(ftr_i);
                
                % br, bl, tr and tl denote the bottom right,
                % bottom left, top right and top left corners of
                % box/rectangle
                % Refer to (Figure 10) in paper
                TL_x = min(max(FL(box_i, 1) + sampled_X, 1), size_1) ;
                TL_y = min(max(FL(box_i, 2) + sampled_Y, 1), size_2) ;
                TR_x = min(max(FL(box_i, 1) + sampled_X, 1), size_1);
                TR_y = min(max(FL(box_i, 2) + FL_size(box_i, 2) + sampled_Y, 1), size_2);
                BL_x = min(max(FL(box_i, 1) + FL_size(box_i, 1) + sampled_X, 1), size_1);
                BL_y = min(max(FL(box_i, 2) + sampled_Y, 1), size_2);
                BR_x = min(max(FL(box_i, 1) + FL_size(box_i, 1) + sampled_X, 1), size_1);
                BR_y = min(max(FL(box_i, 2) + FL_size(box_i, 2) + sampled_Y, 1), size_2);
                integral_tm = cumsum(cumsum(texton_map == texton, 1), 2);
                
                % get T^(t) - the integral image of T 
                % for texton channel ftr_i
                TL_int = integral_tm(sub2ind([size_1 size_2, total_training_images],...
                     TL_x(:), TL_y(:), sampled_dim(:)));
                TR_int = integral_tm(sub2ind([size_1 size_2, total_training_images],...
                     TR_x(:), TR_y(:), sampled_dim(:)));
                BL_int = integral_tm(sub2ind([size_1 size_2, total_training_images],...
                     BL_x(:), BL_y(:), sampled_dim(:)));
                BR_int = integral_tm(sub2ind([size_1 size_2, total_training_images],...
                     BR_x(:), BR_y(:), sampled_dim(:)));
                texton_ratio = (BR_int - TR_int - BL_int + TL_int)/...
                    (FL_size(box_i,1)*FL_size(box_i,2));

                for theta = linspace(0, max(texton_ratio), 20)
                    if((sum(texton_ratio>theta)==0))
                        break;
                    end
                    % Closed form solution. Refer to (21), (22) in paper
                    b = sum(p .* cm_sampled .* (texton_ratio <= theta))/...
                        sum(p .* (texton_ratio <= theta));
                    a = sum(p .* cm_sampled .* (texton_ratio>theta))/...
                        sum(p .* (texton_ratio > theta)) - b;
                    % Each weak learner is a decision stump based on
                    % feature response. Refer to (18)
                    h = a * (texton_ratio > theta) + b;
                    % minimizing an error function Jwse incorporating the weights
                    % Please refer to (19) in paper
                    J = sum(p .*((cm_sampled - h).^2));

                    if(J <= min_J)
                        if (theta>0)
                            theta = theta;
                        end
                        min_J = J; min_fidx = box_i; min_tex = texton; 
                        min_theta = theta; min_a = a; min_b = b;
                        % The confidence value H(c, i) can be reinterpreted as a probability
                        % distribution over c using the soft-max or multiclass
                        % logistic transformation to give the texturelayout
                        % potentials. Refer to (20), (17)
                        h1 = exp(-cm_sampled .* h);
                    end
                end
            end
            
            boost_fidx(iter) = min_fidx;
            boost_tex(iter) = min_tex;
            boost_theta(iter) = min_theta;
            boost_a(iter) = min_a;
            boost_b(iter) = min_b;
            % Separable Texture-Layout Filters. Refer to (16)
            p = p .* h1;
        end
        
        pass_var.boost_fidx(:, i) = boost_fidx;
        pass_var.boost_tex(:, i) = boost_tex;
        pass_var.boost_theta(:, i) = boost_theta;
        pass_var.boost_a(:, i) = boost_a;
        pass_var.boost_b(:, i) = boost_b;
    end
    
    fprintf("Finish Boost all class!\n");
    
    pass_var.FL = FL;
    pass_var.FL_size = FL_size;
end