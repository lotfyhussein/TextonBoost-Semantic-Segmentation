function [gmm] = colorEM(img, num_cluster, num_iter, mask)

    [row, col] = size(mask);
    mask_i = reshape(mask, row*col, 1);
    mask_i = find(mask_i);

    gmm.bool = 0;
    gmm.mu = 0;
    gmm_cov = zeros(3, 3, num_cluster);
    gmm.pi = 0;

    if size(mask_i) > 3

        gmm.bool = 1;
        [row, col, ~] = size(img);
        img_reshaped = single(reshape(img, row*col, 3));
        img_mask_i = img_reshaped(mask_i, :);
        size_1 = size(img_mask_i,1);
        size_2 = size(img_mask_i,2);

        res = zeros(size_1,num_cluster);
        eps = 1e-8;
        % First, the color clusters (4) are learned in an unsupervised
        % manner using K-means.
        [idx, C] = kmeans(img_mask_i, num_cluster, 'Distance', 'cityblock', 'Replicates', 5);
        gmm_mean = C;
        gmm_cov = zeros(3,3,num_cluster);
        gmm_pi = ones(1,num_cluster)/num_cluster;
        for k = 1:num_cluster
            gmm_cov(:,:,k) = single(cov(img_mask_i((idx(:) == k), :)));
            gmm_pi(k) = sum((idx(:) == k)) / numel(idx);
        end

        % An iterative algorithm,
        % reminiscent of EM [12], then alternates between
        % inferring class labeling c*
        for l = 1:num_iter
            for k = 1:num_cluster
                tmp = gmm_cov(:,:,k);
                tmp2 = (tmp + tmp.') / 2;
                p = mvnpdf(img_mask_i, gmm_mean(k,:), tmp2);
                p = gmm_pi(k) * p ;
                res(:,k) = p ;
            end

            for i = 1:size_1
                res(i,:) = res(i,:) / (sum(res(i,:)) + eps );
            end

            gmm_pi = mean(res,1);
            for k = 1:num_cluster
                gmm_mean(k,:) = sum(img_mask_i .* repmat(res(:,k), 1, size_2), 1) ./ (sum(res(:,k)) + eps );
            end

            for k = 1:num_cluster
                img_mask_centered = img_mask_i - repmat(gmm_mean(k,:), size_1, 1);
                tmp = (img_mask_centered .* repmat(res(:,k), 1, size_2))' * img_mask_centered;
                gmm_cov(:,:,k) = tmp ./ ( sum(res(:,k)) + eps );
                gmm_cov(:,:,k) = gmm_cov(:,:,k) + eps;
            end
        end

        gmm.mu = gmm_mean;
        gmm.cov = gmm_cov;
        gmm.pi = gmm_pi;
    end

end