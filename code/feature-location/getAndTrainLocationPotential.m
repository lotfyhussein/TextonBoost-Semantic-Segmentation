function [model] = getAndTrainLocationPotential(VOCopts, cmap)

% f = fopen('VOC2010/ImageSets/Segmentation/train.txt');
% train_list = textscan(f, '%s');
% train_list = train_list{1,1};
% fclose(f);
train_list = VOCopts.trainList;
n_train_img = size(train_list, 1);

square1 = 300;
square2 = 500;
model = zeros(square1 * square2, VOCopts.nclasses);

for t=1:1:n_train_img
    [img_GT, img_GT_map] = imread(sprintf(VOCopts.imgGTpath, train_list{t}), 'png');
    img_GT_RGB = ind2rgb(img_GT, img_GT_map);
        
    [row, col, dim] = size(img_GT_RGB);
    row_sampled = ceil(linspace(1, row, square1));
    col_sampled = ceil(linspace(1, col, square2));
    indices_sampled = repmat(row_sampled', [1 square2]);
    indices_sampled = indices_sampled + (repmat(col_sampled-1, [square1 1])) * row;
    indices_sampled = indices_sampled(:);
    
    img_GT_reshaped = reshape(img_GT_RGB, [row*col, dim]);
    img_GT_sampled = img_GT_reshaped(indices_sampled, :);
    
    for c=1:VOCopts.nclasses
        model_i = (img_GT_sampled(:, 1) == cmap(c, 1)) & (img_GT_sampled(:, 2) == cmap(c, 2)) & (img_GT_sampled(:, 3) == cmap(c, 3));
        model(model_i, c)= model(model_i, c) + 1;
    end
end

model = model / n_train_img;
model = reshape(model, [square1 square2 VOCopts.nclasses]);

end
