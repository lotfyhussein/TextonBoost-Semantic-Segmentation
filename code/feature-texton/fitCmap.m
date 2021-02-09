function [class_map] = fitCmap(VOCopts, cmap, img_GT_RGB, c)
    class_map = zeros(size(img_GT_RGB,1),size(img_GT_RGB,2));
    class_map_match = double(...
        (round(img_GT_RGB(:,:,1),5)==round(cmap(c,1),5))&...
        (round(img_GT_RGB(:,:,2),5)==round(cmap(c,2),5))&...
        (round(img_GT_RGB(:,:,3),5)==round(cmap(c,3),5)));
    class_map_match(class_map_match==0) = -1;
    class_map(:,:) = class_map_match;
end

