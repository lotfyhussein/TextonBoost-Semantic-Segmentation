function label = alphaExpansion(ini_label, c, conf_map, img_rgb)
% ini_label : Array of length num_pixels for the intial label values
% c : an array of number of labels [0 1] 0 is no label or background and 1 is class label 
% conf_map : A matrix having unary potential values
% img_rgb : Input rgb image.
% labeling : Final labelling


% This code working only on 4 neighbour pixels or CRF grid not Dense Grid 

% Include the matlab wrapper folder for alpha expansion
addpath('Bk_matlab');

img = double(img_rgb);
unary = conf_map';

% Unary size
u_row = size(unary,1);
u_col = size(unary,2);


% Image size 
row = size(img_rgb,1);
col = size(img_rgb,2);

n_variables = row * col;

% Set the expansion move pattern to [classes classe]
expans = [c c];

% Compute the pairwise potentials for upper neighbour pixel difference
% and left pixel difference
[pixel_up , pixel_left]= compute_pairwise(img_rgb);
state = reshape(double(ini_label),row,col);
var_state = 1;

for alpha = expans
    
    % Create the mex object
    obj = BK_Create();

    % Create the nodes. = number of pixels in the image
    % or number of variables in the graph
    BK_AddVars(obj,n_variables);
    
    % Compute the unary potential
    un_out = unary(sub2ind([u_row,u_col],state(:)+1,(1:u_col)'))';
    un_out = [un_out; unary(sub2ind([u_row,u_col],repmat(alpha+1,u_col,1),(1:u_col)'))'];
    % Set the unary potenials using the API
    BK_SetUnary(obj,un_out);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %updating the pairwise potential for bottom and right neighbors
    % for every pixel.
    pairmat = ones(row,col);
    pairmat(end,:) = 0;
    down_indx = find(pairmat>0);
    pairmat00 = abs(state(2:end,:)-state(1:end-1,:))>0;
    pairmat01 = abs(alpha-state(1:end-1,:))>0;
    pairmat10 = abs(state(2:end,:)-alpha)>0;
    prt_1 = [down_indx,down_indx+1,pairmat00(:).*pixel_up(:),pairmat01(:).*pixel_up(:),pairmat10(:).*pixel_up(:),zeros(size(down_indx,1),1)];
    
    pairmat = ones(row,col);
    pairmat(:,end) = 0;
    right_indx = find(pairmat>0);
    pairmat00 = abs(state(:,2:end)-state(:,1:end-1))>0;
    pairmat01 = abs(alpha-state(:,1:end-1))>0;
    pairmat10 = abs(state(:,2:end)-alpha)>0;
    prt_2 = [right_indx,right_indx+row,pairmat00(:).*pixel_left(:),pairmat01(:).*pixel_left(:),pairmat10(:).*pixel_left(:),zeros(size(right_indx,1),1)];
    
    pair_out = [prt_1;prt_2];
    
    BK_SetPairwise(obj,pair_out);
    
    % Run the minimization
    e(var_state) = BK_Minimize(obj);
    
    alphaexp = double(BK_GetLabeling(obj))-1;
    
    % Update the expansion
    state(find(alphaexp==1)) = alpha;
    
    label = uint8(reshape(state,row,col));
    
    % Delete obj at the end
    BK_Delete(obj);
    
    var_state = var_state + 1;
end

