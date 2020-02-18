function [qx, qy] = SSD(I,FI_p,mask_q,patch_size)
%%
min_sum = Inf;
[px, py] = find(mask_q >0);
[diffpx, diffpy] = find(FI_p.m == 1);
for k = 1: length(px)
    x = px(k); y = py(k);
    FI_q = I(x-patch_size:x+patch_size,y-patch_size:y+patch_size);
    diff = zeros(size(diffpx));
    for d=1:length(diffpx)
        diff(d) = FI_p.I(diffpx(d),diffpy(d)) - FI_q(diffpx(d),diffpy(d));
    end
    %%kwadrat róznicy
    diff2 = sum(diff.*diff);
    if(diff2 <= min_sum)
        min_sum = diff2;
        qx = x;
        qy = y;
    end
end
end

