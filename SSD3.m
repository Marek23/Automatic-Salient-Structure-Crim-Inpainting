function [qx, qy] = SSD3(I,FI_p,mask_q,patch_size)
%%
min_sum = Inf;
[px, py] = find(mask_q >0);
[diffpx, diffpy] = find(FI_p.m == 1);
for k = 1: length(px)
    x = px(k); y = py(k);
    FI_q = I(x-patch_size:x+patch_size,y-patch_size:y+patch_size,:);
    diff = zeros(size(diffpx));
    for d=1:length(diffpx)
        diff1(d) = FI_p.I(diffpx(d),diffpy(d),1) - FI_q(diffpx(d),diffpy(d),1);
        diff2(d) = FI_p.I(diffpx(d),diffpy(d),2) - FI_q(diffpx(d),diffpy(d),2);
        diff3(d) = FI_p.I(diffpx(d),diffpy(d),3) - FI_q(diffpx(d),diffpy(d),3);
    end
    %%kwadrat róznicy
    diffwyp = sum(diff1.*diff1)+sum(diff2.*diff2)+sum(diff3.*diff3);
    if(diffwyp <= min_sum)
        min_sum = diffwyp;
        qx = x;
        qy = y;
    end
end
end

