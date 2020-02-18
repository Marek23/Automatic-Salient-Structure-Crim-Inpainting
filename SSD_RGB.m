function [qx, qy] = SSD_RGB(I,FI_p,mask_q,patch_size)
 min_sum = Inf;

[nx, ny] = size(mask_q);
FI_q = zeros(2*patch_size+1,2*patch_size+1,3);
for x=1+patch_size: nx-patch_size-1
    for y=1+patch_size: ny-patch_size-1
        
        %%wybieram fi_q ze wzrorca mask_q
        if mask_q(x,y) > 0
            FI_q = I(x-patch_size:x+patch_size,y-patch_size:y+patch_size,:);
        end
        
        %%liczê ró¿nicê tylko pkt fi_p, które s¹ w obrazie do dopasowania
        diff = 10*ones(2*patch_size+1,2*patch_size+1);
        for i=1:2*patch_size+1
            for j=1:2*patch_size+1
                if (FI_p.m(i,j) == 1)
                    %odleg³oœæ Euklidesowa
                    diff(i,j) = sqrt((FI_p.I(i,j,1) - FI_q(i,j,1))^2 + (FI_p.I(i,j,2) - FI_q(i,j,2))^2 + (FI_p.I(i,j,3) - FI_q(i,j,3))^2);
                end
            end
        end
        %%kwadrat róznicy
        diff = sum(sum(diff));
        if(diff <= min_sum)
            min_sum = diff;
            qx = x;
            qy = y;
        end
    end
end

end
