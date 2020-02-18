function [fixx,fixy] = ustawWspolrzednePatcha(px,py,patch_size,nx,ny)
fixx =0; fixy =0;
if(px-patch_size < 1)
    fixx = patch_size - px + 1;
end
if(px + patch_size > nx)
    fixx = -(px + patch_size - nx);
end
if(py-patch_size < 1)
    fixy = patch_size - py + 1;
end
if(py + patch_size > ny)
    fixy = -(py + patch_size - ny);
end
end

