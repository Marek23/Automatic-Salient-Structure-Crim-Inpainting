function [r, theta, fi] = rgb2sph(I)
[nx,ny,nz] = size(I);

r = zeros(nx,ny);
theta = zeros(nx,ny);
fi = zeros(nx,ny);
for x=1:nx
    for y=1:ny
        r(x,y) = norm([I(x,y,1),I(x,y,2),I(x,y,3)]);
        if(r(x,y) > 0)
            theta(x,y) = atan(I(x,y,2)/I(x,y,1));
            fi(x,y) = acos(I(x,y,3)/r(x,y));
        else
            theta(x,y) = 0;
            fi(x,y) = 0;
        end
    end
end
end

