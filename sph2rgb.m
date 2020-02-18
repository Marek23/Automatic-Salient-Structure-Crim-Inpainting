function [I] = sph2rgb(r,theta,fi)

[nx,ny] = size(r); nz = 3;
I = zeros(nx,ny,3);
for x=1:nx
    for y=1:ny
        I(x,y,1) = r(x,y)*cos(theta(x,y))*sin(fi(x,y));
        I(x,y,2) = r(x,y)*sin(theta(x,y))*sin(fi(x,y));
        I(x,y,3) = r(x,y)*cos(fi(x,y));
    end
end


end

