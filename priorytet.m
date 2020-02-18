function [px, py] = priorytet(px,py,C,D,patch_size)
value = px;
for k = 1: length(px)
    pom = 0;
    for i=(px(k)-patch_size):(px(k)+patch_size)
        for j=(py(k)-patch_size):(py(k)+patch_size)
            pom = pom+C(i,j);
        end
    end
    pom = pom /(2*patch_size+1)*(2*patch_size+1);
    value(k) = D(k)*pom;
end
[a,k] = max(value);
px = px(k); py = py(k);
end

