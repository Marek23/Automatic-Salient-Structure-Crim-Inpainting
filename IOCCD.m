function [distance, freq3] = IOCCD(img1CC,img2CC,CENTS1,CENTS2)

%potrzebne do wymiaru macierzy
x     = size(CENTS1,1);
y     = size(CENTS2,1);

img1C = reshape(img1CC,size(img1CC,1)*size(img1CC,2),1);
img2C = reshape(img2CC,size(img2CC,1)*size(img2CC,2),1);

%% czyszcze to gdzie -1
img1Del = img1C == -1;
img2Del = img2C == -1;
img1C(img1Del) = [];
img2C(img2Del) = [];

%% reszta
freq1p = tabulate(img1C); %druga kolumna to iloœæ powtórzeñ 
freq2p = tabulate(img2C);

freq1 = freq1p(:,2)/numel(img1C);  
freq2 = freq2p(:,2)/numel(img2C);
CENTS3 = zeros(1,3);
CENTS4 = zeros(1,3);


DIST  = zeros(x,y);

for i = 1:x % z pierwszego obrazu
    for j = 1:y % z drugiego obrazu
        DIST(i,j) = norm(CENTS1(i,:) - CENTS2(j,:));         %norm zwraca normê euklidesow¹ wektora
    end
end

k=1;
while(size(freq1,2) && size(freq2,2))
    
    val = DIST(1,1);
    for x=1:size(DIST,1)
        for y=1:size(DIST,2)
            if(val >= DIST(x,y))
                val = DIST(x,y);
                minx = x; miny = y;
            end
        end
    end
    if(freq1(minx) < freq2(miny))
        freq3(k) = freq1(minx);
        freq2(miny) = freq2(miny)-freq1(minx);
        CENTS3(k,:) = CENTS1(minx,:);
        CENTS4(k,:) = CENTS2(miny,:);
        
        DIST(minx,:) = [];
        freq1(minx) = [];
        CENTS1(minx,:) = [];
        
    elseif(freq1(minx) > freq2(miny))
        freq3(k) = freq2(miny);
        freq1(minx) = freq1(minx)-freq2(miny);
        CENTS3(k,:) = CENTS1(minx,:);
        CENTS4(k,:) = CENTS2(miny,:);
        
        DIST(:,miny) = [];
        freq2(miny) = [];
        CENTS2(miny,:) = [];
    else
        freq3(k) = freq2(miny);
        CENTS3(k,:) = CENTS1(minx,:);
        CENTS4(k,:) = CENTS2(miny,:);
        
        DIST(minx,:) = [];
        DIST(:,miny) = [];
        freq1(minx) = [];
        freq2(miny) = [];
        CENTS1(minx,:) = [];
        CENTS2(miny,:) = [];
    end
    k = k+1;
end

distance = 0;
for i = 1:k-1
    distance = distance + norm(CENTS4(i)-CENTS3(i))*freq3(i);
end

end


