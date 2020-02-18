clear all; close all;
%Parametry do Criminisi 2004
alfa = 0.1; scale =0.9;xk =0.97;

%% definicja maski - tam gdzie jest 1 tam inpainting. Wczytywana z pliku
pom = imread('test.png'); [nx, ny, nz] = size(pom);
maska = zeros(nx,ny); maska(pom(:,:,2) == 255 & pom(:,:,1) == 0 & pom(:,:,3) == 0) = 1;
maska_org = maska;
figure; imshow(maska);
title('Maska obrazu')

%% wczytanie obrazu i ustawienie filtru
im = imread('test.png'); im = im2double(im);
%% konwersja obrazu na "spherical coordinate system"
[R, Theta, Fi] = rgb2sph(im); R_max = max(max(R)); R = R/R_max;

figure; imshow(im);
title('Obraz');
%% wstêpnie rozmazuje obraz
sigma = 2; %% parametr filtru gaussa do rozmycia obrazu
R_gauss = imgaussfilt(R, sigma);
figure
imshow(R_gauss);
title('Po filtrowaniu');
%% wstawiam maskê do obrazu
im(:,:,1) = im(:,:,1).*~maska;
im(:,:,2) = im(:,:,2).*~maska;
im(:,:,3) = im(:,:,3).*~maska;
R = R.*~maska; R_gauss = R_gauss.*~maska;
%% wykrycie krawêdzi
%SS = edge(R_gauss,'Canny');
SS = edge(R_gauss,'Canny');
maska_gauss = maska;%poszerzam maskê, ¿eby nie wykrywa³o krawêdzi obiektu, który jest usuwany, poszerzam w zale¿noœci od parametru rozmycia obrazu
for i=1:(sigma/2)-1
    maska_gauss = imdilate(maska_gauss,ones(3,3));
end
SS = SS.*~maska_gauss;
figure
imshow(SS);
title('Salient Structure');
%% punkty przeciecia maski z salient structures
for i=1:sigma
    maska_gauss = imdilate(imdilate(maska_gauss,ones(3,3)), ones(3,3)); %poszerzam maskê, ¿eby z³apaæ linie, które nie do koñca dosz³y do maski
end
SSR = maska_gauss.*SS;
%% dylacja geodezyjna uzyskania salient structures
Salient_Structure_Length = 30; %% parametr d³ugoœci linii
for k = 1:Salient_Structure_Length
    SSR = imdilate(SSR,ones(3,3));
    SSR = min(SSR,SS);
end
%figure
%imshow(SSR);
title('SalientStructure z maska');
%% Segregacja linii
ll = 0;
pkt = 0;
for x = 1: nx
    for y = 1: ny
        %%pierwszy pkt.
        if(ll == 0)
            if(SSR(x,y) == 1)
                ll = ll+1; pkt = pkt+1;
                Lx(pkt) = x; Ly(pkt) = y; Ll(pkt)=ll;
                continue;
            end
        else
            %%algorytm dodawania
            if(SSR(x,y) == 1)
                byl = 0;
                for i = 1: pkt
                    d = (Lx(i)-x)^2 + (Ly(i)-y)^2;
                    if(d < 3)
                        byl = 1;
                        pkt = pkt+1;
                        Lx(pkt) = x; Ly(pkt) = y; Ll(pkt) = Ll(i);
                        break;
                    end
                end
                if (byl == 0)
                    pkt = pkt+1; ll = ll+1;
                    Lx(pkt) = x; Ly(pkt) = y; Ll(pkt) = ll;
                end
            end
        end
    end
end

%% dodatkowe ³¹czenie przerwanych linii, zamykanie niedoskonalosci algorytmu
for i = 1: pkt
    for j = 1: pkt
        d = (Lx(i) - Lx(j))^2 + (Ly(i) - Ly(j))^2;
        if(d < 3)
            for k = 1: pkt
                if(Ll(k) == Ll(i) || Ll(k) == Ll(j))
                    Ll(k) = min(Ll(i),Ll(j));
                end
            end
        end
    end
end

%% usuwanie zbyt krótkich salient structures
minSSL = 15; %minimalna d³ugoœæ Salient Structure
t = tabulate(Ll);
occ = t(:,2);
short = find(occ < minSSL);
for i=1: length(short)
    del = find(Ll == short(i));
    Lx(del) = []; Ly(del) = []; Ll(del) = [];
end
pkt = length(Lx);
clear t occ short
%% poprawa numeracji linii
m = max(Ll);i=1;
was = 0;
while(i<m)
    was = 0;
    if(i > m)
        break
    end
    for j = 1: pkt
        if(was == 0 && Ll(j) == i)
            was = 1;
        end
    end
    if(was == 0)
        for k = 1: pkt
            if(Ll(k) == m)
                Ll(k) = i;
            end
        end
    end
    m = max(Ll);i = i+1;
end
%% Rozbijam SalientStructures na podlinie, ka¿da ma swój obraz
for i=1:max(Ll)
    SSLn(:,:,i) = zeros(nx,ny);
    for k=1:pkt
        if(Ll(k) == i)
            SSLn(Lx(k),Ly(k),i) = 1;
        end
    end
end
%% Rysuje rozbite linie
%for i=1:max(Ll)
%    figure
%    imshow(SSLn(:,:,i));
%    title(i);
%end
%% Wyznaczam œrednie œrodki linii
for i=1:max(Ll)
    f = find(Ll == i);
    Lxs(i) = round(mean(Lx(f))); Lys(i) = round(mean(Ly(f)));
end
%% Wyznaczam wspó³czynniki prostych na podstawie Hough transform
for i=1:max(Ll)
    [Hn, t, r] = hough(SSLn(:,:,i));
    peaks = houghpeaks(Hn,1);
    rho_m(i) = r(peaks(1)); theta_SSLn(i) = t(peaks(2));
end

%% Wyznaczam równanie prostej, tylko do sprawdzenia poprawnoœci
% Ipom = zeros(nx,ny,max(Ll));
% for i=1:max(Ll)
%     for x=1:nx
%         y = floor(-(cos(theta_m(i)*pi/180)/sin(theta_m(i)*pi/180))*x+rho_m(i)/sin(theta_m(i)*pi/180));
%         if(y>0 && y<ny)
%             Ipom(x,y,i) = 1;
%         end
%         y = ceil(-(cos(theta_m(i)*pi/180)/sin(theta_m(i)*pi/180))*x+rho_m(i)/sin(theta_m(i)*pi/180));
%         if(y>0 && y<ny)
%             Ipom(x,y,i) = 1;
%         end
%     end
%     figure
%     imshow(Ipom(:,:,i)'+SP(:,:,i));
% end

%% Wyznaczenie patchy do IOCCDm
Patch_size = 24; % wielkoœæ pola brana pod uwagê przy ³¹czeniu Salient Structure Lines
patchs = zeros(2*Patch_size+1,2*Patch_size+1,3,length(Lxs));
pmasks = zeros(2*Patch_size+1,2*Patch_size+1,3,length(Lxs));
for i=1:length(Lxs)
    [fixx, fixy] = ustawWspolrzednePatcha(Lxs(i),Lys(i),Patch_size,nx,ny);
    patchs(:,:,:,i) = im(Lxs(i)+fixx-Patch_size:Lxs(i)+fixx+Patch_size,Lys(i)+fixy-Patch_size:Lys(i)+fixy+Patch_size,:);
    pmasks(:,:,i) = maska(Lxs(i)+fixx-Patch_size:Lxs(i)+fixx+Patch_size,Lys(i)+fixy-Patch_size:Lys(i)+fixy+Patch_size,:);
end
% rysuje patche, sprawdzam poprawnoœæ wyznaczenia
%for i=1:max(Ll)
%    figure
%    imshow(patchs(:,:,:,i));
%    title(i)
%end
%% Segmentacja kolorów patchy
for i=1:size(patchs,4)
    [CENTS(:,:,i),Iss(:,:,:,i),segmented_patchs(:,:,:,i)] = KmeanSegmentation(4,patchs(:,:,:,i),pmasks(:,:,i));
end

%% rysuje posegmentowane patche, tylko do sprawdzenia poprawnoœci
%for i=1:max(Ll)
%    figure
%    imshow(Iss(:,:,:,i));
%    title(['Patch segmented SSL', num2str(i)]);
%end

%% Wyznaczenie odleg³oœci patchy od siebie metod¹ IOCCD(Improved optimal color composition distance method)
IOCCDs = zeros(length(Lxs),length(Lxs));
THETA_FIT = 20;%parametr dopasowanie k¹tu nachylenia Salient Structures
for i=1:size(patchs,4)
    for j=1:size(patchs,4)
        if( abs(abs(theta_SSLn(i)) - abs(theta_SSLn(j))) < THETA_FIT)
            if(abs(theta_SSLn(i)) > 75)
                if(abs(Lys(i)-Lys(j)) < 25)
                    IOCCDs(i,j) = Inf;
                    continue;
                end
            end
            if(abs(theta_SSLn(i)) < 35)
                if(abs(Lxs(i)-Lxs(j)) < 25)
                    IOCCDs(i,j) = Inf;
                    continue;
                end
            end
            IOCCDs(i,j) = IOCCD(segmented_patchs(:,:,:,i),segmented_patchs(:,:,:,j),CENTS(:,:,i),CENTS(:,:,j));
        else
            IOCCDs(i,j) = Inf;
        end
    end
end
IOCCDs
%% parowanie linii
Paired = zeros(1,2,1);
paired=0;
sparowane = 0;
xlength = size(IOCCDs,1);
ylength = size(IOCCDs,2);
%petla ma tyle ile mo¿e byæ par linii, na razie algorytm bez treshhold'u
for i=1:floor((size(patchs,4))/2)
    val = Inf;
    jestMin = 0;
    for x = 1:xlength
        for y = (x+1):ylength
            %sprawdzam czy x albo y nie jest juz sparowany
            for k=1:size(Paired,3)
                if(Paired(1,1,k) == x || Paired(1,2,k) == x || Paired(1,1,k) == y || Paired(1,2,k) == y)
                    sparowane = 1;
                end
            end
            if(~sparowane && val >= IOCCDs(x,y) && IOCCDs(x,y) ~=Inf)
                val = IOCCDs(x,y);
                minx = x; miny = y;
                jestMin = 1;
            end
            sparowane = 0;
        end
    end
    if(jestMin)
        paired = paired+1;
        Paired(:,:,paired) = [minx,miny];
    end
    jestMin = 0;
end
Paired
%% wyznaczam parametry wielomianu drugiegio stopnia dla po³aczonych linii w celu aproksymacji po³¹czonych g³ównych struktur
for l = 1: paired
    SPLinked(:,:,l) = SSLn(:,:,Paired(1,1,l))+SSLn(:,:,Paired(1,2,l));
    [pktx, pkty] = find(SPLinked(:,:,l) == 1);
    if(abs(theta_SSLn(Paired(1,1,l))) > 20)
        Polynomials(l,:) = polyfit(pkty,pktx,2);
    else
        Polynomials(l,:) = polyfit(pktx,pkty,2);
    end
end

%% Rysuje polaczone linie z wype³nieniem
for i=1:paired
    Interpolacja = zeros(ny,nx);
    if(abs(theta_SSLn(Paired(1,1,l))) > 20)
        for x = 1: ny
            y = ceil(x^2*Polynomials(i,1)+x*Polynomials(i,2) + Polynomials(i,3));
            if( y>=1  && y <= nx)
                Interpolacja(x,y) = 1;
            end
        end
    else
        for y = 1: nx
            x = ceil(y^2*Polynomials(i,1)+y*Polynomials(i,2) + Polynomials(i,3));
            if( x>=1  && x <= ny)
                Interpolacja(x,y) = 1;
            end
        end
    end
    SPLinked(:,:,i) = SPLinked(:,:,i) + (Interpolacja'.*maska_gauss);
    figure
    imshow(SPLinked(:,:,i));
    title(i);
end
%% Poczatek algorytm Criminisi
[nx,ny,nz] = size(im);
mask_fill = double(~single(maska)); %% zmieniam maskê, algorytm tam gdzie ma 0 widzi maskê
%% maska musi byæ o dwa piksele szersza ni¿ maska na zniszczonym obrazie, potrzebne do ró¿niczek
mask = imerode(imerode(mask_fill,ones(3,3)),ones(3,3));
%% inicjalizacja Confidence Term
C = mask_fill;
%% parametr rozmmiaru patch'a
patch_size = 8; %jako promien
%% patch nie mo¿e wychodziæ poza granice obrazu
ograniczenie = ones(nx,ny);
ograniczenie(1:patch_size,:) = 0;ograniczenie(nx-patch_size:nx ,:) = 0;
ograniczenie(:,1:patch_size) = 0;ograniczenie(:,ny-patch_size:ny) = 0;
%% wzorzec mówi sk¹d mogê czerpaæ dane
wzorzec = mask_fill;
for i = 1: 1+patch_size
    wzorzec = imerode(wzorzec,ones(3,3)); %%maska poszerzona, ¿eby nie trafiaæ szukaj¹c wzorca w pole objête mask¹
end
maska = wzorzec;
for k=1:40
    wzorzec = imerode(wzorzec,ones(3,3));
end
wzorzec = (maska - wzorzec).*ograniczenie;
figure
imshow(wzorzec);
title('Punkty do czerpania Fi_q')

%% ograniczenie wyznaczam do drugiej czêœci algorytmu po SalientStructure. Nie chcê korzystaæ w drugiej czêœci z wzorców g³ównych struktur
wzorzec_bez_Salient_Structure = zeros(nx,ny);
%% wype³nianie Salient Structures
for k = 1: paired
    %% poszerzam Salient Structures do czerpania ograniczonego wzorca
    wzorzec_linii = SPLinked(:,:,k).*wzorzec;
    for dil=1:5
        wzorzec_linii = imdilate(wzorzec_linii,ones(3,3));
    end
    wzorzec_linii = wzorzec.*wzorzec_linii;
    wzorzec_bez_Salient_Structure = wzorzec_bez_Salient_Structure + wzorzec_linii;
    %% wype³niam Salient Structure
    maska = SPLinked(:,:,k).*~mask_fill;%maska = zeros(nx,ny);
    while(any(maska(:)))
        [px,py] = find(maska > 0);
        px = px(1);
        py = py(1);
        [fixx, fixy] = ustawWspolrzednePatcha(px,py,patch_size,nx,ny);
        FI_p.I = R(px+fixx-patch_size:px+fixx+patch_size,py+fixy-patch_size:py+fixy+patch_size,:);
        %FI_p.I = im(px+fixx-patch_size:px+fixx+patch_size,py+fixy-patch_size:py+fixy+patch_size,:);
        FI_p.m = mask_fill(px+fixx-patch_size:px+fixx+patch_size,py+fixy-patch_size:py+fixy+patch_size);
        [qx, qy] = SSD(R,FI_p,wzorzec_linii,patch_size);
        %[qx, qy] = SSD3(im,FI_p,wzorzec_linii,patch_size);
        FI_q = im(qx-patch_size:qx+patch_size,qy-patch_size:qy+patch_size,:);
        FR_q = R(qx-patch_size:qx+patch_size,qy-patch_size:qy+patch_size,:);
        
        for x= 1:2*patch_size+1
            for y= 1:2*patch_size+1
                if(mask_fill(px+fixx-1+x-patch_size,py+fixy-1+y-patch_size) == 0)
                    im(px+fixx-1+x-patch_size,py+fixy-1+y-patch_size,:) = FI_q(x,y,:);
                    R(px+fixx-1+x-patch_size,py+fixy-1+y-patch_size,:) = FR_q(x,y,:);
                    C(px+fixx-1+x-patch_size,py+fixy-1+y-patch_size) = scale;
                    mask_fill(px+fixx-1+x-patch_size,py+fixy-1+y-patch_size) = 1;
                end
                maska(px+fixx-1+x-patch_size,py+fixy-1+y-patch_size) = 0;
            end
        end
        imshow(R);
    end
    maska = mask_fill;
end
%% wype³nianie algorytmem Criminisi
%%ca³y czas maska musi byæ o dwa piksele szersza od mask_fill, ¿eby liczyæ ró¿niczki
mask = imerode(imerode(mask_fill,ones(3,3)),ones(3,3));
wzorzec_bez_Salient_Structure = wzorzec_bez_Salient_Structure/max(max(wzorzec_bez_Salient_Structure));
for d=1:patch_size
    wzorzec_bez_Salient_Structure = imdilate(wzorzec_bez_Salient_Structure,ones(3,3));
end
wzorzec = wzorzec.*~wzorzec_bez_Salient_Structure;
while(any(~reshape(mask,[1,nx*ny])))
    dOmega = conv2(~mask,[1,1,1;1,-8,1;1,1,1],'same');
    %wycinam wartoœci granicy dla patchy
    dOmega = dOmega.*ograniczenie;
    [px,py] = find(dOmega > 0);
    %inicjalizacja zmiennych
    dmask.u = px; dmask.v = px; dmask.l = px;
    n_p.u = px; n_p.v = px;
    dR.u = px; dR.v= px;
    D = px;
    for k=1: length(px)
        %%wyznaczenie wektora normalnego od maski
        dmask.u(k) = -(mask(px(k)+1,py(k))-mask(px(k)-1,py(k)))/2;
        dmask.v(k) = -(mask(px(k),py(k)+1)-mask(px(k),py(k)-1))/2;
        dmask.l(k) = sqrt((dmask.u(k))*(dmask.u(k)) + (dmask.v(k))*(dmask.v(k)));
        if(dmask.l(k)>0)
            n_p.u(k) = (dmask.u(k))/dmask.l(k);
            n_p.v(k) = (dmask.v(k))/dmask.l(k);
        else
            n_p.u(k) = 0.1;
            n_p.v(k) = 0.1;
        end
        %%wyznaczenie nowego wektora dla obrazu
        dR.u(k) = (R(px(k),py(k)+1)-R(px(k),py(k)-1))/2;
        dR.v(k) = -(R(px(k)+1,py(k))-R(px(k)-1,py(k)))/2;
        %%obliczanie dataTerm
        D(k) = sqrt(((dR.u(k))*(n_p.u(k)))^2 + ((dR.v(k))*(n_p.v(k)))^2)/alfa;
    end
    Px = px; Py = py; clear px py
    [px,py] = priorytet(Px,Py,C,D,patch_size);
    %wyznaczam wartoœæ fix jeœli patch ma obejmowaæ zakres poza obrazem
    [fixx,fixy] = ustawWspolrzednePatcha(px,py,patch_size,nx,ny);
    px = px+fixx; py = py+fixy;
    FI_p.I = im(px-patch_size:px+patch_size,py-patch_size:py+patch_size,:);
    %FI_p.I = R(px-patch_size:px+patch_size,py-patch_size:py+patch_size,:);
    FI_p.m = mask_fill(px-patch_size:px+patch_size,py-patch_size:py+patch_size);
    [qx, qy] = SSD3(im,FI_p,wzorzec.*ograniczenie,patch_size);
    %[qx, qy] = SSD(R,FI_p,wzorzec.*ograniczenie,patch_size);
    FI_q = im(qx-patch_size:qx+patch_size,qy-patch_size:qy+patch_size,:);
    FR_q = R(qx-patch_size:qx+patch_size,qy-patch_size:qy+patch_size,:);
    for x= 1:2*patch_size+1
        for y= 1:2*patch_size+1
            if(mask_fill(px-1+x-patch_size,py-1+y-patch_size) == 0)
                im(px-1+x-patch_size,py-1+y-patch_size,:) = FI_q(x,y,:);
                R(px-1+x-patch_size,py-1+y-patch_size,:) = FR_q(x,y,:);
                C(px-1+x-patch_size,py-1+y-patch_size) = scale;
                mask_fill(px-1+x-patch_size,py-1+y-patch_size) = 1;
            end
            
            %%ca³y czas maska musi byæ o dwa piksele szersza od mask_fill, ¿eby liczyæ ró¿niczki
            mask = imerode(imerode(mask_fill,ones(3,3)),ones(3,3));
        end
    end
    scale = scale*xk;
    imshow(im);
end