function [CENTS,Is,C] = KmeanSegmentation(K,I,M)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
F = reshape(I,size(I,1)*size(I,2),3);                 % Color Features
pixels = size(F,1);                                           % Cluster Numbers
CCENTS = F( ceil(rand(K,1)*pixels),:);             % Cluster Centers // ceil zaokr�gla do g�ry warto��, razy size zeby losowac tylko z przedzia�u obrazu
% : na ko�cu bo kolory
Mask = reshape(M,size(M,1)*size(M,2),1);             % maska
DAL   = zeros(pixels,K+2);                         % Distances and Labels
KMI   = 10;                                           % K-means Iteration
for n = 1:KMI
    for i = 1:pixels
        if (Mask(i) == 0)%maska to u mnie 1!
            for j = 1:K
                DAL(i,j) = norm(F(i,:) - CCENTS(j,:));         %norm zwraca norm� euklidesowsk� wektora
            end
            [Distance, CN] = min(DAL(i,1:K));               % 1:K are Distance from Cluster Centers 1:K CN to indeks najmniejszej warto��i
            DAL(i,K+1) = CN;                                % K+1 is Cluster Label
            DAL(i,K+2) = Distance;                          % K+2 is Minimum Distance
        else
            DAL(i,K+1) = -1;                                % K+1 is Cluster Label
            DAL(i,K+2) = -1;                                % K+2 is Minimum Distance
        end
    end
    for i = 1:K
        CNS = (DAL(:,K+1) == i);                          % wsp�rz�dne pkt'�w, kt�re maj� w sobie warto�� i-tego clustera
        CCENTS(i,:) = mean(F(CNS,:));                     % New Cluster Centers //wylicza ich sredni�
        if sum(isnan(CCENTS(:))) ~= 0                    % If CENTS(i,:) Is Nan Then Replace It With Random Point
            NC = find(isnan(CCENTS(:,1)) == 1);           % Find Nan Centers
            for Ind = 1:size(NC,1)
                CCENTS(NC(Ind),:) = F(randi(size(F,1)),:);
            end
        end
    end
end
[nx, ny] = size(F);
X = zeros(nx,ny);
for i = 1:K
    idx = find(DAL(:,K+1) == i);
    X(idx,:) = repmat(CCENTS(i,:),size(idx,1),1); %% zamiast p�tli kopiuj� warto�ci macierzy do macierzy X
end
    idx = find(DAL(:,K+1) == -1);
    pom = -ones(1,1,3);
    X(idx,:) = repmat(pom,size(idx,1),1); %% zamiast p�tli kopiuj� warto�ci macierzy do macierzy X
Is = reshape(X,size(I,1),size(I,2),3);%rozpisane kolorami
C = reshape(DAL(:,K+1),size(I,1),size(I,2));%%rozpisane numerami segmentacja
CENTS = CCENTS;
end

