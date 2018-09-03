% uhat=wmedian(image, Number of iterations, Ratio of weights, Neighborhood size)
% Creates a Neighborhood of specified size (only size 3 or 5) and performs
% the function specified in û i,j= median(Neighbors (k) ∪ Data)
% where Neighbors (k) = {û i',j'} for (i',j') ∈ N i,j and
% û (0) = u as well as (k)
% Data = {u i,j , u i,j ± λ3/λ 2 , u i,j±2(λ3)/λ2· · · , 
%         u i,j ± |N i,j |λ3/λ2 },
function uhat=wmedian(u,n,r,nb)
if nargin<2
    nb=5;
    r=10;
    n=5;
end
if nargin<3
    r=10;
    n=5;
end
if nargin<4
    nb=5;
end
uhat=gpuArray(u);
    deviations=[0 (1:5)*r -(1:5)*r];
for i=1:n
    uStackNeighbors=imageNeighborStack(uhat,nb);


    uStackData=bsxfun(@plus,uhat,shiftdim(deviations,-1));
    uhat=median(cat(3,uStackNeighbors,uStackData),3,'omitNaN');
%     uhat=gather(uhat_GPU);
end