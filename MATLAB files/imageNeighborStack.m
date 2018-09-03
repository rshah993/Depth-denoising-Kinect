function uStack=imageNeighborStack(u,nb)
if nb==3
    neighbors=[-1 -1; -1 0; -1 1; 0 -1; 0 0; 0 1; 1 -1; 1 0; 1 1;];

else
    neighbors=[-2 -2; -2 -1 ; -2 0; -2 1; -2 2; -1 -2; -1 -1; -1 0; -1 1; -1 2; 0 -2; 0 -1; 0 0; 0 1; 0 2; 1 -2; 1 -1; 1 0; 1 1; 1 2; 2 -2; 2 -1; 2 0; 2 1; 2 2;];
end

padDim=max(abs(neighbors))+1;
sizeU=size(u);

uPad=gpuArray(padarray(u,padDim,NaN));

NNeighbors=size(neighbors,1);
uStack=gpuArray(zeros([size(u) NNeighbors]));
for iNeighbor=1:NNeighbors
    xShift=gpuArray(neighbors(iNeighbor,1));
    yShift=gpuArray(neighbors(iNeighbor,2));
    uStack(:,:,iNeighbor)=...
        uPad(padDim+1+xShift:padDim+sizeU(1)+xShift,...
        padDim+1+yShift:padDim+sizeU(2)+yShift);
end
