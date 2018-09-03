close all;

depthMax=10;

% rosinit;
% add the ipc_bridge_matlab binaries to your path
image = rossubscriber('/camera/depth/image');
image2 = rossubscriber('/camera/rgb/image_color');


while true
%   tic;  
  sd = receive(image,10);
  
%   if(~isempty(sd))
  c=sd.Data;
%   cFloat=im2double()
  cFloat=typecast(c,'single');
  cFloat=reshape(cFloat,640,480)';
  cRescaled=1-cFloat/depthMax;
%   blur=imfilter(cRescaled,j,'replicate');
  result=wmedian(cRescaled,5,10);
%   j=imnoise(result,'speckle');
%   result2=wmedian(result,5,0.05);
%     subplot(2,1,1);
%     imshow(result);
%     title('Result of Weighted Median Filter','FontSize',14,'FontWeight','Bold');
%     subplot(2,1,2);
%     imshow(cRescaled);
%     title('Kinect: Depth','FontSize',14,'FontWeight','Bold');
     fprintf('Min: %.3f, Max: %.3f\n',min(cFloat(:)),max(cFloat(:)))
%     colormap gray
%     axis image;drawnow;
%   else
%     fprintf('depth is empty\n');
% end
  sd2=receive(image2,10);
  c2=sd2.Data;
  cRGB=permute(flipud(reshape(c2,3,640,480)),[3 2 1]);
%   subplot(121)
  h1=imshow(result);
%   set(h1,'AlphaData',0.5)
%   hold on

%   title('No. of iterations = 5, Weight ratio = 0.05')
%   subplot(122)

%    h2=imshow(cRGB);
%    set(h2,'AlphaData',0.5)
%    hold off
%   toc;
end

