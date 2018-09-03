close all;
figure
plot(1:size(art(200,:),2),art_smooth_final(200,:)); 
hold on;
plot(1:size(art(200,:),2),art(200,:));
legend('smoothed values','ground-truth values');
axis tight;
% imwrite(a,'art_line.png');
figure
plot(1:size(books(200,:),2),books_smooth_final(200,:)); 
hold on;
plot(1:size(books(200,:),2),books(200,:));
legend('smoothed values','ground-truth values');
axis tight;
% imwrite(b,'books_line.png');
figure
plot(1:size(dolls(200,:),2),dolls_smooth_final(200,:)); 
hold on;
plot(1:size(dolls(200,:),2),dolls(200,:));
legend('smoothed values','ground-truth values');
axis tight;
% imwrite(c,'dolls_line.png');
figure
plot(1:size(laundry(200,:),2),laundry_smooth_final(200,:)); 
hold on;
plot(1:size(laundry(200,:),2),laundry(200,:));
legend('smoothed values','ground-truth values');
axis tight;
% imwrite(d,'laundry_line.png');
figure
plot(1:size(moebius(200,:),2),moebius_smooth_final(200,:)); 
hold on;
plot(1:size(moebius(200,:),2),moebius(200,:));
legend('smoothed values','ground-truth values');
axis tight;
% imwrite(e,'moebius_line.png');
figure
plot(1:size(reindeer(200,:),2),reindeer_smooth_final(200,:));
hold on;
plot(1:size(reindeer(200,:),2),reindeer(200,:));
legend('smoothed values','ground-truth values');
axis tight;
% imwrite(f,'reindeer_line.png');
imwrite(art,'art.png');
imwrite(books,'books.png');
imwrite(dolls,'dolls.png');
imwrite(laundry,'laundry.png');
imwrite(moebius,'moebius.png');
imwrite(reindeer,'reindeer.png');