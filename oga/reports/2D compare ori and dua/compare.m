clear;
BASE_SIZE=400;
N=50;
ht=1;
hb=0.1;

[uk1,err1_L2,err1_H1]=OGA_2D_ori(BASE_SIZE,N,ht,hb);
[uk2,err2_L2,err2_H1]=OGA_2D_dua(BASE_SIZE,N,ht,hb);


subplot(1,2,1);
plot(log10((2:2:BASE_SIZE)),log10(err1_L2(2:2:end)),'.r');
hold on;
plot(log10((2:2:BASE_SIZE)),log10(err2_L2(2:2:end)),'.g');
subplot(1,2,2);
plot(log10((2:2:BASE_SIZE)),log10(err1_H1(2:2:end)),'.r');
hold on;
plot(log10((2:2:BASE_SIZE)),log10(err2_H1(2:2:end)),'.g');