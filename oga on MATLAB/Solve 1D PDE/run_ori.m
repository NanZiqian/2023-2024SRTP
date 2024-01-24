
clear
clc

% super parameter
hd = 0.001;
k=2;%ReLU's power
f = @(z) (1+pi^2)*cos(pi*z);
N=1000;%Gauss Legrendre quadrature discretion number
syms x;
u = cos(pi*x);

% parameter defined by above
b = (-2.0:hd:2.0)';nd = 2*length(b);% number of dictionary
%% 32
BASE_SIZE = 32;
error_index=1;
%% 128
BASE_SIZE = 128;
error_index=2;
%% 256
%BASE_SIZE = 256;
%error_index=3;
%% core code
[un_1_ori,err_l2,err_H] = OGA_1D_ori(BASE_SIZE,nd,f,k,N);

%% draw
figure()
plot(log10((1:BASE_SIZE)'),log10(err_l2),'.b');