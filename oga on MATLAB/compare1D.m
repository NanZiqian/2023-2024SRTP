
clear
clc

% super parameter
hd = 0.001;%hd = 5e-5;
k=2;%ReLU's power
max_basesize_power = 7;%max base size = 2^max_basesize_power
N=1000;%Gauss Legrendre quadrature discretion number
f = @(z) (1+pi^2)*cos(pi*z);
syms x;
u = cos(pi*x);

% parameter defined by above
b = (-2.0:hd:2.0)';nd = 2*length(b);% number of dictionary
error_ori = zeros(3,max_basesize_power);%at 0, l2 error, H error.
error_dual = zeros(3,max_basesize_power);

%% core code
for error_index = 1:max_basesize_power
    BASE_SIZE = 2^error_index;
    [un_1_ori,err_l2,err_H] = OGA_1D_ori(BASE_SIZE,nd,f,k,N);
    % for ii = 1:BASE_SIZE
    %     if id_ori(ii)>nd/2
    %         g_ori(ii) = RELU(-x + b( mod(id_ori(ii)-1,nd/2)+1 ),k);
    %     else
    %         g_ori(ii) = RELU(x + b(id_ori(ii)),k);
    %     end
    % end
    % un_ori = g_ori*C_ori;
    error_ori(1,error_index) = log10(abs(double(un_1_ori(1)-subs(u,x,0))));
    error_ori(2,error_index) = log10(err_l2(end));
    error_ori(3,error_index) = log10(err_H(end));
    
    [un_1_dual,err_l2,err_H] = OGA_1D_Duality(BASE_SIZE,nd,f,k,N);
    % for ii = 1:BASE_SIZE
    %     if id_dual(ii)>nd/2
    %         g_dual(ii) = RELU(-x + b( mod(id_dual(ii)-1,nd/2)+1 ),k);
    %     else
    %         g_dual(ii) = RELU(x + b(id_dual(ii)),k);
    %     end
    % end
    % un_dual = g_dual*C_dual;
    error_dual(1,error_index) = log10(abs(double(un_1_dual(1)-subs(u,x,0))));
    error_dual(2,error_index) = log10(err_l2(end));
    error_dual(3,error_index) = log10(err_H(end));
end
%% draw
% % error converge in a single OGA
% figure();
% plot(log10((1:BASE_SIZE)'),log10(err),'.r');

%compare error between ori and dual
figure();
scatter(1:max_basesize_power,error_ori(1,1:end),'b');
hold on
scatter(1:max_basesize_power,error_dual(1,1:end),'r');
title('error at 0');
legend('ori error at 0','duality error at 0');
xlabel('BASE SIZE = 2^x');

figure();
scatter(1:max_basesize_power,error_ori(2,1:end),'b');
hold on
scatter(1:max_basesize_power,error_dual(2,1:end),'r');
title('error L2');
legend('ori error l2','duality error l2');
xlabel('BASE SIZE = 2^x');

%% functino RELU
function r=RELU(x,k)
    r=piecewise(x<=0,0,x>0,x^k);
end
