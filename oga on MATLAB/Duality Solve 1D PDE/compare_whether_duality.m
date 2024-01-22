
clear
clc

% super parameter
%hd =1e-2;
hd = 5e-5;
k=2;%ReLU's power
f = @(z) (1+pi^2)*cos(pi*z);
syms x;
u = cos(pi*x);

% parameter defined by above
b = (-2.0:hd:2.0)';nd = 2*(4/hd+1);% number of dictionary
error_at0_ori = zeros(1,3);
error_l2_ori = zeros(1,3);
error_at0_dual = zeros(1,3);
error_l2_dual = zeros(1,3);
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
% [id_ori,C_ori] = OGA_1D_ori(BASE_SIZE,nd,f);
% for ii = 1:BASE_SIZE
%     if id_ori(ii)>nd/2
%         g_ori(ii) = RELU(-x + b( mod(id_ori(ii)-1,nd/2)+1 ));%RELU1
%     else
%         g_ori(ii) = RELU(x + b(id_ori(ii)));
%     end
% end
% un_ori = g_ori*C_ori;
% error_at0_ori(k) = abs(double(subs(un_ori,x,0)-subs(u,x,0)));
% error_l2_ori(k) = sqrt(double(abs(int((un_ori-u)^2,x,0,1))));

[id_dual,C_dual,err] = OGA_1D_Duality(BASE_SIZE,nd,f,k);
% for ii = 1:BASE_SIZE
%     if id_dual(ii)>nd/2
%         g_dual(ii) = RELU(-x + b( mod(id_dual(ii)-1,nd/2)+1 ),k);
%     else
%         g_dual(ii) = RELU(x + b(id_dual(ii)),k);
%     end
% end
% un_dual = g_dual*C_dual;
% error_at0_dual(error_index) = abs(double(subs(un_dual,x,0)-subs(u,x,0)));
% error_l2_dual(error_index) = sqrt(double(abs(int((un_dual-u)^2,x,0,1))));
%% draw
% fplot(un_ori,[0,1],':r');
% hold on
% fplot(u,[0,1],'-b');

% fplot(un_dual,[0,1],':r');
% hold on
% fplot(u,[0,1],'-b');
plot(log10((1:BASE_SIZE)'),log10(err),'.r');

%% functino RELU
function r=RELU(x,k)
    r=piecewise(x<=0,0,x>0,x^k);
end
