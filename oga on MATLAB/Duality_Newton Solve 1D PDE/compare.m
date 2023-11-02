
hd = 5e-5; b = (-2.0:hd:2.0)'; 
nd = length(b)*2;%number of dictionary = 160002
f = @(z) (1+pi^2)*cos(pi*z);
g = @(w,b) max(w*qpt+b,0);
syms x;
u = cos(pi*x);


% 32
BASE_SIZE = 32;
k=1;
%% 64
% BASE_SIZE = 64;
% k=2;

%% core code
% [w_ori,b_ori,C_ori] = OGA_1D_ori(BASE_SIZE,nd,f);
% for ii = 1:BASE_SIZE
%     g_ori(ii) = max(w_ori(ii)*x+b_ori(ii),0);
% endc
% un_ori = g_ori*C_ori;
% error_at0_ori(k) = abs(double(subs(un_ori,x,0)-subs(u,x,0)));
% error_l2_ori(k) = sqrt(double(abs(int((un_ori-u)^2,x,0,1))));

[w_dual,b_dual,C_dual] = OGA_Newton_1D_Duality(BASE_SIZE,nd,f);
for ii = 1:BASE_SIZE
    g_dual(ii) = max(w_dual(ii)*x+b_dual(ii),0);
end
un_dual = g_dual*C_dual;
error_at0_dual(k) = abs(double(subs(un_dual,x,0)-subs(u,x,0)));
% error_l2_dual(k) = sqrt(double(abs(int((un_dual-u)^2,x,[0,1]))));

%% draw
% fplot(un_ori,[0,1],':r');
% hold on
% fplot(u,[0,1],'-b');

fplot(un_dual,[0,1],':r');
hold on
fplot(u,[0,1],'-b');
