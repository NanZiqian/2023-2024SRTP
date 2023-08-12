
hd = 5e-5; b = (-2.0:hd:2.0)'; 
nd = length(b)*2;%number of dictionary = 160002
f = @(z) (1+pi^2)*cos(pi*z);
syms x;
u = cos(pi*x);

%% 32
BASE_SIZE = 32;
k=1;
%% 64
BASE_SIZE = 64;
k=2;
%% core code
[id_ori,C_ori] = OGA_1D_ori(BASE_SIZE,nd,f);
for ii = 1:BASE_SIZE
    if id_ori(ii)>nd/2
        g_ori(ii) = RELU(-x + b( mod(id_ori(ii)-1,nd/2)+1 ));%RELU1
    else
        g_ori(ii) = RELU(x + b(id_ori(ii)));
    end
end
un_1 = g_ori*C_ori;
error_at0_ori(k) = abs(double(subs(un_1,x,0)-subs(u,x,0)));
error_l2_ori(k) = sqrt(double(abs(int((un_1-u)^2,x,0,1))));

[id_dual,C_dual] = OGA_1D_Duality(BASE_SIZE,nd,f);
for ii = 1:BASE_SIZE
    if id_dual(ii)>nd/2
        g_dual(ii) = RELU(-x + b( mod(id_dual(ii)-1,nd/2)+1 ));
    else
        g_dual(ii) = RELU(x + b(id_dual(ii)));
    end
end
un_1 = g_dual*C_dual;
error_at0_dual(k) = abs(double(subs(un_1,x,0)-subs(u,x,0)));
error_l2_dual(k) = sqrt(double(abs(int((un_1-u)^2,x,0,1))));
%% draw
fplot(un_1,[0,1],':r');
hold on
fplot(u,[0,1],'-b');

%% functino RELU
function r=RELU(x)
    k=1;
    r=piecewise(x<=0,0,x>0,x^k);
end
