
hd = 5e-5; b = (-2.0:hd:2.0)'; 
nd = length(b)*2;%number of dictionary = 160002
f = @(x) (1+pi^2)*cos(pi*x);
syms x;
u = cos(pi*x);

%% 32
BASE_SIZE = 32;
k=1;
%% 64
BASE_SIZE = 64;
k=2;
%% core code
[id,C] = OGA_1D_ori(BASE_SIZE,nd,f);
for ii = 1:BASE_SIZE
    if id(ii)>nd/2
        g(ii) = -x + b( mod(id(ii)-1,nd/2)+1 );
    else
        g(ii) = x + b(id(ii));
    end
end
un_1 = g*C;
error_at0_ori(k) = abs(double(subs(un_1,x,0)-subs(u,x,0)));
error_l2_ori(k) = double(abs(int(un_1*u,x,0,1)));

[id,C] = OGA_1D_Duality(BASE_SIZE,nd,f);
for ii = 1:BASE_SIZE
    if id(ii)>nd/2
        g(ii) = -x + b( mod(id(ii)-1,nd/2)+1 );
    else
        g(ii) = x + b(id(ii));
    end
end
un_1 = g*C;
error_at0_dual(k) = abs(double(subs(un_1,x,0)-subs(u,x,0))));
error_l2_dual(k) = double(abs(int(un_1*u,x,0,1)));
