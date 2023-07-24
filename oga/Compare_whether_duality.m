

syms x;
u=cos(pi*x);
%% BASE_SIZE = 4
BASE_SIZE = 4;
k=1;

%% BASE_SIZE = 6
BASE_SIZE = 6;
k=2;

%% BASE_SIZE = 10
BASE_SIZE = 10;
k=3;

%% BASE_SIZE = 14
BASE_SIZE = 14;
k=4;

%% core code area
% Duality 
t1=clock;
uk = Duality_approx_simple_PDE(BASE_SIZE);
t2=clock;
time_dual(k)=etime(t2,t1);
errorat0_dual(k)=double(subs(uk,x,0)-subs(u,x,0));
average_error_dual(k)=int(uk-u,x,0,1);

% original
t1=clock;
uk = Approx_simple_PDE(BASE_SIZE);
t2=clock;
time_ori(k)=etime(t2,t1);
errorat0_ori(k)=double(subs(uk,x,0)-subs(u,x,0));
average_error_ori(k)=int(uk-u,x,0,1);

%相同的BASE_SIZE，验证考虑了duality后运算时间变化微小。（比较两者time）
%相同的BASE_SIZE，考虑了duality的是否在0处误差更小？（比较两者errorat0）
%考虑了duality是否以在其他地方误差更大为代价？（比较两者average_error）