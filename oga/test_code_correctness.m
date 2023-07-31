

syms x;
u=cos(pi*x);

%% BASE_SIZE = 2
BASE_SIZE = 2;
k=1;

%% BASE_SIZE = 4
BASE_SIZE = 4;
k=2;

%% BASE_SIZE = 6
BASE_SIZE = 6;
k=3;

%% BASE_SIZE = 8
BASE_SIZE = 8;
k=4;

%% BASE_SIZE = 10
BASE_SIZE = 10;
k=5;

% core code area
%% Duality 
t1=clock;
uk = Duality_approx_simple_PDE(BASE_SIZE);
t2=clock;
time_dual(k)=etime(t2,t1);
errorat0_dual(k)=abs(double(subs(uk,x,0)-subs(u,x,0)));
average_error_dual(k)=double(int(uk-u,x,0,1));
%     fplot(uk,[0,1],':r');
%     hold on
%     fplot(cos(pi*x),[0,1],'-b');

%% original
t1=clock;
uk = Approx_simple_PDE(BASE_SIZE);
t2=clock;
time_ori(k)=etime(t2,t1);
errorat0_ori(k)=abs(double(subs(uk,x,0)-subs(u,x,0)));
average_error_ori(k)=double(int(uk-u,x,0,1));

%相同的BASE_SIZE，验证考虑了duality后运算时间变化微小。（比较两者time）
%相同的BASE_SIZE，考虑了duality的是否在0处误差更小？（比较两者errorat0）
%考虑了duality是否以在其他地方误差更大为代价？（比较两者average_error）

% 结果总结，画图
%% error at 0 comparation
graph_x = 2:2:10;
plot(graph_x,errorat0_dual,'--r',graph_x,errorat0_ori,'-k');
set(gca,'XTick',[0:2:10]);
legend('dual error','ori error');   %右上角标注
xlabel('基的规模');  %x轴坐标描述
ylabel('与0的误差'); %y轴坐标描述

%% 收敛速度比较
plot(errorat0_dual,time_dual,'--r',errorat0_ori,time_ori,'-k');
legend('dual time','ori time');   %右上角标注
xlabel('与0的误差');  %x轴坐标描述
ylabel('运行时间');   %y轴坐标描述
% time_dual=time_dual';
% errorat0_dual=errorat0_dual';
% average_error_dual=average_error_dual';
% time_ori=time_ori';
% errorat0_ori=errorat0_ori';
% average_error_ori=average_error_ori';