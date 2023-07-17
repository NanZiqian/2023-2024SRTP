## 实验报告

### 主体代码

#### 原问题

```
%% preparation
    clear;
    x=sym("x");
    % -u''+u=f
    f=(1+pi^2)*cos(pi*x);

    N=100;%离散区域数量
    Ck=0;
    w=0;
    b=0;
    k=0;

    %% while
    %while GaussIntegral(abs(u-uk_1),0,1,N) >10^-8 && k < 10^5
    while k < 5
        k=k+1;% 即将产生第k个基

        % g_n=argmax |< g,un_1-u >_H|
        [w(k),b(k)] = argmax_g_product_r(Ck,w,b,f,1);
  
        %w,b包含span(g1,...,gk)的信息
        Ck=projection(f,w,b);
  
        gi(k)=RELU(w(k)*x+b(k),1);
    end

    % 需要C与w,b才能获得uk_1=(g1,...,gk_1)C
    uk=gi*Ck;

    %% 在这之后分别输出un_1,u的图像，进行比较
    fplot(uk,[0,1],':r');
    hold on
    fplot(cos(pi*x),[0,1],'-b');
   
```





#### 对偶问题
```
%% preparation
    clear;
    x=sym("x");
    % -u''+u=f,
    f=(1+pi^2)*cos(pi*x);
    %离散区域数量
    N=100;
    %基的数目，对偶问题中最终基数目为2*BASE_SIZE
    BASE_SIZE=10;

    Ck=0;% un_1的系数列向量
    Cprimek=0;% phin_1的系数列向量
    w=0;
    b=0;
    k=0;

    %% while
    %while GaussIntegral(abs(u-uk_1),0,1,N) >10^-8 && k < 10^5
    while k < BASE_SIZE
        k=k+1;% 即将产生第2k-1,2k个基

        % g_n=argmax |< g,un_1-u >_H|
        [w_g(k),b_g(k)] = argmax_g_product_ru(Ck,w,b,f,1);
        % phi_n=argmax |< g,phin_1-phi >_H|
        [w_h(k),b_h(k)] = argmax_g_product_rphi(Cprimek,w,b,1);


        %w前k个为g的信息，后k个为h的信息；缺点：每一轮都得重写
        w=w_g;% g(i)=RELU(x*w_g(i)+b_g(i),1);
        w(1,k+1:2*k)=w_h;% h(i)=RELU(x*w_h(i)+b_h(i),1);
        b=b_g;
        b(1,k+1:2*k)=b_h;

        %w,b包含span(g1,...,gk,h1,...,hk)的信息
        Ck=projection_u(f,w,b);
        Cprimek=projection_phi(w,b);

    end

    %% 在这之后分别输出un_1,u的图像，进行比较
        % 需要C与w,b才能获得uk=(g1,...,gk,h1,...,hk)Ck
    uk=0;
    for i=1:2*k
        uk=uk+Ck(i)*RELU(w(i)*x+b(i),1);
    end
    
    fplot(uk,[0,1],':r');
    hold on
    fplot(cos(pi*x),[0,1],'-b');


```

### 实例

#### 原问题迭代5次

![](./5.png)

![](./5_time.png)

#### 对偶问题迭代5次

![](./5_dua.png)
相比于原问题，对偶问题运行时间约为原问题2倍。

#### 原问题迭代10次

![](./10.png)

![](./10_time.png)

#### 对偶问题迭代10次

![](./10_dua.png)

![](./10_dua_time.png)



#### 对偶问题迭代15次

![](./15_dua.png)

![](./15_dua_time.png)
运行时间长达20min，但从图像看，精度误差在预期内。
