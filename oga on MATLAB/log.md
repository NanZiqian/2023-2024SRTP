# 研究进展

### 2024.01.21

16:04

我们发现将b的步长hd控制在1e-2时，误差反弹现象消失了；而b的步长hd在5e-4时，误差反弹现象明显，虽然高ReLU power时可以延缓反弹、增加精度，但前者进行500多次迭代都不会反弹。

我们初步预测，这是因为迭代次数高时，argmax会重复选择字典中的神经网络基函数，使得Pn中线性系统A奇异，算法

```
C_g = lsqminnorm(A(1:2*i,1:2*i),rhs_g(1:2*i));
```

将会按照一定规则给出C_g，即un_1=神经网络基函数线性组合的系数。

并且该规则在**字典小**时，给出的系数可以让高次迭代增加的神经网络基函数线性组合相互抵消；在**字典大**时，给出的系数无法让新选择的抵消，从而造成误差反弹。

但是为何反直觉地**只在字典小时给出的系数可以相互抵消，误差不会反弹**，仍需探究；为何说反直觉，因为b字典连续时，理应argmax更不容易取到相同的基，A更不容易奇异，因此**b字典连续时，argmax是否会取到相同的基**也是个关键问题。

未来要解决的问题：

- 在选到重复基时修改矩阵A，避免奇异性；
- 探究方法lsqminnorm的底层原理（小字典时给出的C是什么？），或者更换求解线性系统方法

### 2024.01.22

17:07

因为在有限函数空间里找u的正交最佳逼近，扩大函数空间后找不可能比小函数空间找到的|| u-un_1 || 更大，顶多给新的基0系数，现在误差变得更大，最可能的来源是求解线性方程的误差。

因此我们推测，hb大一点，就是限定住两个g最接近的情况不能超过这个hb，防止argmax两次迭代选中相近g，令A奇异。而小步长误差反弹就是因为argmax两次迭代选中了相近g，求解A*C=rhs误差极大。

证据如1D_ori_data.xlsx，**小步长的系数C拥有上千的元素，而大步长的都在0.01左右**。

未来要解决的问题：

- 小步长到底有没有选择两个接近的g，如何定义“接近”
- 到底是什么让A奇异了
- 为什么A奇异会解出来上千的系数C

# 2024.01.24

11:55

我们发现一切都是因为Gauss积分精度不够，在离散点N取到600以上时，就完美地收敛了。

我们需要回到对比原问题和对偶问题收敛速度的方向上来，