---
title: 神经网络与深度学习
layout: post
img: nn.jpg
tags: [神经网络, 机器学习]
---

* TOC
{:toc}

## 1. 使用神经网络识别手写数字

### 1.1 感知器

一个感知器接受几个二进制输入，$x_1,x_2,...$ ，并产生一个二进制输出：

<p style="text-align:center"><img src="/assets/img/post/感知器.JPG"></p>

有一个简单的规则来计算输出：引入**权重**$\omega_1,\omega_2,...$ ，表示相应输入对于输出重要性的实数；神经元的输出由分配权重后的总和$\sum_{j} \omega_j x_j$小于或者大于一些**阈值**决定。规则的代数形式为：
$$
output=\left\{\begin{matrix}
0 & if \sum_{j} \omega_j x_j \leq threshold\\ 
1 & if \sum_{j} \omega_j x_j > threshold
\end{matrix}\right.\tag{1-1}
$$
令$\omega\equiv (\omega_1,\omega_2,...,\omega_j),x\equiv(x_1,x_2,...x_j)^T$，偏置$b\equiv-threshold$ ，那么感知器的规则可以重写为：
$$
output=\left\{\begin{matrix}
0 & if  \; \omega \cdot x + b \leq 0\\ 
1 & if  \; \omega \cdot x + b > 0
\end{matrix}\right.\tag{1-2}
$$

### 1.2 S型神经元

S型神经元和感知器类似，但是被修改为权重和偏置的微小改变只引起输出的微小变化。这对于神经元网络的学习是很关键的。

<p style="text-align:center"><img src="/assets/img/post/感知器.JPG"></p>

其中：
$$
output=\left\{\begin{matrix}
0 & if  \; \sigma(\omega \cdot x + b) \leq 0\\ 
1 & if  \; \sigma(\omega \cdot x + b) > 0
\end{matrix}\right.\tag{1-3}
$$

$$
\sigma (z)\equiv \frac{1}{1+e^{-z}}\tag{1-4}
$$

<p style="text-align:center"><img src="/assets/img/post/S函数.JPG"></p>

如果$z=\omega\cdot x+b\gg0$，那么$e^{-z}\approx 0,\sigma(z)\approx1$。

如果$z=\omega\cdot x+b\ll0$，那么$e^{-z}\rightarrow\infty,\sigma(z)\approx 0$。

### 1.3 神经网络的架构

<p style="text-align:center"><img src="/assets/img/post/神经网络.JPG"></p>

神经网络可以由一个输入层、一个输出层和任意（可以是$0$）个隐藏层组成，而一般的BP神经网络有一到两个隐藏层。

### 1.4 一个简单的分类手写数字的网络

我们将使用一个三层神经网络来识别单个数字：

<p style="text-align:center"><img src="/assets/img/post/分类网络.JPG"></p>

网络的输入层包含给输入像素的值进行编码的神经元。我们给网络的训练数据会有很多扫描得到的$28\times 28$的手写数字的图像，所有输入层包含有$784=28\times 28$个神经元。输入像素是灰度级的，值为$0.0$表示白色，值为$1.0$表示黑色，中间数值表示逐渐暗淡的灰色。

示例使用了一个小的隐藏层，仅包含$15$个神经元。

输出层包含$10$个神经元，如果第一个神经元激活，即输出$\approx1$，那么表明网络认为数字是一个$0$，以此类推。更确切地说，我们把输出神经元的输出赋予编号$0$到$9$，并计算哪个神经元有最高的激活值。激活值最高的神经元的编号即为网络猜测的数字。

### 1.5 使用梯度下降算法进行学习

我们将使用 *MNIST* 数据集作为训练数据集，该数据集包含数以万计的手写数字的扫描图像，所有图像都是$28\times28$大小的灰度图像。用 $x$ 表示一个训练输入，每个训练输入 $x$ 可以看作一个 $28\times28=784$ 维的向量。用 $y=y(x)$ 表示对应的期望输出，$y$ 是一个$10$维的向量。用 $a$ 表示输入为 $x$ 时的实际输出，则 $a=\sigma(\omega \cdot x + b)$。我们希望有一个算法能够找到合适的权重 $\omega$ 和偏置 $b$ ，使网络的输出 $a$ 能够拟合 $y(x)$。

为了量化这个目标，我们定义一个代价函数 (*Cost Function*) 来描述拟合误差：
$$
C(\omega,b) \equiv \frac{1}{2n}\sum_{x}\left\|y(x)-a\right\|^{2}\tag{1-5}
$$
其中 $n$ 是训练样本的总数。我们要找到使 $C(\omega,b)$ 最小的 $\omega$ 和 $b$。

为了最小化 $C(v)$，我们希望 $\Delta v$ 变化的方向使 $\Delta C$ 小于 $0$，而 $\Delta C\approx\nabla C\cdot\Delta v$，其中$\nabla C\equiv(\frac{\partial C}{\partial v_{1}},...,\frac{\partial C}{\partial v_{n}})$。若选取一个较小的正数 $\eta$，使得 $\Delta v=-\eta \nabla C$，则 $\Delta C \approx -\eta\left \| \nabla C \right \|^{2}$。因此，我们可以改变 $v=v-\eta \nabla C$，直到 $C(v)$ 足够小：

$Repeat:$

​		$b = b -\eta \frac{\partial C}{\partial b}$

​		$\omega = \omega -\eta \frac{\partial C}{\partial \omega}$

上述算法称为批量梯度下降 (*Batch Gradient Descend*)。

然而有时候由于训练数据过于庞大，计算梯度的开销太大，因此研究人员提出了随机梯度下降 (*Stochastic Gradient Descent*)。这种算法把代价函数定义为单个训练样本的代价：
$$
C_{SGD}(\omega,b)\equiv \frac{1}{2}\left\|y(x^{(i)})-a^{(i)}\right\|^{2}\tag{1-6}
$$
其中 $x^{(i)}$ 表示全部训练样本 $x$ 中的第 $i$ 个样本，$a^{(i)}$ 表示输入为 $x^{(i)}$ 时的实际输出。该算法可以表示为：

$Repeat:$

​	$for$  $i=1:n:$

​		$b = b -\eta \frac{\partial C_{SGD}}{\partial b}$

​		$\omega = \omega -\eta \frac{\partial C_{SGD}}{\partial \omega}$

*SGD*虽然在收敛速度上优于*BGD*，但由于每次计算梯度时只对单个样本进行计算，因此容易陷入局部最优解。随着迭代次数的增加，*BGD*的精度将优于*SGD*。

为了兼顾收敛速度和精度，研究人员提出了小批量梯度下降 (*Mini-batch Gradient Descent*)。该算法在计算梯度时使用全部训练样本的一个子集。假设全部训练样本的数量为 $n$，每个子集中样本的数量为 $m$，则整个训练集可分为 $n/m$ 个*mini-batch*。若用 $x_{i}$ 表示第 $i$ 个子集，则该子集的代价函数为：
$$
C_{MGD}(\omega,b) \equiv \frac{1}{2m}\sum_{x_{i}}\left\|y(x_{i})-a_{i}\right\|^{2}\tag{1-7}
$$
因此，*MGD*算法可表示为：

$Repeat:$

​	$for$  $i=1:\frac{n}{m}:$

​		$b = b -\eta \frac{\partial C_{MGD}}{\partial b}$

​		$\omega = \omega -\eta \frac{\partial C_{MGD}}{\partial \omega}$

**注：一般情况下，工业界所说的随机梯度下降*SGD*实际上是指小批量梯度下降*MGD*。**







## 2. 反向传播算法如何工作##

上一章我们得出了降低拟合误差的算法，但算法中涉及到的梯度计算还未解决。对于梯度计算，链式法则是本质且直观的解法，但由于其代数式非常复杂，我们需要寻找更简便的方法。反向传播算法最初在$1970$年代提及，但直到$1986$年*David Rumelhart, Geoffrey Hinton, Ronald Williams*的著名论文人们才发现反向传播算法比传统方法更快。现在，反向传播算法已经是神经网络学习的重要组成部分了。

### 2.1 神经网络中使用矩阵快速计算输出的方法###

我们使用 $\omega_{jk}^{l}$ 表示从第 $l-1$ 层的第 $k$ 个神经元到第 $l$ 层的第 $j$ 个神经元的链接上的权重，使用 $b_{j}^{l}$ 表示第 $l$ 层第 $j$ 个神经元的偏置，使用 $a_{j}^{l}$ 表示第 $l$ 层第 $j$ 个神经元的激活值。有了这些表示，第 $l$ 层的第 $j$ 个神经元的激活值就和第 $l-1$ 层的激活值通过方程联系起来了：
$$
a_{j}^{l}=\sigma\left ( \sum_{k} \omega_{jk}^{l}a_{k}^{l-1}+b_{j}^{l} \right )\tag{2-1}
$$
为了用矩阵的形式重写这个表达式，我们对第 $l$ 层定义一个**权重矩阵** $\omega^{l}$，权重矩阵的元素就是连接到第 $l$ 层所有神经元的权重。再对第 $l$ 层定义一个**偏置向量** $b^{l}$，偏置向量的元素就是第 $l$ 层所有神经元的偏置。最后对第 $l$ 层定义一个**激活向量** $a^{l}$，激活向量的元素就是第 $l$ 层所有神经元的激活值。此外，我们还需要引入向量化函数，其含义是作用函数到向量或矩阵中的每个元素。根据以上这些表示，我们就可以写出矩阵形式的表达式：
$$
a^{l}=\sigma(\omega^{l}a^{l-1}+b^{l})\tag{2-2}
$$
在使用这个方程计算 $a^{l}$ 的过程中，我们计算了中间量 $z^{l}\equiv\omega^{l}a^{l-1}+b^{l}$。我们将 $z^{l}$ 称之为第 $l$ 层所有神经元的**带权输入**。

### 2.2 关于代价函数的两个假设###

反向传播的目标是计算代价函数 $C$ 分别关于 $\omega$ 和 $b$ 的偏导数 $\partial C/\partial \omega$ 和 $\partial C/\partial b$。原始的二次代价函数为 ：
$$
C=\frac{1}{2n}\sum_{x}\left\|y(x)-a^{L}\right\|^{2}\tag{2-3}
$$
其中 $L$ 是网络的总层数。

为了让反向传播可行，我们需要做出关于代价函数的两个假设。

1. 代价函数可以被写成一个在每个训练样本 $x$ 上的代价函数 $C_{x}=\frac{1}{2}\left\|y(x)-a^{L}\right\|^{2}$ 的均值 $C=\frac{1}{n}\sum_{x}C_{x}$。
2. 代价函数可以写作神经网络输出的函数。对于二次代价函数来说，$C=\frac{1}{2n}\sum_{x}\left\|y(x)-a^{L}\right\|^{2}$，对于确定的训练集，$n,x$ 是确定的，从而 $y(x)$ 也是确定的，所以该代价函数是神经网络输出 $a^{L}$ 的函数。

### 2.3 *Hadamard* 乘积###

假设 $s$ 和 $t$ 是两个维度相同的向量或矩阵，那么我们使用 $s\odot t$ 表示**按元素**的乘积，这种乘法称为***Hadamard* 乘积**。

*numpy*中可以使用 “$ * $” 运算符实现，*Matlab* 则使用 “$.*$” 运算符实现。

### 2.4 反向传播的四个基本方程###

反向传播其实是对权重和偏置变化影响代价函数过程的理解，最终极的含义其实就是计算偏导数 $\partial C/\partial \omega_{jk}^{l}$ 和 $\partial C/\partial b_{j}^{l}$。为了计算这些值，我们引入一个中间变量 $\delta_{j}^{l}$，并将其称之为第 $l$ 层第 $j$ 个神经元上的误差。反向传播将给出计算误差 $\delta_{j}^{l}$ 的流程，然后将其关联到计算 $\partial C/\partial \omega_{jk}^{l}$ 和 $\partial C/\partial b_{j}^{l}$ 上。

如果在第 $l$ 层第 $j$ 个神经元的带权输入上有一个很小的变化 $\Delta z_{j}^{l}$，使神经元的输出由 $\sigma(z_{j}^{l})$ 变成 $\sigma(z_{j}^{l}+\Delta z_{j}^{l})$。这个变化会向后传播，最终导致整个代价产生 $\frac{\partial C}{\partial z_{j}^{l}}\Delta z_{j}^{l}$ 的改变。我们的目的是找到让代价变小的 $\Delta z_{j}^{l}$。假设 $\frac{\partial C}{\partial z_{j}^{l}}$ 有一个较大的值（可正可负），那么我们可以选择与 $\frac{\partial C}{\partial z_{j}^{l}}$ 符号相反的 $\Delta z_{j}^{l}$ 来降低代价。而当 $\frac{\partial C}{\partial z_{j}^{l}}$ 接近于$0$时，我们并不能通过改变带权输入 $z_{j}^{l}$ 来改善太多代价，这时神经元已经接近最优了。因此我们可以认为 $\frac{\partial C}{\partial z_{j}^{l}}$ 是神经元的误差的度量，即 $\delta_{j}^{l}\equiv\frac{\partial C}{\partial z_{j}^{l}}$。

1. **输出层误差的方程：**

$$
\delta_{j}^{L}=\frac{\partial C}{\partial a_{j}^{L}}{\sigma}'(z_{j}^{L})\tag{2-4}
$$

等式右端的第一项 $\frac{\partial C}{\partial a_{j}^{L}}$ 表示代价随着第 $j$ 个输出激活值的变化而变化的速度。假如 $C$ 不太依赖于一个特定的输出神经元 ，那么 $\delta_{j}^{L}$ 就会很小，这也是我们想要的效果。等式右端的第二项 ${\sigma}'(z_{j}^{L})$ 刻画了在 $z_{j}^{L}$ 处激活函数 $a_{j}^{L}=\sigma(z_{j}^{L})$ 变化的速度。

等式中的每一项都很好计算， $\frac{\partial C}{\partial a_{j}^{L}}$ 可以直接通过代价函数的表达式计算。对于二次代价函数 $C=\frac{1}{2n}\sum_{x}\left\|y(x)-a^{L}\right\|^{2}$，$\frac{\partial C}{\partial a^{L}}=a^{L}-y$。而 ${\sigma}'(z^{L})=\sigma(z^{L}) \left [ 1-\sigma(z^{L}) \right ]$。由此可以得出方程$1$的矩阵形式：
$$
\begin{eqnarray}\delta^{L}&=&\frac{\partial C}{\partial a^{L}}\odot{\sigma}'(z^{L})\\
&=&(a^{L}-y)\odot\left \{\sigma(z^{L}) \left [ 1-\sigma(z^{L}) \right ] \right \}\end{eqnarray}\tag{2-5}
$$

2. **使用下一层的误差 $\delta^{l+1}$ 来计算当前层的误差 $\delta^{l}$：**

$$
\begin{eqnarray}\delta^{l}&=&\left[ (\omega^{l+1})^{T}\delta^{l+1} \right]\odot{\sigma}'(z^{l})\\
&=&\left[ (\omega^{l+1})^{T}\delta^{l+1} \right]\odot\left \{\sigma(z^{l}) \left [ 1-\sigma(z^{l}) \right ] \right \}\end{eqnarray}\tag{2-6}
$$

结合方程$1$和方程$2$，我们可以计算任何层的误差。

3. **代价关于网络中任意偏置的改变率**

$$
\frac{\partial C}{\partial b^{l}}=\delta^{l}\tag{2-7}
$$

4. **代价关于网络中任意权重的改变率**

$$
\frac{\partial C}{\partial \omega^{l}}=\delta^{l}(a^{l-1})^{T}\tag{2-8}
$$

### 2.5 四个基本方程的证明###

1. 由定义可知，$\delta_{j}^{l}=\frac{\partial C}{\partial z_{j}^{l}}$。应用链式法则，$\delta_{j}^{L}=\sum_{k}\frac{\partial C}{\partial a_{k}^{L}} \cdot \frac{\partial a_{k}^{L}}{\partial z_{j}^{L}}$。由于第 $k$ 个神经元的输出激活值 $a_{k}^{L}$ 只依赖于第 $k$ 个神经元的带权输入 $z_{k}^{L}$，所以当 $k\neq j$ 时， $\frac{\partial a_{k}^{L}}{\partial z_{j}^{L}}$的值为 $0$，$\delta_{j}^{L}$ 的结果就简化为 $\delta_{j}^{L}=\frac{\partial C}{\partial a_{j}^{L}} \cdot \frac{\partial a_{j}^{L}}{\partial z_{j}^{L}}=\frac{\partial C}{\partial a_{j}^{L}} \cdot{\delta}'(z_{j}^{L})$。这正是分量形式的方程$1$。
2. $\delta_{j}^{l}=\frac{\partial C}{\partial z_{j}^{l}}=\sum_{k}\frac{\partial C}{\partial z_{k}^{l+1}} \cdot\frac{\partial z_{k}^{l+1}}{\partial z_{j}^{l}}=\sum_{k}\frac{\partial z_{k}^{l+1}}{\partial z_{j}^{l}} \cdot\delta_{k}^{l+1}$，由于 $z_{k}^{l+1}=\sum_{j}\omega_{kj}^{l+1}a_{j}^{l}+b_{k}^{l+1}$，因此 $\frac{\partial z_{k}^{l+1}}{\partial z_{j}^{l}}=\omega_{kj}^{l+1}\cdot {\sigma}'(z_{j}^{l})$，代入前式可得 $\delta_{j}^{l}=\sum_{k}\omega_{kj}^{l+1}\delta_{k}^{l+1} {\sigma}'(z_{j}^{l})$。
3. $\delta_{j}^{l}=\frac{\partial C}{\partial z_{j}^{l}}=\frac{\partial C}{\partial b_{j}^{l}}\frac{\partial b_{j}^{l}}{\partial z_{j}^{l}}$。由于 $z_{k}^{l}=\sum_{j}\omega_{kj}^{l}a_{j}^{l-1}+b_{k}^{l}$，所以 $\frac{\partial b_{j}^{l}}{\partial z_{j}^{l}}=1$，故 $\frac{\partial C}{\partial b_{j}^{l}}=\delta_{j}^{l}$。
4. $\delta_{j}^{l}=\frac{\partial C}{\partial z_{j}^{l}}=\frac{\partial C}{\partial \omega_{jk}^{l}}\frac{\partial \omega_{jk}^{l}}{\partial z_{j}^{l}}$。由于 $\frac{\partial \omega_{jk}^{l}}{\partial z_{j}^{l}}=\frac{1}{a_{k}^{l-1}}$，所以 $\frac{\partial C}{\partial \omega_{jk}^{l}}=a_{k}^{l-1}\delta_{j}^{l}$。

### 2.6 反向传播算法###

1. 输入 $x$ ：为输入层设置对应的激活值 $a^{1}$。
2. 前向传播：对每个 $l=2,3,...,L$ 计算相应的 $z^{l}=\omega^{l}a^{l-1}+b^{l}$ 和 $a^{l}=\sigma(z^{l})$。
3. 输出层误差 $\delta^{L}$：计算向量 $\delta^{L}=\frac{\partial C}{\partial a^{L}}\odot{\sigma}'(z^{L})$。
4. 反向传播误差：对每个 $l=L-1,L-2,...,2$，计算 $\delta^{l}=\left[ (\omega^{l+1})^{T}\delta^{l+1} \right]\odot{\delta}'(z^{l})$。
5. 输出：代价函数的梯度由 $\frac{\partial C}{\partial \omega^{l}}=\delta^{l}(a^{l-1})^{T}$ 和 $\frac{\partial C}{\partial b^{l}}=\delta^{l}$ 给出。







## 3. 改进神经网络的学习方法##

### 3.1 交叉熵代价函数###

在一般的神经网络中，神经元在犯错比较明显（当前的权重或偏置与理想的权重或偏置差距较大）时的学习速度缓慢。由于神经元是通过改变权重和偏置，并以代价函数的偏导数 $\partial C/\partial \omega$ 和 $\partial C/\partial b$ 所决定的速度学习，因此神经元学习缓慢的根本原因就是偏导数偏小。

此前使用的二次代价函数为 $C=\frac{(y-a)^{2}}{2}$。由于 $a=\sigma(z)$，$z=\omega x+b$，所以 $\frac{\partial C}{\partial \omega}=(a-y){\sigma}'(z)x$，$\frac{\partial C}{\partial b}=(a-y){\sigma}'(z)$。

<p style="text-align:center"><img src="/assets/img/post/S函数.JPG"></p>

根据 $\sigma(z)$ 的函数性质，当神经元的实际输出与期望输出差距很大（例如期望输出为$0$，而实际输出接近$1$）时，${\sigma}'(z)$ 的值很小，从而使 $\partial C/\partial \omega$ 和 $\partial C/\partial b$ 的值变得很小，导致神经元学习缓慢。

#### 3.1.1 引入交叉熵代价函数####

为了解决这种问题，我们引入交叉熵代价函数：$C=-\frac{1}{n}\sum_{x}\left [ y\ln a +(1-y)\ln(1-a) \right ]$，其中，$n$ 是训练数据的总数，求和是在所有的训练输入 $x$ 上进行的，$a$ 是输入为 $x$ 时神经元的实际输出，$y$ 是对应的目标输出。交叉熵能够作为代价函数有两点原因：
1. 交叉熵始终是非负的；
2. 对于所有的训练输入，如果神经元的实际输出接近目标值，那么交叉熵将接近$0$。

综上所述，交叉熵具备代价函数应该具备的基本特性，这些特性也是二次代价函数具备的。但是交叉熵代价函数有一个比二次代价函数更好的特性就是可以避免学习速度下降的问题。

将 $a=\sigma(z)$ 代入交叉熵，并应用链式法则，可得：
$$
\begin{eqnarray} \frac{\partial C}{\partial \omega}&=&-\frac{1}{n}\sum_{x}\left ( \frac{y}{\sigma(z)}-\frac{(1-y)}{1-\sigma(z)} \right )\frac{\partial \sigma}{\partial \omega}\\&=&-\frac{1}{n}\sum_{x}\left ( \frac{y}{\sigma(z)}-\frac{(1-y)}{1-\sigma(z)} \right ){\sigma}'(z)x\\&=&-\frac{1}{n}\sum_{x}\frac{y(1-\sigma(z))-(1-y)\sigma(z)}{\sigma(z)(1-\sigma(z))}\sigma(z)(1-\sigma(z))x\\&=&\frac{1}{n}\sum_{x}x(\sigma(z)-y) \end{eqnarray} \tag{3-1}
$$
同理可得 $\frac{\partial C}{\partial b}=\frac{1}{n}\sum_{x}(\sigma(z)-y)$。从以上结果可以看出权重学习速度受到 $\sigma(z)-y$，也就是输出误差的影响。误差越大，学习速度越快。这就是我们期待的结果。

以上讨论只研究了一个神经元的交叉熵，现将其推广到多层多神经元网络上。假设 $y=y_1,y_2,...$ 是输出神经元上的目标值，而 $a_1^L,a_2^L,...$ 是实际输出值，那么我们定义交叉熵如下：
$$
C=-\frac{1}{n}\sum_x \sum_j \left[ y_j \ln a_j^L + (1-y_j) \ln (1-a_j^L) \right]\tag{3-2}
$$
需要注意的是，上述讨论只在输出神经元是S型神经元时有效，即当输出神经元是S型神经元时，交叉熵代价函数是比二次代价函数更好的选择。然而当输出神经元使用线性神经元时，由于 ${\sigma}'(z)$ 项的消失，二次代价函数将不会导致学习速度下降的问题。因而在这种情况下，二次代价函数就是一种合适的选择。

#### 3.1.2 交叉熵的来源####

使用二次代价函数的神经网络学习速度下降的原因在于代价函数关于网络参数的偏导数中包含 ${\sigma}'(z)$ 这一项，因此我们希望对于一个训练样本，其代价函数的偏导数满足：
$$
\left\{\begin{matrix}
\frac{\partial C}{\partial \omega}&=&x(a-y)\\ 
\frac{\partial C}{\partial b}&=&(a-y)
\end{matrix} \right.\tag{3-3}
$$
如果我们选择的代价函数满足这些条件，那么就可以实现初始误差越大，神经元学习越快的特性。实际上，我们可以通过数学的直觉推导出交叉熵的形式。由链式法则，我们有 $\frac{\partial C}{\partial b}=\frac{\partial C}{\partial a}{\sigma}'(z)$，代入 ${\sigma}'(z)=\sigma(z)(1-\sigma(z))=a(1-a)$ 后上个等式就变成 $\frac{\partial C}{\partial b}=\frac{\partial C}{\partial a}a(1-a)$。结合我们希望得到的 $\frac{\partial C}{\partial b}=(a-y)$ 就有了 $\frac{\partial C}{\partial a}=\frac{a-y}{a(1-a)}$。然后对此方程关于 $a$ 进行积分，得到 $C=-\left [ y\ln a +(1-y)\ln(1-a) \right ]+constant$。这是一个单独的训练样本对代价函数的贡献。为了得到整个训练集的代价函数，我们需要对所有的训练样本进行平均，得到了 $C=-\frac{1}{n}\sum_{x}\left [ y\ln a +(1-y)\ln(1-a) \right ]+constant$。

#### 3.1.3 柔性最大值####

对于学习速度缓慢的问题还有另一种解决问题的方法：**柔性最大值 (*softmax*)** 神经元层。

柔性最大值的想法其实就是为神经网络定义一种新的输出层，它首先计算带权输入 $z_{j}^{L}=\sum_{k}\omega_{jk}^{L}a_{k}^{L-1}+b_{j}^{L}$，然后在 $z_{j}^{L}$ 上应用**柔性最大值函数**。根据这个函数，第 $j$ 个神经元的激活值为
$$
a_{j}^{L}=\frac{e^{z_{j}^{L}}}{\sum_{k}e^{z_{k}^{L}}}\tag{3-4}
$$
其中，分母的求和是在柔性最大值层的所有输出神经元上进行的。柔性最大值具有以下性质：

1. 柔性最大值层的所有输出激活值加起来正好为$1$。


2. 柔性最大值层的所有输出激活值都是正数。

   *结合以上两点，我们看到柔性最大值层的输出是一些相加为 1 的正数的集合。换言之，柔性最大值的输出可以被看做是一个概率分布。*

3. 如果增加某一个带权输入分量，将会导致对应的输出激活值分量增加，同时其他输出激活值同比下降。这个性质可以通过柔性最大值关于 $z_k^L$ 的偏导数来证明。

   当 $j\neq k$ 时：
   $$
   \begin{eqnarray} \frac{\partial a_j^L}{\partial z_k^L}&=&\left( \frac{e^{z_j^L}}{\sum_k e^{z_k^L}}\right)_{z_k^L}^{'}\\
   &=&\frac{0\cdot \sum_k e^{z_k^L}-e^{z_j^L}\cdot e^{z_k^L}}{\left(e^{z_k^L}\right)^2}\\
   &=&-\frac{e^{z_j^L}}{e^{z_k^L}}<0\end{eqnarray}\tag{3-5}
   $$
   ​

   当 $j=k$ 时：
   $$
   \begin{eqnarray} \frac{\partial a_j^L}{\partial z_k^L}&=&\left( \frac{e^{z_k^L}}{\sum_k e^{z_k^L}}\right)_{z_k^L}^{'}\\
   &=&\frac{e^{z_k^L}\cdot \sum_k e^{z_k^L}-e^{z_k^L}\cdot e^{z_k^L}}{\left(\sum_k e^{z_k^L}\right)^2}\\
   &=&\frac{e^{z_k^L}}{\left(\sum_k e^{z_k^L}\right)^2}\left(\sum_k e^{z_k^L}-e^{z_k^L}\right)>0\end{eqnarray}\tag{3-6}
   $$
   因此，增加 $z_j^L$ 会提高对应的输出激活值 $a_j^L$ 并降低其他所有输出激活值。显然， $a_j^L$  的值不单单取决于  $z_j^L$ ，它取决于所有的  $z_k^L$。

柔性最大值神经元层使用**对数似然 (*log-likelihood*)** 代价函数来解决学习缓慢问题：
$$
C\equiv-\ln a_y^L\tag{3-7}
$$
其中，$y$ 表示网络输入为 $x$ 时对应的目标输出。对于 *MNIST* 手写数字识别，输入数字 $7$ 的图像，对应的对数似然代价就是 $-\ln a_7^L$。当网络表现好的时候，神经元估计的概率 $a_7^L$ 跟 $1$ 非常接近，此时代价 $-\ln a_7^L$ 就会很小；反之，网络表现糟糕的时候，概率 $a_7^L$ 就变得很小，代价 $-\ln a_7^L$ 随之增大。

对于使用对数似然代价函数的柔性最大值神经元层，有：
$$
C=\ln \sum_k e^{z_k^L}-\ln e^{z_y^L}\tag{3-8}
$$

$$
\frac{\partial \sum_k e^{z_k^L}}{\partial b_j^L}=e^{z_k^L}\tag{3-9}
$$

$$
\frac{\partial \sum_k e^{z_k^L}}{\partial \omega_{jk}^L}=e^{z_k^L}\cdot a_k^{L-1}\tag{3-10}
$$

$$
\frac{\partial e^{z_y^L}}{\partial b_j^L}=e^{z_y^L}\cdot \mathbf{y}_j\tag{3-11}
$$

$$
\frac{\partial e^{z_y^L}}{\partial \omega_{jk}^L}=e^{z_y^L}\cdot a_k^{L-1} \cdot \mathbf{y}_j\tag{3-12}
$$

其中，$\mathbf{y}$ 是一个第 $y$ 位为 $1$，其他所有位都是 $0$ 的向量。当 $j=y$ 时，$\mathbf{y}=1$；当 $j\neq y$ 时，$\mathbf{y}=0$。结合以上各式有：
$$
\begin{eqnarray}\frac{\partial C}{\partial b_j^L}&=&\frac{1}{\sum_k e^{z_k^L}}\cdot \frac{\partial \sum_k e^{z_k^L}}{\partial b_j^L}-\frac{1}{e^{z_y^L}}\cdot \frac{\partial e^{z_y^L}}{\partial b_j^L} \\
&=&\frac{e^{z_{j}^{L}}}{\sum_{k}e^{z_{k}^{L}}}-\frac{e^{z_y^L}}{e^{z_y^L}}\cdot \mathbf{y}_j \\
&=&a_j^L-\mathbf{y}_j
\end{eqnarray}\tag{3-13}
$$

$$
\begin{eqnarray}\frac{\partial C}{\partial \omega_{jk}^L}&=&\frac{1}{\sum_k e^{z_k^L}}\cdot \frac{\partial \sum_k e^{z_k^L}}{\partial \omega_{jk}^L}-\frac{1}{e^{z_y^L}}\cdot \frac{\partial e^{z_y^L}}{\partial \omega_{jk}^L} \\
&=&a_k^{L-1} \left( \frac{e^{z_{j}^{L}}}{\sum_{k}e^{z_{k}^{L}}}-\frac{e^{z_y^L}}{e^{z_y^L}}\cdot \mathbf{y}_j \right) \\
&=&a_k^{L-1} \left( a_j^L-\mathbf{y}_j \right)
\end{eqnarray}\tag{3-14}
$$

从这两个偏导数可以看出使用对数似然代价函数的柔性最大值神经元层在输出误差较大时，学习速度越快。

最后为了使用反向传播算法，我们需要计算 $\delta_j^L$。由定义，$\delta_j^L\equiv\frac{\partial C}{\partial z_j^L}=\frac{1}{\sum_k e^{z_k^L}}\cdot \frac{\partial \sum_k e^{z_k^L}}{\partial z_j^L}-\frac{1}{e^{z_y^L}}\cdot \frac{\partial e^{z_y^L}}{\partial z_j^L}$，而$\frac{\partial \sum_k e^{z_k^L}}{\partial z_j^L}=e^{z_j^L}$，$\frac{\partial e^{z_y^L}}{\partial z_j^L}=e^{z_j^L}\cdot \mathbf{y}_j$，故 $\delta_j^L=a_j^L-\mathbf{y}_j$。

### 3.2 过度拟合和规范化###

即使一个模型能够很好的拟合已有的数据，但并不表示是⼀个好模型。因为这可能只是因为模型中足够的⾃由度使得它可以描述几乎所有给定大小的数据集，而不需要真正洞察现象的本质。所以发生这种情形时，模型对已有的数据会表现的很好，但是对新的数据很难泛化。对⼀个模型真正的测验就是它对没见过的场景的预测能力。

过度拟合是神经网络的⼀个主要问题。这在现代网络中特别正常，因为网络权重和偏置数量巨大。为了高效地训练，我们需要⼀种检测过度拟合是不是发生的技术，这样我们不会过度训练。并且我们也想要找到⼀些技术来降低过度拟合的影响。

⼀般来说，最好的降低过度拟合的方式之⼀就是增加训练样本的量。有了足够的训练数据，就算是⼀个规模非常大的网络也不大容易过度拟合。不幸的是，训练数据其实是很难或者很昂贵的资源，所以这不是⼀种太切实际的选择。

#### 3.2.1 规范化

增加训练样本的数量是⼀种减轻过度拟合的方法。除此之外，⼀种可行的方式就是降低网络的规模。然而，大的网络拥有⼀种⽐小网络更强的潜力，所以这里存在⼀种应用冗余性的选项。幸运的是，还有其他的技术能够缓解过度拟合，即使我们只有⼀个固定的网络和固定的训练集合。这种技术就是**规范化**。

本节会给出⼀种最为常⽤的规范化手段——有时候被称为**权重衰减（weight decay）**或者**L2 规范化**。L2 规范化的想法是增加⼀个额外的项到代价函数上，这个项叫做**规范化项**。下⾯是规范化的交叉熵：
$$
C=-\frac{1}{n}\sum_{xj}\left [ y_{j} \ln a_{j}^{L}+(1-y_{j})\ln (1-a_{j}^{L}) \right ]+\frac{\lambda}{2n}\sum_{\omega}\omega ^{2} \tag{3-15}
$$
其中第一项就是常规的交叉熵表达式，第二项是使用因子 $\lambda/2n$ 进行了量化调整的所有权重的平方和，$\lambda>0$ 称为**规范化参数**，$n$ 是训练集的大小。 当然，对其他的代价函数也可以进行规范化，例如二次代价函数：
$$
C=\frac{1}{2n}\sum_{x}\left \| y-a^{L} \right \|^{2}+\frac{\lambda}{2n}\sum_{\omega}\omega ^{2} \tag{3-16}
$$
两者都可以写成以下形式：
$$
C=C_0+\frac{\lambda}{2n}\sum_\omega \omega^2 \tag{3-17}
$$
其中 $C_0$ 是原始的代价函数。

直觉地看，规范化的效果是让网络倾向于学习小⼀点的权重，其他的东西都⼀样的。大的权重只有能够给出代价函数第⼀项足够的提升时才被允许。换言之，规范化可以当做⼀种寻找小的权重和最小化原始的代价函数之间的折中。这两部分之前相对的重要性就由$\lambda$ 的值来控制了：$\lambda$ 越小，就偏向于最小化原始代价函数，反之，倾向于小的权重。

要实现规范化的神经网络，我们需要计算网络中所有权重和偏置的偏导数，对方程 $3$-$17$ 进行求导得：
$$
\begin{eqnarray}
\frac{\partial C}{\partial \omega}&=&\frac{\partial C_0}{\partial \omega}+\frac{\lambda}{n}\omega \\
\frac{\partial C}{\partial b}&=&\frac{\partial C_0}{\partial b}
\end{eqnarray} \tag{3-18}
$$
其中，$\partial C_0/\partial \omega$ 和 $\partial C_0/\partial b$ 可以通过反向传播计算，所以计算规范化的代价函数的梯度其实很简单：仅仅需要反向传播，然后加上 $\frac{\lambda}{n}$ 得到所有权重的偏导数，而偏置的偏导数根本没有变化。
$$
\begin{eqnarray}
b &\rightarrow& b-\eta \frac{\partial C_0}{\partial b} \\
\omega &\rightarrow& \omega-\eta \frac{\partial C_0}{\partial \omega}-\frac{\eta \lambda}{n}\omega \\
&=&\left ( 1-\frac{\eta \lambda}{n} \right )\omega-\eta \frac{\partial C_0}{\partial \omega}
\end{eqnarray} \tag{3-19}
$$

#### 3.2.2 为何规范化可以帮助减轻过度拟合

通常的说法是：小的权重在某种程度上，意味着更低的复杂性，也就对数据给出了⼀种更简单却更强大解释，因此应该优先选择。

假设神经网络中⼤多数是很小的权重，更小的权重意味着网络的行为不会因为我们随便改变了⼀个输入而改变太大。这会让局部噪声更难在规范化网络的学习中造成影响，即规范化⼀种让单个的证据不会影响网络输出太多的方式。相对的，规范化网络对整个训练集中频繁出现的数据有更好的学习效果。对比看，大权重的网络可能会因为输入的微小改变而产生比较大的行为改变。简言之，规范化网络受限于根据训练数据中常见的模式来构造相对简单的模型，而能够抵抗训练数据中的噪声的影响。这可以让我们的网络对看到的现象进行真实的学习，并能够根据已经学到的知识更好地进行泛化。

#### 3.2.3 规范化的其他技术

**L1规范化**：
$$
C=C_0+\frac{\lambda}{n}\sum_\omega \left | \omega \right |\tag{3-20}
$$
对上式求导可得：
$$
\frac{\partial C}{\partial \omega}=\frac {\partial C_0}{\partial \omega}+\frac{\lambda}{n}sgn(\omega)\tag{3-21}
$$
对L1规范化的网络进行更新的规则是：
$$
\omega \rightarrow \omega-\frac{\eta \lambda}{n}sgn(\omega)-\eta\frac{\partial C_0}{\partial \omega}\tag{3-22}
$$
与L2规范化相比，这两种规范化都惩罚大的权重，但权重缩小的方式不同。在L1规范化中，权重通过一个常量向0进行缩小；而在L2规范化中，权重通过一个和 $\omega$ 成比例的量进行缩小。所以，当一个特定的权重绝对值 $\left | \omega \right |$ 很大时，L1规范化的权重缩小得远比L2规范化小得多。相反，当一个特定的权重绝对值 $\left | \omega \right |$ 很小时，L1规范化的权重缩小得远比L2规范化大得多。最终的结果就是：L1规范化倾向于聚集网络的权重在相对少量的高重要度连接上，而其他权重就会被驱使向0接近。

### 3.3 权重初始化

创建网络之后，我们需要进行权重和偏置的初始化，之前采用的初始化方式是根据独立高斯随机变量来选择权重和偏置，它们被归一化为均值为0，标准差为1。但这种方式会对网络学习的速度造成潜在影响。假设我们使用1000维的训练输入 $x$ ，其中一般的输入为1，另一半为0，那么隐藏神经元的带权和为 $z=\sum_j \omega_j x_j + b$。由于独立随机变量和的方差是每个独立随机变量方差的和，所以 $z$ 本身是一个均值为0，标准差为 $\sqrt{501}\approx 22.4$ 的高斯分布：

<p style="text-align:center"><img src="/assets/img/post/宽高斯分布.JPG"></p>

从图中可以看出 $z$ 的模会变得非常的大，那么隐藏神经元的输出 $\sigma(z)$ 就会接近1或者0，也就表示我们的隐藏神经元会饱和，这些权重会学习的非常缓慢。

为了避免这种类型的饱和，我们使用均值为0，标准差为 $1/\sqrt{n_{in}}$ 的高斯随机分布初始化这些权重（$n_{in}$为神经元的输入个数）。假设我们使用1000维的训练输入 $x$ ，其中一般的输入为1，另一半为0，那么隐藏神经元的带权和 $z=\sum_j \omega_j x_j + b$ 就变成了一个均值为0，标准差为 $\sqrt{3/2} \approx 1.22$ 的高斯分布：

<p style="text-align:center"><img src="/assets/img/post/窄高斯分布.JPG"></p>

这样的神经元更不可能饱和，基本不会遇到学习速度下降的问题。



## 4. 深度学习

### 4.1 卷积神经网络的向后传播

https://www.cnblogs.com/pinard/p/6494810.html

持续更新中……
