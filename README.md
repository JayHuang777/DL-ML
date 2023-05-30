# 从0学会神经网络

### 数学基础

在研究一元函数时，从研究函数的变化率引入了导数的概念，对于多元函数同样需要讨论它的变化率.由于多元函数不止一个自变量，研究起来要复杂得多.但是，我们可考虑多元函数关于其中一个自变量的变化率，例如：理想气体的体积：

![img](https://pic2.zhimg.com/80/v2-5f1ea590ed0b7fe794fe6fe103338001_720w.webp)

因此，我们引入下面的偏导数概念。

偏导数的定义

![img](https://pic4.zhimg.com/80/v2-f355683a5bf54bad91f3026d44d34023_720w.webp)

偏导数的计算

从偏导数的定义可以看出，计算多元函数的偏导数并不需要新的方法，若对某一个自变量求导，只需将其他自变量常数，用一元函数微分法即可.于是，一元函数的求导公式和求导法则都可以移植到多元函数的偏导数的计算上来.

![img](https://pic4.zhimg.com/80/v2-0c478b99f4ea754fe05bad0cc9e0eb03_720w.webp)

![img](https://pic3.zhimg.com/80/v2-eeb75ff725674ea7327660e0613352d2_720w.webp)

### 线性代数

转置矩阵：将矩阵A的行列互换，而不改变其先后次序得到的n×m阶矩阵，记为

![img](https://pic4.zhimg.com/80/v2-7511ae384c5fa1db53cb31d3127a8027_720w.webp)

或A’。

###### 特征向量和特征值

矩阵A的某个特征值为 m1, 对应的特征向量是 x1。x1是以A为坐标系的坐标向量，将其变换到以![image-20230530115055550](C:\Users\78300\AppData\Roaming\Typora\typora-user-images\image-20230530115055550.png)为坐标系后得到的坐标向量 与 它原来的坐标向量 永远存在一个 m1 倍的伸缩关系。

为了方便理解举一个简单的例子，假如矩阵A如下，可以看到它的特征值有2个，分别是1,100，分别对应2个特殊的特征向量，即 [1,0],[0,1]。

![image-20230530115011772](C:\Users\78300\AppData\Roaming\Typora\typora-user-images\image-20230530115011772.png)

所以矩阵A左乘任意的一个向量x，其实都可以理解成是把向量x沿着这2个特征向量的方向进行伸缩，伸缩比例就是对应的特征值。可以看到这2个特征值差别是很大的，最小的只有1，最大的特征值为100。

看下图的例子，矩阵A和向量 [1,1]相乘得到 [1,100]，这表示原来以A为坐标系的坐标[1,1]，经过转换到以![image-20230530115026946](C:\Users\78300\AppData\Roaming\Typora\typora-user-images\image-20230530115026946.png)为坐标系后 坐标变成了 [1,100]。我们直观地理解就是矩阵A把向量[1,1]更多地往y轴方向拉伸。

![img](https://pic4.zhimg.com/80/v2-e26d7fe8fd2898a8ae7d2c72c179fc27_720w.webp)

假如A是多维(n)矩阵，且有n个不同的特征值，那么就可以理解成这个矩阵A和一个向量x相乘其实就是把向量x往n个特征向量的方向进行拉伸，拉伸比例是对应的特征值。那这样有什么作用呢？

###### 特征值和特征向量的应用

意义就在于如果我们知道了特征值的大小，有时为了减少计算了，我们可以只保留特征值较大的，比如上面的图片中，我们可以看到变换后的向量x轴适合原来一样的，而y轴方向拉伸了100倍，所以通常为了实现压缩算法，我们可以只保留y轴方向的变换即可。 对应到高维情况也是类似的，多维矩阵会把向量沿着多个方向拉伸，有的方向可能拉伸幅度很小，而有的很大，我们只需要保留幅度大的即可达到压缩的目的。

### 张量

张量是一个多维数组。更正式地说，一个 N 阶张量是 N 个向量空间元素的张量积，每个向量空间都有自己的坐标系。张量的阶数（the order of a tensor）也称为维数（dimensions）、模态（modes）、或方式（ways）。一阶张量是一个矢量，二阶张量是一个矩阵，三阶或更高阶的张量叫做高阶张量。

### ![img](https://img-blog.csdnimg.cn/20200603170342703.png)

###### 张量相当于在矩阵的基础上增强了对数据空间的拓展：

一维张量：比如：[1,2,3]

二维张量：比如：[[1,2,3],[4,5,6]]

三维张量：比如：[[[1,2,3],[1,2,3]],[[4,5,6],[4,5,6]],[[7,8,9],[7,8,9]]]

.....

三维主要是引入了RGB，四维主要引入了时间序列

### python基础

[Python 环境搭建](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzkxNDI3NjcwMw%3D%3D%26mid%3D2247493298%26idx%3D1%26sn%3Df7c43c9afb686b9c4828ad2645126e53%26chksm%3Dc1724e82f605c794472526d90c8cf5258171fd11e2efc946005bd64b367c68ba543a9d55c8f5%26scene%3D21%23wechat_redirect)

[Python 基础语法](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzkxNDI3NjcwMw%3D%3D%26mid%3D2247493299%26idx%3D1%26sn%3Dfe6d1d42f6b1323b4b38556171c0e809%26chksm%3Dc1724e83f605c795bfdd76d05f705072b66c1a914eed18703573368cabd0da7474c610011612%26scene%3D21%23wechat_redirect)

[Python 变量与数据类型](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzkxNDI3NjcwMw%3D%3D%26mid%3D2247493300%26idx%3D1%26sn%3D55b9b8dc521557f54aff515913104cf2%26chksm%3Dc1724e84f605c7924fac8bd336917c0b6820699097a5022e30416cd7c6f1b36a62b42aa38ef1%26scene%3D21%23wechat_redirect)

[Python 流程控制](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzkxNDI3NjcwMw%3D%3D%26mid%3D2247493301%26idx%3D1%26sn%3D64fb9b42e63a7c6380f3f1982d48c6d5%26chksm%3Dc1724e85f605c793af86e5fd2b43002c07e75b73f0ba3a520ed19954edda3b39e9a71578ebed%26scene%3D21%23wechat_redirect)

[Python函数](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzkxNDI3NjcwMw%3D%3D%26mid%3D2247493302%26idx%3D1%26sn%3D0609a5efc9165481da4a031431c3e639%26chksm%3Dc1724e86f605c7904e2429488bc5e6d43a8d7a94a15d613660da08b2b78743f85322f1aa72c1%26scene%3D21%23wechat_redirect)

[Python 模块和包](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzkxNDI3NjcwMw%3D%3D%26mid%3D2247493303%26idx%3D1%26sn%3Dfb7fab817946cfd49af61a519e47fe91%26chksm%3Dc1724e87f605c791b82f0f62271e216fd64d93d57ce6acee6bc2362363c5681dcee0779ed745%26scene%3D21%23wechat_redirect)

[Python 数据结构--序列](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzkxNDI3NjcwMw%3D%3D%26mid%3D2247493304%26idx%3D1%26sn%3D0e9a0d34138a55ce787794ab8a07eeb4%26chksm%3Dc1724e88f605c79ee061280feb2df946d5ce0fc8f58ae1bdeed05eecf1c7c76d4511d6925fd9%26scene%3D21%23wechat_redirect)

[Python List](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzkxNDI3NjcwMw%3D%3D%26mid%3D2247493305%26idx%3D1%26sn%3D775716cf6e1178570d3b04ae9c140a19%26chksm%3Dc1724e89f605c79f3a1ed293daa2802d24ede6f14b4fee4127f4688aedc8da2aa48b2c7aa1db%26scene%3D21%23wechat_redirect)

[Python tupple](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzkxNDI3NjcwMw%3D%3D%26mid%3D2247493306%26idx%3D1%26sn%3D5f312f55b123c6675c2deb8772727c88%26chksm%3Dc1724e8af605c79cb92faab2fc6f00c6cc4677d0b77d6dad9013f06171681a1bddaf45e70859%26scene%3D21%23wechat_redirect)

[Python 类与对象](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzkxNDI3NjcwMw%3D%3D%26mid%3D2247493311%26idx%3D1%26sn%3D096f1e5bd899942bbb30c9d39dcc0ff7%26chksm%3Dc1724e8ff605c799d01eeb404d55f2c1f719d6bcd71cbdaa12a82c219460314d5b1048a0c3ab%26scene%3D21%23wechat_redirect)

[Python 字典](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzkxNDI3NjcwMw%3D%3D%26mid%3D2247493312%26idx%3D1%26sn%3D5a0fea8a37f0700435f6a640c010805f%26chksm%3Dc1724ef0f605c7e65c33ab036eaa02aa557478bab93abf428b0d82568c99e18a0fde6405ff8e%26scene%3D21%23wechat_redirect)

[Python 集合](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzkxNDI3NjcwMw%3D%3D%26mid%3D2247493314%26idx%3D1%26sn%3Dcb92c32937dfd1f1bba3a0ae5862b7f4%26chksm%3Dc1724ef2f605c7e48460f02819a8c862d8a85ed23ead53e452ebf8467db8f9dd7b5a2e15157e%26scene%3D21%23wechat_redirect)

[Python 函数的参数](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzkxNDI3NjcwMw%3D%3D%26mid%3D2247493315%26idx%3D1%26sn%3D635e925802e4ff535ccbb847072e1425%26chksm%3Dc1724ef3f605c7e56da4c5d710261504a4ef20016843ab4a2993241b4a3656d332e946fc155f%26scene%3D21%23wechat_redirect)

[Python 高阶函数](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzkxNDI3NjcwMw%3D%3D%26mid%3D2247493316%26idx%3D1%26sn%3D4ae3bedbc64482979bcaf369ed1a19dd%26chksm%3Dc1724ef4f605c7e214fcf1f5ba3c86eb49fcd63bd7a1de6b64e6589aeea843c2f3330d8942f8%26scene%3D21%23wechat_redirect)

[Python 输入输出](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzkxNDI3NjcwMw%3D%3D%26mid%3D2247493317%26idx%3D1%26sn%3De6d6559ab903560d65e8d5f8dee3e016%26chksm%3Dc1724ef5f605c7e39b6c4024cc710bc398261a66bf60805e160bf910024c7a26f169e1559a26%26scene%3D21%23wechat_redirect)

[Python 错误和异常](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzkxNDI3NjcwMw%3D%3D%26mid%3D2247493318%26idx%3D1%26sn%3Da29f032ef1ec007dec3a05b7e50a5ecf%26chksm%3Dc1724ef6f605c7e0523a5ad06e67e29e74bd497571bf52c6d16b06eec34e594546f70f41a463%26scene%3D21%23wechat_redirect)

[正则表达式](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzkxNDI3NjcwMw%3D%3D%26mid%3D2247493369%26idx%3D1%26sn%3Db2591d7d1dd78f1404a1c99297b98315%26chksm%3Dc1724ec9f605c7df148d831ccc01d3ca97b6119509ea8cf973dcaeb6c25b02e74d2e4e726e31%26scene%3D21%23wechat_redirect)

# 好的，你现在已经有相关的基础了，那么我们就开始机器学习的范畴吧

### 监督学习：


