---
title: 解决安装AI算法库TensorFlow 2.0的最新入坑指南以及详细的安装教程【分别在linux和windows系统下安装】
categories: 安装与配置
date: 2020-1-05 22:57:06  
photos: 
  - "https://images.wallpaperscraft.com/image/single/lake_mountains_trees_129959_1280x720.jpg"

---

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS0yNjE2NGY2ZDFiYzRiYjNk?x-oss-process=image/format,png)

**（转载作者请注明出处）**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS03MTc4OWUyZGY2ZDZlZDZm?x-oss-process=image/format,png)

**废话不多说，直接上干货**

....

**正文开始**

众所周知，学习深度学习和人工智能技术的科技工作者对Tensorflow的安装一直是件麻烦的事情，其实也没那么难，只是在于操作的方法是否合理和规范而已。合理正确的安装命令和正确的操作环节是成功的关键因素。为此，我们在本文中将详细说明linux和windows两种OS系统关于TensorFlow的安装教程，并且将重点说明目前最新版本TensorFlow2.1.0的安装要素和测试检验。
<!-- more -->
（注意：本文的linux系统教程适用于deepin和ubuntu 18.04两种，其他系统安装类似）

**TensorFlow简介**

```
https://baike.baidu.com/item/Tensorflow/18828108
```

**一、在Windows10系统开展如下操作**

安装前提：

已经安装好Anaconda3, 由于国外的镜像文件下载较慢，所以我选择了国内的镜像——*h**ttps://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/*

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1jZTdjNmUwNTg3YTdjNDg4?x-oss-process=image/format,png)

以上请自行安装....

**1、前期准备：****Anaconda3的终端上的过渡**

打开已经安装好的Anaconda3

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1iNTdmN2IyM2RlODIzNDc4?x-oss-process=image/format,png)

运行 Anaconda Prompt，输入如下命令检验是否安装成功

```
conda list
```

输出：

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS01YTg0OTA3Yzc5NTAxYzNi?x-oss-process=image/format,png)

再次输入清华镜像检验一下

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
```

这里是因为我已经安装过了，所以结果是这样的。

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1hODBhM2UyMDM3MWU5ZDc2?x-oss-process=image/format,png)

**2、开始安装TensorFlow**

**（1）运行 Anaconda Prompt，输入以下命令，创建版本为python 3.7.4的tensorflow 环境：**

```
conda create -n tensorflow python=3.7.4
```

这里的TensorFlow我以前已经安装过了，所以下面再给大家安装另一个tensorflow2环境：输入命令

```
conda create -n tensorflow2 python=3.7.4
```

输出：

```
(base) C:\Users\kangs>conda create -n tensorflow2 python=3.7.4
```

然后打开Anaconda Navigator，进入点击中间的选项applications on可以看到我们刚刚创建的**TensorFlow2**，这里的**TensorFlow**是我以前安装的环境：

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1lNTMxMjNiMTFhZGM3Zjc5?x-oss-process=image/format,png)

打开上述左侧菜单栏，也就是开始菜单下的Anaconda Navigator 左边的 Environments，点击之后可看到此环境：

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1hOTQ4NTUzNWYyMjc5ODg5?x-oss-process=image/format,png)

**（2）启动tensorflow2环境**

这里有两种启动方式：第一种是连续刚刚的 Anaconda Prompt终端输入

```
activate tensorflow2
```

输出

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS0xOTJkNTMwMWZlZDA2MjNl?x-oss-process=image/format,png)

第二种打开方式，启动刚刚的Anaconda Navigator，进入环境点击

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS05MGRkNGMzYmE4MDNhNzJm?x-oss-process=image/format,png)

**（3）安装cpu版本的tensorflow **

有两种方法可以安装：

方法一：cpu版本（我推荐的）

```
pip install --ignore-installed --upgrade tensorflow
```

方法二：gpu版本, 注意: gpu版要事先选好, 并装好CUDA和cuDNN

```
pip install --ignore-installed --upgrade tensorflow-gpu
```

执行结果：

```
​
```

这样就安装成功了，当不使用TensorFlow时，可以通过deactivate来关闭TensorFlow环境：

(不过先不要着急关闭，我们下面还有进行检验，如果最后不用，再关掉)

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1kNzhhODRjMWQzMzBjNmE5?x-oss-process=image/format,png)

**（4）测试tensorflow**

```
python --version
```

创建一个项目：

```
import tensorflow as tf
```

执行结果显然是可行的，测试成功： 

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1jOTdiMWE3ZjhmYWQ0MDI4?x-oss-process=image/format,png)

**提示**：如果遇到No module named ‘tensorflow’，那是因为没有在tensorflow的环境下打开它们，所以记得激活tensorflow的环境。

**（5）解决你安装的Tensorflow环境不能安装ipython,spyder等插件**

**错误提示如下图：**

**“无法定位程序输入点OPENSSL_sk_new_reserve于动态链接库...... libssl-1_1-x64.dll上的问题”‘’**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS01OTgwOWZjNDMyYWZjMTQx?x-oss-process=image/format,png)

**解放方法：****互换 libssl-1_1-x64.dll文件，时间必须一样**

第一步 打开D:\Anaconda\Library\bin文件夹下面的 libssl-1_1-x64.dll

第二步 D:\Anaconda\DLLs 文件夹下面的 libssl-1_1-x64.dll

如果两个文件夹时间不一样的话，把两个文件换成一样的就可以了。

**注意：**为了避免系统出错，要事先备份好一样的文件，以避免修改后出错而找不会原来的文件。最简单的方法就是复制换成另一个名称的，例如我加了一个“源文件”。

**互换如下图：**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1hM2Q4ZTA0M2RlM2NiYWNm?x-oss-process=image/format,png)

**换文件之后,时间都是****2018/6/28 21:00**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS03N2ZiYzE1YTY1MDczNWMw?x-oss-process=image/format,png)

**下面再次点击安装插件就可以了**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1iZDg1NzM4YTM2NmViY2Rk?x-oss-process=image/format,png)

**安装成功，如图下所示** 

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1mYjc3ZjEzZGQwNzkwMWVj?x-oss-process=image/format,png)

**二**、在Linux******系统开展如下操作**

这里使用的虚拟机VMware Workstation Pro 15.0安装的linux系统进行操作的，如图：

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1hYjFlZDI3NjQ3ZGYxMDlm?x-oss-process=image/format,png)

**下面打开deepin系统开始进行安装**

**1、先安装Anaconda3**

(1)、下载：https://www.anaconda.com/distribution/#linux

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1mYTE1NTU2NDYxNzQ4YjBm?x-oss-process=image/format,png)

将下载好的anacoda3放在主目录下（home）

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1iZjExYjIyNDc2YzY3YzMy?x-oss-process=image/format,png)

（2）、安装命令：

```
bash Anaconda3-2019.10-Linux-x86_64.sh
```

回车运行如下

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1hYmM2MDY1N2ExYmMzZTY4?x-oss-process=image/format,png)

按回车键继续，会读取许可，出现个More，一直按回车以及yes,然后会出现下图：

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1lN2Q0NmNlNzIzNWI0MmY1?x-oss-process=image/format,png)

一直往下回车和yes

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1mZDcyZTk1NjI5ZTBiYmM0?x-oss-process=image/format,png)

安装成功

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS02ZWI0NGU2MjIxNGRmNGI1?x-oss-process=image/format,png)

（3）、配置环境变量

为了方便编辑环境变量，这里安装一个编辑器gedit，命令

```
sudo apt-get install gedit
```

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS01M2I1YWU4NTQwMWEyODhl?x-oss-process=image/format,png)

打开环境变量，并使用命令：sudo gedit ~/.bashrc查看系统环境，文件末尾输入以下代码:

```
在文件的末尾加上下述代码：
```

如下图

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1mMDk0NzBlMTU1ZmVjYmQy?x-oss-process=image/format,png)

（5）、更新bashrc，并查看Anacoda3的安装情况

```
source ~/.bashrc
```

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS02MTljYjdhMDkzOTRlYWNm?x-oss-process=image/format,png)

查询当前已经安装的conda库

```
conda list
```

安装库(这里的***代表库名称）, 如果没有你的库文件，可以选择这个命令：

```
conda install ***
```

更新库

```
conda update ***
```

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS0zOTFhODQ0MWQzMzRiZDAz?x-oss-process=image/format,png)

（5）、(如果你不用还可以卸载)，不卸载就跳过此处，建议不要卸载

卸载conda: 直接删除anaconda文件夹即可：

```
rm -rf anaconda3
```

(6)、进入和退出 conda base 环境

进入 conda base 环境

```
conda activate base
```

退出 conda base 环境

```
conda deactivate
```

编辑 conda 环境变量

```
vim ~/.bashrc 或者 sudo gedit ~/.bashrc
```

**2、使用以上Anaconda3安装Tensorflow**

(1)、在linux终端或cmd中输入以下命令搜索当前可用的tensorflow版本，如果没有就要创建：

```
conda search -t conda tensorflow
```

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1mZThhNGUxMTIzNTk1ZTgw?x-oss-process=image/format,png)

安装完成之后，最后还提示激活与退出

```
# To activate this environment, use
```

（2）、正式安装tensorflow

激活虚拟环境后conda activate tf，我们开始用conda安装tensorflow吧！

 如果你还不知道GPU是什么东东，那你的计算机里肯定没有安装cuda、cudnn 、显卡之类的东西，那就安装CPU版本；

安装CPU版本的tensorflow(推荐安装)：

```
conda install tensorflow
```

安装GPU版本

（如果你不理解建议不要安装，初学者还是卸载CPU版本的tensorflow吧！）

```
conda install tensorflow-gpu
```

等待几分钟之后，看看自己是否安装成功：

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1jOGNiNDNjMWU4ZDZiYzM0?x-oss-process=image/format,png)

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS00MWRhMmRhOGRhNjVkNzBh?x-oss-process=image/format,png)

（3）、测试安装是否成功：在python3.7.6下导入tensorflow：

(不报错的话说明成功安装了)

```
import tensorflow as tf
```

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1kZjJjMWU5OTMwZTYzYjQ0?x-oss-process=image/format,png)

至此我们完整地安装完毕，以下是一些附件资料

* * *

**附件资料**

## Conda的环境管理及知识点

```
#conda版本查看
```

### Conda的包管理知识点

```
# 查看当前环境下已安装的包
```

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS00M2FhYTQwNTdmMmYxZDU2?x-oss-process=image/format,png) 
**参考文献：**

```
​https://blog.csdn.net/qq654129588/article/details/79917515
https://ask.csdn.net/questions/657580
https://www.cnblogs.com/HongjianChen/p/8385547.html
https://blog.csdn.net/qq_45100771/article/details/102868264
https://blog.csdn.net/superjunenaruto/article/details/9539068
```

***您可能喜欢的文章：***

*   [数据分析开篇：一个简单的应用（2019/11/04）](http://mp.weixin.qq.com/s?__biz=MzIwNzUwOTY1Nw==&mid=2247485246&idx=1&sn=21e5385c26fe6866b056619e87d8829b&chksm=971006c8a0678fde4c18db3115ed91490b8c653b25aebd5bd61605eda5eeb992008c946a447a&scene=21#wechat_redirect)

*   [2020年数据分析必知必会（一）：NumPy数组](http://mp.weixin.qq.com/s?__biz=MzIwNzUwOTY1Nw==&mid=2247485315&idx=1&sn=ddec9e455757a2af1df8a3fca319b173&chksm=97100675a0678f6316f39e5569d92b5727397d86888ec28fc49ae5e3fe71ab2cc8df00889e8a&scene=21#wechat_redirect)

*   [2020年数据分析必知必会（二）：NumPy摘要----文章末尾附Python](http://mp.weixin.qq.com/s?__biz=MzIwNzUwOTY1Nw==&mid=2247485315&idx=1&sn=ddec9e455757a2af1df8a3fca319b173&chksm=97100675a0678f6316f39e5569d92b5727397d86888ec28fc49ae5e3fe71ab2cc8df00889e8a&scene=21#wechat_redirect)

*   [2020年数据分析必知必会（三）：数组的形状和属性（有福利赠予）](http://mp.weixin.qq.com/s?__biz=MzIwNzUwOTY1Nw==&mid=2247485351&idx=1&sn=c0d14713527646dd7aa113b1b04452d4&chksm=97100651a0678f47fff6692d934aaab91f53b52348dc7f567be99e21358207064ae4b0c328b1&scene=21#wechat_redirect)

*   [数据分析必知必必会（四）：数组的转换，视图，拷贝，索引和广播（这里的“广播”是一个数组的应用：数据处理旧手机铃声）](http://mp.weixin.qq.com/s?__biz=MzIwNzUwOTY1Nw==&mid=2247485406&idx=1&sn=50b327edec6ea33605a6551623133a9d&chksm=97100628a0678f3ee979a6f01850b63dd1298dc07ca0e5f098e973ff1b8c888fd0125257e98a&scene=21#wechat_redirect)

*   [2020年数据分析必知必会（五）：统计和线性代数（使用Numpy与Scipy实现）](http://mp.weixin.qq.com/s?__biz=MzIwNzUwOTY1Nw==&mid=2247485459&idx=1&sn=5d9eaed9972b6dcd7e838530b6c9445f&chksm=971009e5a06780f3d8f401ecb337ce88de166edfec059d43c9a0be2274c01322e32ce84a5718&scene=21#wechat_redirect)

*   [2020年数据分析必知必会（六）：离散式复制的创建（以北京最近的猪肉价格为例子）](http://mp.weixin.qq.com/s?__biz=MzIwNzUwOTY1Nw==&mid=2247485579&idx=1&sn=a5128b4d508a51a5f8d3ba0b82364752&chksm=9710097da067806bcd58e326e24971690d0736fe69a85f390b8043415600ae7ceb72d4729a52&scene=21#wechat_redirect)

*   [2020年数据分析必知必会（七）：pandas入门与数据结构基础](http://mp.weixin.qq.com/s?__biz=MzIwNzUwOTY1Nw==&mid=2247485593&idx=1&sn=7a7801b8daa3f5b43a79130df721aba1&chksm=9710096fa0678079c647b363eb92facda180402defcb4375322f30035f96902b57843226b8da&scene=21#wechat_redirect)

*   [2020年数据分析必知必会（八）：使用pandas查询数据和统计分析的应用（短小但强大）](http://mp.weixin.qq.com/s?__biz=MzIwNzUwOTY1Nw==&mid=2247485601&idx=1&sn=c8915dfba4416efc8149eb11090c70bd&chksm=97100957a067804179633e45417e803481a0057dd4bc842e6652ba4cc1a19cb24709b16c0d68&scene=21#wechat_redirect)

*   [2020年数据分析必知必会（九）：数据的分组与聚合](http://mp.weixin.qq.com/s?__biz=MzIwNzUwOTY1Nw==&mid=2247485699&idx=1&sn=e4a9946cfa5eb87e1796a57b2dacf111&chksm=971008f5a06781e3a09e1eef39a7edf78703f2e44451158da06d6570a84dd4e5fe3f94bb6f28&scene=21#wechat_redirect)

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS01MDkwOTViM2RmZDc4NTA4?x-oss-process=image/format,png)

***认认真真系统学习数据分析***

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS0yN2NjOTg5YmU0MDJiYTZj?x-oss-process=image/format,png)

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS0yYThlNmRiNzY3OGMxMzMx?x-oss-process=image/format,png)
