---
title: 值得你阅读的Hexo个人博客搭建：不用购买服务器，不用购买域名，不要钱，不用敲代码等等，是的，你没有看错，快来转载学习吧！
categories: 教程学习
# comments: true
tags: 博客
date: 2020-04-05 22:57:06 
photos: 
  - "https://images.wallpaperscraft.com/image/single/laptop_code_programming_212332_1280x720.jpg"
---

本文的原文在我的微信公众号，欢迎点击下面蓝色字体链接进入主页

[值得你阅读的Hexo个人博客搭建](https://mp.weixin.qq.com/s?__biz=MzIwNzUwOTY1Nw==&mid=2247484965&idx=1&sn=757e038fdc0eb968008ff73752b2456c&chksm=971007d3a0678ec5ae9fa1c1c871ff8cf15b67656f227327e3b507654a7cac28002f867ddb01&mpshare=1&scene=1&srcid=&sharer_sharetime=1571762807076&sharer_shareid=d33f5fb4b35f5c6867199f25b3752a3a&key=b97804376b1264af0c3ca188cf96e5b8d3b5990c40809c5c043e6da161476ef8c6e3cef3a9329dcc6d4eec6f605d83d617cd92d6f6883c7b9b632ca8aaf81c0f84c846463da87113881750250a56879b&ascene=1&uin=MjE2NzAwOTgyNA%3D%3D&devicetype=Windows+7&version=62060844&lang=zh_CN&pass_ticket=fLo4Au8Vos0w%2Fs3CVdzWInlkCBCduYag%2BmLa3dny9lkniGqUIl0wnHtlfaWbwpJJ)


![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS01ZDJjYWQyNDY1NmQ2ZWM2?x-oss-process=image/format,png)

**Hexo快速搭建个人博客（2019/10/22更新）**

* * *

**使用到的工具**  **（本教程统一在Windows系统下搭建）**

**Node.js、Hexo、Git、Github账号、Sublime Text3**

**请自行注册一个Github账号**

**最后的部署到网上的博客展示**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS0wNDNhNWFjNzIwZWNmNTBl?x-oss-process=image/format,png)

**文章目录：**

***前言***

***值得你知道的话***

***一、从创建到部署博客***

*****二、博客的网页主题*****

*******三、更换域名+博客测试成功*******
<!-- more -->
# 前言

今天一直在钻研这个博客，并搜索了大量有关hexo搭建博客的教程进行学习。我作为一个第一次使用Hexo搭建个人博客的菜鸟，**我发现我踩了不少坑**，哈哈，在这里我不得不吐槽一下某些撰写Hexo搭建博客的技术人员，用一个字来形容他们的博客就是**"乱"**，乱是因为我读完他们的博客写的内容发现逻辑顺序简直看得我**一头雾水**、细节内容对于他们来说就是一个摆设，难怪有好多人看不懂也是应该的。当然，可能是我的水平不够，也或许是在拜读他们的大作时候不够认真和严谨。

但我还是要告诫一下一些技术编辑者：

**如果是怕别人偷学你的内容，那就不要发在网上；****如果你发在网上，请考虑我们读者的感受，要对自己花费那么多时间撰写的内容负责，要****让别人看懂你的文章，让别人欣赏你的作品**。其实，有时候还能看出一个人的品性。

吐槽到此结束~**下面开始进入博客搭建环节**



* * *

**值得你知道到话：**

**是的，你没有看错！**

**不用服务器，不用注册域名，不用花钱，不用敲大代码等等**

**一个博客就值得你拥有**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1lOTg2YzEzZTc4ODRiYTVl?x-oss-process=image/format,png)

* * *

# 一、从创建到部署博客**

**1、安装好Node.js**

**别忘了用命令npm检验Node.js安装是否完成，**

**关于hexo的安装教程比较简单，**网上有很多完整的教程，在这里就不再赘述。****

**安装Hexo 命令：**

**npm install -g hexo-cli**
补充:安装hexo-helper-live2d 看门动画插件 npm install --save hexo-helper-live2d \
```
一定要在你博客目录的指定路径下（E:\hexo\KangChou）执行，否则在node_modules之中安装不了
npm uninstall hexo-helper-live2d  
npm install --save hexo-helper-live2d
npm install live2d-widget-model-hibiki 
npm install npm install --save live2d-widget-model-xxx来安装你喜欢的模型
参考：https://zhuanlan.zhihu.com/p/349278862
```
参考文献:
```
https://blog.csdn.net/qq_36239569/article/details/104104894
https://zhuanlan.zhihu.com/p/350654582 
解决办法：
首先
npm config set proxy null 代理置为空

运行npm cache clean --force清理缓存

然后尝试执行
npm config set registry http://registry.npmjs.org/
如果嫌安装依赖慢的话 可以使用国内淘宝镜像
npm config set registry https://registry.npm.taobao.org
```
国内镜像下载就是快：
![image](https://user-images.githubusercontent.com/36963108/163660165-9fbbd2f0-fbdd-40a5-ac7c-34a5cac56a45.png)


做完了这一步之后，恭喜你，前期的准备工作已经完成，环境这一步结束了。

**2、安装好Git**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1hNGU4NjkxNGQ3ZmI5ZTkx?x-oss-process=image/format,png)

**3、在C盘下创建hexo文件夹**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS00MzA1ZmEyMzVlMjI4NjNj?x-oss-process=image/format,png)

* * *

**4、打开Hexo文件夹下，右键点击Git bash 下执行命令**

工程文件目录：

![image](https://user-images.githubusercontent.com/36963108/163580819-8490257b-00cd-425d-9b9f-f2489aa1d3ae.png)


**再使用一次这个命令：npm install hexo-cli -g在终端使用npm安装hexo**

**创建博客KangChou：hexo init KangChou**
 
**cd KangChou**

**npm install**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS04NjYyNmNhODY3NzQ4NTFm?x-oss-process=image/format,png)

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1jYTFkZDI2NDNiMTc2M2Rm?x-oss-process=image/format,png)

* * *

**5、命令hexo server启动github服务器**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS0xZTk0ZDI2ODFlZGNiNGM3?x-oss-process=image/format,png)

* * *

**6、浏览博客**

安照5中提示的网址**http://localhost:4000/**

复制该网址在浏览器中打开，如下图所示：这样一个博客的架子就出来了。

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS00MTNhNWJiNmYzYTAyNmQ1?x-oss-process=image/format,png)

<iframe js_editor_cpcad="" class="js_cpc_area res_iframe cpc_iframe" src="https://mp.weixin.qq.com/cgi-bin/readtemplate?t=tmpl/cpc_tmpl#1571758997949" data-category_id_list="1|11|16|17|22|24|26|27|28|29|3|31|32|35|36|37|39|41|42|43|45|46|47|48|49|5|50|51|52|53|54|55|6|7|8" data-id="1571758997949" style="margin: 14px 0px; padding: 0px; border: 0px; width: 636.997px; height: 378px; z-index: 9999; position: relative; display: block;"></iframe>

* * *

**7、部署前哨（一）：添加Github仓库地址**

在部署之前，我们需要先知道博客的部署地址，它需要对应 GitHub 的一个 Repository 的地址，这个信息需要我们来配置一下。(这里我就省略了，自己去布置)，这是我的这个Github仓库

**https://github.com/KangChou/KangChou.github.io.git**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS0zN2EyODQ2NzBiOTIyOWU5?x-oss-process=image/format,png)

打开文件Hexo下的KangChou文件根目录下的 _config.yml 文件，我使用编辑器Sublime Text3打开的（或者你使用其他代码编辑器打开，**千万别用文本编辑器打开**），找到 Deployment 这个地方(提示：文件最后)，把刚才新建的 Repository 的地址贴过来

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1mMTNiODgyYjdjYWEyYWEx?x-oss-process=image/format,png)

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1lZDVlMDdlOTRhZjE3NGQ2?x-oss-process=image/format,png)

* * *

**8、部署前哨（二）：****部署插件**

需要安装一个支持 Git 的部署插件： **hexo-deployer-git**，有了它我们才可以顺利将其部署到 GitHub 上面(如果不安装的话，在执行部署命令时会报错).

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1kMzJhYmJmZWZlYjg5OTNj?x-oss-process=image/format,png)

* * *

**9、下面开始部署到Github**

如果8的插件部署没有问题就开始进行部署，首先输入部署命令如下：**hexo deploy**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1hNzUyMjBjZTQ3YWI3ZmJi?x-oss-process=image/format,png)

结尾....

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS00OTNlNmVhYzIzOGYzMDE1?x-oss-process=image/format,png)

可以发现出现了上面的报错提示：

```
Error: Spawn failed
```

**解决方法第一次：**

经搜索大量资料发现了下面的这个博客出现类似上述的一样问题，看了这个解决办法，我就斗胆试一试：

**https://blog.csdn.net/njc_sec/article/details/89021083**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1hYjhlZDY5ZTc0N2UzMDYz?x-oss-process=image/format,png)

可惜我再重新安装上述的三个命令输入之后还是出先一样的错误。


**解决方法第二次：**

因此我怀疑可能是仓库的地址出错，因此去看看了看地址

这是原来的地址：

```
deploy:
```

我按照出现错误提示中的网址去打开它：

 **https://hexo.io/docs/troubleshooting.html**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS04YzUxZDQ3Y2M3ZWYyNzcx?x-oss-process=image/format,png)

并找到了部署到Github目前的语法规定的网页下

**https://hexo.io/docs/github-pages**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1kNmIxMjllMTQxYTgxNDVk?x-oss-process=image/format,png)

按照上面对部署仓库的地址，我将上述的Deploy的源码修改为

```
deploy:
```

于是我再试了上述的三个命令：

**hexo clean**

**hexo g**

**hexo d**

最终出现下面的结果:说明出现的问题解决了

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1lN2Y3NTdkYTk5NGZjNzdh?x-oss-process=image/format,png)

由于我起初没有部署仓库的密钥，所以要去仓库部署。



* * *

**10、创建的 ssh 密钥的密码**

(1)、我打开了我得仓库，并找到了设置

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1iOGJmOTlmMzQxNjViYzVh?x-oss-process=image/format,png)

（2）查看部署密钥指南以了解更多信息

**https://developer.github.com/v3/guides/managing-deploy-keys/#deploy-keys**

**找到了设置密码的步骤（经过翻译以后，目前我们进行到下面的5）** 

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS04ZTdiODQxZmNmMDQyMTVk?x-oss-process=image/format,png)

（3）怎么创建SSH密钥?

步骤：

*   找到本地环境：C:\Users\admin\.ssh   这个路径下的用户/名称/.ssh

*   在这路径下，打开gitbub的命令控制台

*   (I): git  init  //初始化一下，看看路径对不对

       (II):ssh-keygen -t rsa -C "你的邮箱"

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1iMjgxYmNlMzI1ZDc0NWNm?x-oss-process=image/format,png)

*   到本地环境.ssh路径下查看，是否生成id_rsa,id_rsa.pub这个两个文件

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS0xMzE1MDViZmY5MzlkMGMz?x-oss-process=image/format,png)

*   生成后，现在把id_rsa.pub里面的内容复制到githubd 的add github key 的key里面（也就是刚刚仓库的密钥添加的地方）

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS0zYzgxZjM1Njc4OWQ3OWQ1?x-oss-process=image/format,png)

**点击Add SSH key获得下面结果**

**注意：第一次提交，配置密钥，需要输入github的密码，如下就是添加秘钥成功**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS04OWUyMDMzYmU5M2VkZjMy?x-oss-process=image/format,png)

*   密钥配置成功后，要验证一下是否配置成功

    命令：ssh -T git@github.com 

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1jYzZhMzM1ZjIxOGIzMzIw?x-oss-process=image/format,png)

出现下面提示，说明配置成功。
Hi KangChou/KangChou.github.io! You've successfully authenticated, but GitHub does not provide shell access



* * *

**11、再次使用密钥部署**

仍然使用命令：

**hexo clean**

**hexo g**

**hexo d**

如果都没有问题就会出现下面的结果，输入你刚刚设置的**名**和**SSH的密码**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1kZWY1M2FjNzQ3OGE5NWE0?x-oss-process=image/format,png)

但是这里又出错了

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1kMWJjMTc2ZWM0MmM2Y2Yw?x-oss-process=image/format,png)

追根溯源，我怀疑还是部署的仓库出现了问题：因此我再次打开，并做了修改

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS03NGQ5NWMyMGUyODRlMjNi?x-oss-process=image/format,png)

命令hexo d执行又出错

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1kOTJhODBmYjdjMGRhZWRl?x-oss-process=image/format,png)

下面终于找到了答案。这里要特别感谢**@李典金 @崔庆才**两位网络开发大佬的**细节**点拨才通过了上面的一个小环节，从而我力挽狂澜，一气呵成，搭建了后面的所有框架。

备注：ssh-keygen -t rsa -C "kangchou666@gmail.com"不用输入密码，回车就可以生成。如果需要严密一点，可以进行加密功能的部署。

![image](https://user-images.githubusercontent.com/36963108/163664932-10ba6060-bfd4-4538-8cbf-200ad8d34472.png)

* * *

**12、终于部署成功**

**最后终于找到了错误的原因，这是因为我创建的仓库下Github的SSH错误**

因此，我去仓库找到了

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1mM2VlNjk2YzRjYjg3Y2Qx?x-oss-process=image/format,png)

将红色的部分复制到hexo文件目录下也就是你的博客文件末尾，打开修改如下

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS00NDY0ZjhmNmE5M2Y0ODA5?x-oss-process=image/format,png)

**再次输入命令hexo d执行以后出现下面的结果，**

**出现Deploy done :git说明已经部署成功**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS05YzViOTc2Y2U3ZDc2NDk5?x-oss-process=image/format,png)

这时候我们访问一下 GitHub Repository 同名的链接，

比如我的 KangChou 博客的 Repository 名称取的是 KangChou.github.io，

那我就访问 **http://KangChou.github.io**

这时候我们就可以看到跟本地一模一样的博客内容了。

（此时你用手机同样可以打开该网站）

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS03MGVkMjY3OWY0MTk5YTUx?x-oss-process=image/format,png)

# 二、博客的网页主题**

主题的设置包括中文页面、整个页面的样式、页面风格等等，

目前 Hexo 里面应用最多的主题基本就是 Next 主题，

这个主题还是挺好看的，并且它支持的一些插件和功能都极为丰富，

配置了这个主题，我们的博客可以支持更多的扩展功能，比如阅览进度条、中英文空格排版、图片懒加载等等。

* * *

**13、下载主题**

打开我的电脑创建的Hexo文件夹下的KangChou目录，

单击右键Git bush输入下面的命令，执行结果如下：

**git clone https://github.com/theme-next/hexo-theme-next themes/next**

将下载后的themes主题替换原文件landscape中里所有的文件，并输入启动服务器命令

**hexo server**

执行结果如下

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS01MjE0ZTY1MGQ1NmI1MmU4?x-oss-process=image/format,png)

* * *

备注：使用数学公式需要安装这个工具：npm install hexo-math

**14、配置中文环境**

在博客kangchou目录下打开_config.yml修改语言为中文汉语**zh-Hans**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS05N2E4NzU3ZWRlODBkODgz?x-oss-process=image/format,png)

**执行的结果如下**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1lNzAxOTk4N2ExOTBlMzdi?x-oss-process=image/format,png)

**由于这只是部分为中文，而我的目的是大部分是中文的，**

**为了方便还要在网页上手动添加更多中文描述**

* * *

****15、配置中文菜单栏****

打开C:\Hexo\KangChou\themes\landscape\languages

**发有三种汉语:简体中文、香港繁体、台湾繁体**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1mOWVhYjFkNDNlNGM4OGIw?x-oss-process=image/format,png)

然后点开`zh-Hans.yml`其中的配置项就是已经翻译的文本

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1mZWIyNTVmNWUzMWM3MWM4?x-oss-process=image/format,png)

网站会根据你`站点配置``yml`中的语言配置来去读取对应的语言文件

打开你`languages``皮肤配置``yml`你会看到菜单栏基础配置：

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS05N2QyOGM1ZGE1MTRjZWQ4?x-oss-process=image/format,png)

**发现home和archives菜单是开启的，**

现在我们**全部开****启**，只需要去掉前面的#，刷新浏览器

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1lN2QyNzI2N2UzNTEzYzAw?x-oss-process=image/format,png)

**尝试修改站点配置yml语言，重启服务后刷新浏览器**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1iOThlM2UxMjI0NzdhMWYy?x-oss-process=image/format,png)

**显然结果很成功，****下面关闭git,将结果上传到Github页面：**

**重新打开输入部署的三个命令：**

**hexo clean** 

**hexo g** 

**hexo d**

**结果和上面一样，此时就可以访问了.**

**访问网站****https://kangchou.github.io/**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS02YTFlZmE2MjBlOTVlNTMw?x-oss-process=image/format,png)

* * *

**实际上文章到这里就已经结束博客的搭建了，至于其他的**

**比如上传文章、上传图片，添加logo等这些我这里就不说了，**

**hexo官网以及其他网站都能搜索到具体的教程，**

**想继续完善博客网站部署的朋友可以去搜索相关文献学习。**

# 三、更换域名**

**相信所有做互联网开发的科技工作者都知道，如果拥有属于自己的网站一定得看起来很专业、很官方、很大气。****因此，有些科技工作者就想更换自己网站的域名，让自己的域名看起来官方标准。也还有另一个原因，因为Github毕竟是外国网站，国内用户访问相对较慢，因此，如果有国内的域名作为辅助会事半功倍。****事实上，我个人觉得只要可以搭建网站，即便是不换域名也没什么区别。****不过，既然我给大家写这个教程，我还是有必要说一下，毕竟有很多人还是愿意换域名的。****如果不想花钱买域名的，这一小节可以跳过。********16、购买域名+注册阿里网+实名认证********自行注册，如果你是在校大学生，包括硕士、博士购买域名都是有学生价优惠的，但是****一定要使用自己在学校注册的电子邮箱****，因为阿里云官网数据库可以识别你的学生信息的学年期限。此外，注册以后一定要进行学生认证、实名认证。****![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1mM2M4ZTdkMTcyODNmNDNk?x-oss-process=image/format,png)** **然后去买域名，域名的形式有很多，按照自己的需求进行设置域名名称和域名后缀。（实名认证最快是两天的时间）****https://www.aliyun.com/**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1hMDI4MzE3ZDg0MmQzNGI4?x-oss-process=image/format,png)

******![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1iY2FmMDc1MjMyYWNkYmRk?x-oss-process=image/format,png)****** 

 ********17、在阿里云添加域名解析******

cmd+ping你的http://github.io域名，得到一个IP

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1kMThhMTFlNjhlMjRlMDI3?x-oss-process=image/format,png)

修改你的域名解析记录

**添加两个A记录，用得到的IP，一个主机记录为："www"，一个为"@"，**

**这样通过https://coomatrix.com/就能访问到你的博客了**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS05Y2QxZjFjMjc5NWM1YjM0?x-oss-process=image/format,png)

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1iZjYxNGFkZWQ3NTRhYTA3?x-oss-process=image/format,png)

******18、******填写绑定的域名在你的本地文件下也就是hexo—>你的博客（我的是KangChou）本地目录下找到 文件夹`source` ，并在该文件目录下面新建一个文件CNAME文件，那么一定要注意创建的CNAME文件**没有任何扩展名**（切记）

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS00NTQwYTE0ZTBmNjk1OGQ1?x-oss-process=image/format,png)

再一次使用部署三命令**hexo clean****hexo g****hexo d****完成以后，**进入Github设置，找到 Custom domain添加域名后保存即可

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS1lMTlhMDI3OTk3YTEyZTA3?x-oss-process=image/format,png)

******19、刷新网页+更改域名成功******

**如果上面的17没有出错的话，那么你填完域名保存以后会出现下面的结果**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS0zYTc5ZmFhYjc1ZmVlMjk5?x-oss-process=image/format,png)

那么就是更改域名成功了，此时你只需要点击上图的域名就可以访问啦。

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS03MmM2YmEyY2RjM2RkZGU2?x-oss-process=image/format,png)

**……到此完成了本博客的搭建……**

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS00ZWYyZWFhYzNiOWJjY2Q2?x-oss-process=image/format,png)



投稿--->展示你的才华

请发邮件到

**kangsinx@yeah.net**

标题注明【投稿】

告诉我们

你是谁，从哪来，投什么

我们会及时回复你

![image](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTg2MzE3MS04MDhkMTI5Y2VhMDM3MGVi?x-oss-process=image/format,png)


