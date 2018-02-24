---
title: Ubuntu 备忘录
layout: post
img: ubuntu.jpg
tags: [Linux, 备忘]
---

在 Ubuntu 中安装软件主要有两种方式：本地安装和在线安装。本地安装需要提前将软件包下载到工作目录，然后执行：

`sudo dpkg -i <software.deb>`

然而如果系统中没有安装该软件包所需要的依赖，安装会失败。此时需要修复依赖关系：

`sudo apt-get install -f`

然后再执行安装命令即可。

以上命令需要 sudo 权限，并且软件会安装到系统目录，所有使用该计算机的用户都能使用软件。若没有计算机权限，可以使用以下命令将软件安装的自己的目录下：

```shell
apt-get source <package-name>	# 获取软件源
cd <package-name>
./configure --prefix=$HOME	# 设置软件安装目录，若没有配置文件可直接修改 Makefile
make
make install
```

有时由于网络不稳定，国内用户访问软件源的速度极慢，这时可以考虑用国内的软件源替换官方软件源：

```shell
sudo mv /etc/apt/sources.list /etc/apt/sources.list.backup		# 备份官方软件源
sudo mv sources.list /etc/apt/		# 更换其他软件源
sudo apt-get update		# 更新软件源
```

替换用的软件源要自行下载，国内很多高校机构和互联网公司都有自己的软件源，例如[中科大软件源](https://mirrors.ustc.edu.cn/repogen/)，需要时可自行搜索。

有些软件没有在系统软件源中收录，但开发者提供了第三方软件源，可以添加到系统软件源中，例如 Typora：

````shell
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA300B7755AFCFAE
sudo add-apt-repository 'deb https://typora.io linux/'
sudo apt-get update
sudo apt-get install typora
````