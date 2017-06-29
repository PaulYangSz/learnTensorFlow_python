# learnTensorFlow_python
Just a start of TensorFlow

## 首先是TensorFlow的安装
参考[官网的安装链接](https://www.tensorflow.org/install/)
我选择的是在windows下安装预编译好的TensorFlow，而且是通过Anaconda框架。
具体过程按照官网的一步步来就可以了，当然安装过程中要翻墙下载。
其中关于Anaconda和Conda的介绍可以参考这个链接：[Anaconda使用总结](http://python.jobbole.com/86236/)

## 在Pycharm中创建TensorFlow的工程
可以看到，前面的步骤主要是在conda中搭建了一个tensorflow的环境，那么在Pycharm中也把这个搭建好的环境加入进去即可。
**Setting -> Project Interpreter**界面上点击右边的齿轮，选择**add local**，然后添加：*C:\ProgramData\Anaconda2\envs\tensorflow*下面的python.exe

至此，Pycharm的TensorFlo环境搭建就完成了。然后就可以**import tensorflow as tf**啦

