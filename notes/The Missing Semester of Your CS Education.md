# 动机

- 帮助我们更高效的解决问题的工具
	- 现有工具、新的工具、探索与开发更多的工具

# 课程结构

- 11个一小时讲座

# 主题 1: The Shell

## shell 是什么？

- 操作界面（图形/语音）限制了操作方式。
	- 回到最根本的Shell
- 本节课使用Bourne Again SHell（“bash”）。

## 使用 shell

- 打开终端。
- `<主机名>:<当前工作目录>$`
- *命令*
	- `date`
- *参数*
	- `echo <参数>`
	- 处理空格：转义`"\ "`
- *环境变量*：`$PATH`
	- _可执行程序_

## 在shell中导航

- _绝对路径_与_相对路径_
	- `pwd`
	- `cd`
	- `.`
	- `..`
	- `ls`
		- `-h`
		- `-l`
	- `mv`
	- `cp`
	- `mkdir`

## 在程序间创建连接

- “流”
	- `> file`与`< file`
	- `>>`
	- 管道：`|`

## 一个功能全面又强大的工具

- 根用户（root user）
	- `sysfs`：内核参数
	- `cd /sys/class/backlight/thinkpad_screen`后：
		- `sudo echo 3 > brightness`是错误的。
		- `$ echo 3 | sudo tee brightness`是正确的。

## 课后练习

### 1.1

```shell
$ echo $SHELL
/bin/zsh
```

### 1.2

```shell
$ sudo mkdir /tmp/missing
```

### 1.3

```shell
$ man touch
```
### 1.4

```shell
$ sudo touch semester
```

### 1.5

```shell
$ echo '#!/bin/sh' | sudo tee semester
$ echo 'curl --head --silent https://missing.csail.mit.edu' | sudo tee -a semester
```

### 1.6

因为用户没有执行权限。

```shell
$ ls -l semester
-rw-r--r-- 1 root 61  1  9 20:58 semester
```

### 1.7

用`sh`执行的话，有读权限就可以。

### 1.8

man页面的描述是`chmod - change file mode bits`。

### 1.9
