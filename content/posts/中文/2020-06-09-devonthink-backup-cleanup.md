---
title: "DEVONthink 备份清扫"
date: 2020-06-09T22:40:29+10:00
tags: ["macOS"]
author: "Rui Qiu"
showToc: false
TocOpen: false
draft: false
hidemeta: false
comments: true
description: ""
# canonicalURL: "https://canonical.url/to/page"
disableHLJS: false # to disable highlightjs
disableShare: true
hideSummary: true
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
cover:
    image: "<image path/url>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
editPost:
    URL: "https://github.com/rexarski/blog/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

Devon 本意是英国的德文郡，由于此地的地层最早被研究，便用 Devonian 称呼在此处发现的特地地层。尔后翻译为中文便唤做「泥盆纪」。

<!--more-->

那么 DEVONthink 3 是我在 macOS 上常用的文档管理工具（虽然它能够发挥的作用远大于此），但自己其实并没有弄清楚它的备份与同步功能，尤其是和它自己的 iOS 版本 app 之间的关系更是一头雾水。偶然之间今天发现本地 iCloud Drive 文件夹虽然只有大约小几百 mb 的文件，但从 `~/Library/Mobile Documents` 路径访问则会看到令人发指的 30gb 文件大小。而且非常确定的是，这些文件既占用了本地硬盘，又占用了 iCloud Drive 存储空间，而且还并没有应有的作用。于是乎，本着救硬盘于水火的想法，想要把冗余的备份清扫干净。

首先要做的事情是，在 macOS 版本的 DEVONthink 中将原有的 sync location 反向勾选 iCloud 这一项。（本地备份还是比较明朗的，插一块外接硬盘，选择想要备份的 databases 就算完工了。而且备份文件可以直接通过 Finder 访问，要杀要剐任君使唤。）

![Sync Locations](/image/screenshot/devonthink.png)

然后要做的就是去 iCloud 中找到 DEVONthink 备份所占用的文件夹路径。此处有两种方法：

1. 第一种方法仅限用于用户历史以来只通过 iCloud 备份过一次数据的情况下。通过 iOS 的系统设置 Settings - Apple ID - iCloud - Manage Storage - DEVONthink To Go - Delete Data 就可以完成。但实际情况往往不是这样，在先前的无数次尝试中，或多或少我已经尝试设置过多次以 iCloud 为备份地址，那么实际上 iCloud 中就会在一个路径下存放有多次备份文件，这也就是为什么我只是备份了 Inbox 和另外一个小 database，但总的备份文件所占空间有 30gb。具体而言，就是先前我一定备份过更多的数据，而备份的残骸都留存在了 iCloud 的这个备份路径下。
2. 第二种方法，就是直接杀到 iCloud 里干掉整个备份文件路径。如果想直接通过 Finder 访问，你多半只是看到了 iCloud Drive 里零零星星的一些文件。要知道啊，iCloud Drive 只是 iCloud 的一个子集。所以核心问题就在于，我想去看存在 iCloud 里的 DEVONthink 备份路径，为什么你非把我往放文件的路径带呢？终端走起。

```bash
cd ~/Library/Mobile\ Documents
ls
rm -rf iCloud~679S2QUWR8~com~devon-technologies~sync
```

⚠️ **注意：** 在 `rm -rf` 之前还是 `ls` 确认一下这个备份路径确实存在，包含 `com-devon-technologies` 的关键词，然后用 `sudo` 权限清理干净。

此时，我不禁联想到，那我曾经尝试过的别的 sync locations 会不会也有同样的「隐藏备份文件」的问题呢？于是我如上图所示，重新添加了 Dropbox 作为备份地址。备份数据库我选择了只有三个文件的 Global Inbox，方便试错。有趣的是，在新建备份名称的时候 DEVONthink 居然给我自动联想了两个我曾经的备份名称！我本以为当时在 DEVONthink 的设置里删除掉这个备份路径，就自动把备份路径里的备份文件也删除了，结果并不是这样。我试图从 Dropbox 的桌面客户端访问到 Apps 文件夹，看一看 DEVONthink 是否真的有几个备份文件存在这里，结果发现这个文件夹作为与 Dropbox 关联的 Linked App，并没有出现在本地。切换到网页端，却真的发现了曾经的两个备份文件都静静地躺在 Dropbox 的存储空间里。果断选择了删除，于是 Dropbox 的空间也得到了释放。

所以，总结一下今天的发现：

**在 DEVONthink 里删除 sync locations 并不等同于删除已经创建的备份文件。如果需要删除备份文件，需要到指定备份地址的 explicit location 去强行删除这些备份文件；否则则会有很多空间被偷偷占用，根本也无从知晓。**
