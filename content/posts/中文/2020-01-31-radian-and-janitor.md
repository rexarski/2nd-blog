---
title: "radian 与 janitor"
date: 2020-01-31T16:52:40+10:00
tags: ["R"]
author: "Rui Qiu"
showToc: false
TocOpen: false
draft: false
hidemeta: false
comments: true
description: ""
# canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
disableShare: true
hideSummary: false
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

两个好东西：

[radian](https://github.com/randy3k/radian). 号称是「A 21 century R console」.

<!--more-->

其实 21 世纪也都过了 20 年了，下个月 [R 4.0](https://blog.revolutionanalytics.com/2019/12/preview-of-r-400.html) 也将赶着 20 周年的日子发布。但这都不是重点。重点是每次为了运行一两行指令而打开「笨重」的 IDE 从来都不是我所喜欢的操作行为。之前用过 console 里的 R，但总觉得差点意思。如果说用一张图来概括 radian 的优点，那就是这样的：

![img](https://user-images.githubusercontent.com/1690993/30728530-b5e9eb5c-9f26-11e7-8453-73a2e880c9de.png)

另外一个是老朋友 [janitor package](https://garthtarr.github.io/meatR/janitor.html) 了，曾经在看 [R for Data Science](https://r4ds.had.co.nz/) 的时候用过一阵子，但是 package 这种东西你看到了不用就废了，或者用过一次之后隔一段时间不用，也等于废了。于是抽空在 radian 里把几个主要 functions 过了一下，觉得还是在处理「比较规整的、网上下载的数据」的时候比较方便。总结了一下功用：

- `clean_names()`. 转为小写+下划线的变量名格式，能够处理特殊符号。
- `tably()`. 输入 vector 输出频率和频数统计 table.
  - 另外如果含有 `NA` 则不会计入统计。
  - 可以用 `remove_empty(x, which = c("rows", “cols”))` 去除整行和（或）整列为 `NA`的数据。
  - `tably()` 后跟随多个 vectors 则输出 crosstabulation 结果。
- `get_dupes(x)`. 输入 data frame 和 变量名 `x`，输出重复的数据（行）。
- `excel_numeric_to_data()`. 将某些载入 Excel 文件时被转化的日期还原为 `Date` 类型。
