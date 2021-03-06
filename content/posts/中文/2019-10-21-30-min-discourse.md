---
title: "30 分钟 Discourse 论坛搭建笔记"
date: 2019-10-21T22:42:57+10:00
tags: ["self-host"]
author: "Rui Qiu"
showToc: true
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

如果顺利的话，30 分钟应该够。实际上我大概花了 2 个小时，主要周旋于各家邮件服务提供商之间。

<!--more-->

---

最近发现很多产品的售后/社群都在用一个「神秘」的 bbs 框架，一直不知道其尊姓大名。比如[Agenda](https://agenda.community), [Airtable](https://community.airtable.com), 还有某些神秘的小论坛 😅. 多番寻访在 GitHub 找到了这个名叫 Discourse 的开源项目——[GitHub - discourse/discourse: A platform for community discussion. Free, open, simple.](https://github.com/discourse/discourse)

因为项目需要，我打算现在自己的 1GB RAM 小服务器上先尝试一下。正如 Discourse 团队在他们的帮助文档中所言，想要成功部署一个最小需求的网站需要三样东西：

1. 一台「刚刚好够用」的服务器。
2. 一个「你觉得 👌 就 👌」的便宜域名。
3. 一个「在免费和收费边缘徘徊」的邮件发送服务商。

服务器我使用的是 [Vultr](https://www.vultr.com/products/cloud-compute/#pricing) 的 1GB RAM, 25GB SSD, 收费为 \$5/month. 域名则是在 [GoDaddy](https://au.godaddy.com) 购买的。吐槽一下 GoDaddy 的 DNS 设置，相比于 AWS 操作简单不少，但引导其实有点模糊。邮件发送服务商我本想启用已经使用了一段时间的 [Mandrill](https://mandrillapp.com)，但是发现它在我把 Discourse 设置在 subdomain 上时总是行不通。

官方的[帮助文档](https://github.com/discourse/discourse/blob/master/docs/INSTALL-email.md)在邮件的问题上推荐了不少服务提供商，而根据我个人的摸索，最终选择的是 [SendGrid](https://app.sendgrid.com)。Free tier 在调试阶段基本够用，当然正式上线肯定是不行的。具体的[流量限制](https://sendgrid.com/pricing/))为每天 100 封免费。

服务器的部署按照推荐走 Ubuntu LTS 版本就可以了 ，不用做过多的调试。之后 `ssh` 登录，并 `git clone` 整个 Discourse 项目到 `/var/discourse` 路径下。

然后是一系列点点点点乱操作：

- 在 SendGrid 中新建一个 api key.
- 在 SendGrid 中 Settings -- Sender Authentication — Domain Authentication. 验证自己是邮件发送域名的真实拥有者，这一步就按着提示走。
- 然后转到 GoDaddy 的 DNS Settings 新增三条 CNAME Record. 这里有一个暗坑：SendGrid 会提示你这三条 records 的 host 是 `something.example.com` 但是你不能直接复制粘贴到 GoDaddy 中，而只能粘贴 `something` 作为 host. 而 record 的 value 则直接放心粘贴。
- 然后在 GoDaddy 中创一个 A Record, host 为你想设置的 subdomain name，比如 `talk`, `discourse`, `forum`，仍然不要带 `example.com`. value 则填入你的服务器 public ip 地址。

回到已经 ssh 连接的服务器：

```bash
cd /var/discourse
./discourse-setup
```

过程中碰到一个报错，返回 `No Public Key Error`. 参考 [Stack Overflow](https://stackoverflow.com/questions/10255082/installing-r-from-cran-ubuntu-repository-no-public-key-error) 的解答，少什么 KEYID 就使用以下指令添加就可以 了：

```bash
sudo apt-key adv —keyserver keyserver.ubuntu.com —recv-keys KEYID
```

部署过程中有多次需要填写的信息：

```
Hostname		:	subdomain.example.com 推荐使用注册注册域名的某个 subdomain
Email			:  	me@anything.com 随便挑选一个自己在用的邮箱地址作为 admin 的注册邮箱
SMTP address	:	smtp.sendgrid.net 是 email service provider 的具体设置
SMTP port		: 	587 同上
SMTP username	:	apikey	同上
SMTP password :	*****	生成 api 的时候一般会给出
```

大约 3 到 5 分钟的安装时间，然后就可以直接访问 `subdomain.example.com` 来查看是否安装成功。一般来说安装的问题不大，下一步验证 admin 邮箱才比较容易出错，如果够幸运，之前没有踩到暗坑一蹴而就，你的 `me@anything.com` 邮箱里则会多出一封邮件（注意查 spam 📮）。验证完成会看到欢迎页面。

大功告成。

如果中间碰到了什么问题，直接在服务器上（`/var/discourse`) 重跑一遍 `./discourse-setup` 指令就可以了。困难主要还是集中在三项服务的配置。

另外指出一点，虽然在我安装过程中帮助不大，但 Discourse 自己的官方论坛也有不少可以参考的资源，比 repo 里 issue 有价值多了。地址是 [`meta.discourse.org`](https://meta.discourse.org).
