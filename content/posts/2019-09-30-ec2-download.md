---
title: "How to download files from AWS EC2"
date: 2019-09-30T23:38:21+10:00
tags: ["AWS"]
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
hideSummary: true
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
math: false
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

~~SCP~~ `scp` is always our friend.

```bash
scp -i /path/to/permission/file username@ec2.url:/path/to/remote/file /path/to/local/directory
```

**Update on 2019-10-14.** _What if I need to copy files that need root access?_

```bash
ssh -i /path/to/permission/file username@ec2.url "sudo cat /var/log/nginx/access.log" > ~/Downloads/access.log
```

![img](/image/screenshot/ec2-root-download.png)

Generally, if it is a log that could be printed out. Just print and save it to local.
