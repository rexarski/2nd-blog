---
title: "The lost R packages (after updating R)"
date: 2019-07-24T00:23:57+10:00
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

On Windows, we have a package called [`installr`](https://cran.r-project.org/web/packages/installr/index.html). Use function `copy.packages.between.libraries()`, then problem solved.

On macOS, unfortunately, we don’t have that handy tool.

But we can still use the following to retrieve all current installed packages’ names:

```r
to_install <- as.vector(installed.packages()[,1])
install.packages(to_install)
```

A more concrete [solution](https://www.r-bloggers.com/quick-way-of-installing-all-your-old-r-libraries-on-a-new-device/) would be only updating those non-base-R packages:

```r
installed <- as.data.frame(installed.packages())
write.csv(installed, 'installed_previously.csv')

installedPreviously <- read.csv('installed_previously.csv')
baseR <- as.data.frame(installed.packages())
toInstall <- setdiff(installedPreviously, baseR)

install.packages(toInstall[,1])
```

Still, I wish those old packages can be transferred to a new version of R painlessly.
