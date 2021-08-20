---
title: "COVID confirmed cases Shortcut"
date: 2020-06-14T23:45:37+10:00
tags: ["shortcut"]
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

This post is a subsequent blog to my daily experience basically in the last two months. As the general situation of the COVID-19 pandemic in Australian Capital Territory keeps is keeping stable, sometimes I almost forget to check on this Shortcut.

Long story short, general idea is if you have very specific information in need, you can always create a temporary iOS Shortcut to assist you in capturing such information. In my case, it is the number of confirmed cases in ACT. Very specific. Therefore, all I need to do is to go to the government’s website, find the page containing that value, and scrape it in HTML. And BOOM, you get it out of there.

I’ll specify the steps in the context of Shortcut:

1. Create an action called “URL”. Insert the value as `https://www.health.act.gov.au/about-our-health-system/novel-coronavirus-covid-19`.
2. Create an action called “Get Contents of URL” with method `GET`.
3. Create an action called “Make rich text from HTML”. Replace `HTML` with `Contents of URL`.
4. Create an action called “Match Text”. Replace the two placeholders with `(Confirmed cases \d+)` and `Rich Text from HTML` respectively.
5. Create one last action called “Show results”. And change the last variable to `Matches`.

---

[iCloud link](https://www.icloud.com/shortcuts/4a686cfbcb38406f8228a5e805567bb5) to this Shortcut.
