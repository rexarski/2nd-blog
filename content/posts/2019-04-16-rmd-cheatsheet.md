---
title: "What a Rmd cheatsheet doesn't tell me"
date: 2019-04-16T23:54:03+10:00
tags: ["R"]
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
hideSummary: true
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
math: true
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

**Updated on 2020-08-29:** I've migrated my blog from blogdown to hugo so this post might be outdated. See the [original post](https://github.com/rexarski/interesting-retrospectives/blob/master/content/blog/post/2019-04-16-what-a-rmd-cheatsheet-doesnt-tell-me.Rmd) for examples.

**Updated on 2019-09-08:** Today I realized the best approach to insert a centered image inside a html-supported markdown post is:

```html
<img class="special-img-class" style="width:100%" src="url-to-image" />
```

<img class="special-img-class" style="width:100%" src="https://media.giphy.com/media/d3mlE7uhX8KFgEmY/giphy.gif" />

---

Two common misunderstanding for longtime `blogdown` lurkers (like me) or R markdown newbies:

1. Blogging with `blogdown` is equivalent to writing reports with `knitr`. There is no need to "knit" every post you write, no matter which format (`.md` or `.Rmd`) you are using, since `blogdown` will handle them itself.

2. Deploying HTML inside a R markdown file can make the blog post more expressive than merely using Rmd syntax.

I didn't realize the second one until when I was browsing Nan Xiao's [blog](https://nanx.me/blog/), who is the author of this theme I'm using. At one point, I was wondering why his post looks so different from mine. Out of curiosity, I digged a little into his raw files and came up with these following "hacks".

### Insert an image with style

This has nothing to do with direct usage of HTML, but still is different from traditional markdown rendering. R markdown will not render the name/caption of an image automatically, but it's not the same case in `blogdown` here.

I used to like inserting an image with placeholder name like `img`, so the R markdown codes would look like this:

```
![img](path-to-this-image)
```

Somehow if this is the case, `blogdown` will render the image without any problem but leave you a weird image caption "img".

**Solution:** Either you ignore the `img` with a pair of blank square brackets,

```
![](path-to-this-image)
```

or you figure out a nice caption for the image and do the rest like before. And you can even insert a hyperlink inside the caption. For example:

```
![This is a [logo](https://www.r-project.org/Rlogo.png) of R project.](https://www.r-project.org/Rlogo.png)
```

**Update on 2019-04-17:** You might also want to _[center images](https://stackoverflow.com/questions/24677642/centering-image-and-text-in-r-markdown-for-a-pdf-report)_ in the post. This can be done with a pair of `<center></center>` tags.

But somehow the caption disappears. Try this:

```
<center>
  ![This is a [logo](https://www.r-project.org/Rlogo.png) of R
  project.](https://www.r-project.org/Rlogo.png)
  <p class="caption">"Logo here!"</p>
</center>
```

### Embed a YouTube video

~~Here's something R markdown cannot do. We need to borrow the power from HTML.~~

**Update on 2019-04-17:** According to this post "[Use shortcodes to embed tweets, videos, etc. in your blogdown blog posts ](http://www.reigo.eu/2018/04/shortcodes/)", one line code can solve the problem:

```
blogdown::shortcode("youtube", "jKp-WuK6iv8")
```

Note that in the code blog, we need to flag the argument `eval=TRUE`.

Additionally, I guess one line of HTML as caption `<p class="caption">"How could this happen?"</p>` will increase the "stylelishness" of an embedded media by all means.

```html
<p class="caption">"How could this happen?"</p>
```

### Pre-formatted table

R markdown tables look nice most of the time. But sometimes we need more. With html tag `<pre></pre>`, one can easily display a pre-formatted table.

```
<pre><output>
item1   item2   item3   item4
=============================
Gou     Li      Guo     Jia
-----------------------------
Qi      Yin     Huo     Fu
=============================
run_time: 59.0s
total_time: 60.0s
</output></pre>
```

A real example of [WYSIWYG](https://en.wikipedia.org/wiki/WYSIWYG).

### Code display

One way to do so is, of course, to write in a code block. Alternatively, we can attach an external link like this:

```html
<script src="https://gist.github.com/rexarski/92e120c3ffad70877716d99f6bfd66a3.js"></script>
```

### And finally this ...

I thought the shadow under some inserted images come from some extra HTML scripts. The fact is that it's a native feature in macOS.

- **Command + Shift + 4:** Take a screen shot with area selection, traditionally.
- **Command + Shift + 5:** Bring out the screenshot tool bar. If a screenshot is taken to capture the selected window, it will generate some shadow below.
