---
title: "Shiny + scrollytell can tell a story"
date: 2022-01-14T15:18:00-05:00
tags: ["viz", "shiny"]
author: "Rui Qiu"
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: true
description: "Learn scrollytelling by a tutorial."
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
    image: "<image cover>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
editPost:
    URL: "https://github.com/rexarski/blog/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

My understanding on storytelling in the format of a “scrollytelling” is telling a story with graphs that are self-explainable so that the reader could spare some attention to what you are trying to say in words.

Yesterday I came across with an article one Datawrapper, _[Three decades of European government leaders](https://blog.datawrapper.de/longest-terms-european-leaders/)_. The article features a scatterplot with custom lines indicating European government leaders in a time span of roughly 40 years.

I’m not sure if there is specific name for this type of graph, it does look like hundreds of space fighters taking off at the same time. Out of curiosity, I decided to recreate this chart, but without the lines, in a scrollytelling fashion.

I followed Connor Rothschild’s [tutorial](https://www.connorrothschild.com/post/automation-scrollytell), and here’s the result[^1]:

{{< giphy JzJVXHdBsg77nHbXkg >}}

Also accessible at **[shinyapps.io](https://rexarski.shinyapps.io/shiny-scrollytell/)**.

Have I mentioned that "storytelling" would be the theme of my year? I don't know where to start but I feel like it's about time. Realistic speaking, all my courses this semester could be presented with a nicely tinkered interactive story.

I also summoned a long-gone friend [rexarski.com](https://rexarski.com) to be the playground where I can experiment some interactive visualizations.

Of course, I need to **master** Shiny too.

Second thought: isn’t the original plot a [rotated lollipop chart](https://www.r-graph-gallery.com/303-lollipop-plot-with-2-values.html)?

[^1]: Note that some records’ genders are not specified in the raw data, so I brutally categorized them as “male”. I also filled the “story” with placeholder contents.
