---
title: "What does top rated mean in Reddit’s ranking system"
date: 2019-03-02T22:56:07+10:00
tags: ["maths", "R"]
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

Though Reddit's comment sorting system has been in use for almost a decade, the fact is that I wasn't so exposed to western internet community till 2013, it is still, a rather new thing to me.

Randall, the author of [xkcd](https://xkcd.com), also my favourite internet comics, wrote a blog[^1] explaining what the algorithm is. Meanwhile, a detailed version is introduced by Evan Miller[^3].

[^1]: [reddit’s new comment sorting system](https://redditblog.com/2009/10/15/reddits-new-comment-sorting-system/).
[^3]: [How Not To Sort By Average Rating](http://www.evanmiller.org/how-not-to-sort-by-average-rating.html).

In the latter, Evan gave two examples of two "wrong" ranking methods:

1. Score = positive - negative;
2. Score = positive / total.

To say they are "wrong", does not mean that they are not possible to give a rough idea in all scenarios. But the reality is always complex, a rough idea is probably a synonym of "meaninglessness". To be specific, the first case ignore the "ratio" part in the term of "highest rated", that means a more controversial comment might exceed a quality post simply due to more people voting on it. The second case, however, ignores the scenario where the sample space is limited. For instance, we can hardly say that a 1 of 1 upvoted comment is better than that of 99 of 100 upvoted comment.

How Reddit deals with this problem is trying to reach a confident balance between positive proportion and small number of observations. Now suppose the following:

{{< math.inline >}}
<ol>
<li>Each voting event is independent.</li>
<li>Each event can either be a positive or a negative.</li>
<li>The total number of votes is \(n\), the number of positive votes is \(k\), \(\hat{p}=k/n\) is the positive proportion.</li>
</ol>

<p>
Now the idea is to first find each \(\hat{p}\), then calculate the corresponding confidence intervals, and finally rank the items by their lower bounds of confidence intervals.
</p>
{{</ math.inline >}}

The perfect expression of these confidence intervals here is:

$$ \left(\hat{p}+\frac{z^2_{\alpha/2}}{2n} \pm z_{\alpha/2}\sqrt{[\hat{p}(1-\hat{p})+z^2_{\alpha/2}/4n]/n}\right)/(1+z^2_{\alpha/2}/n) $$

In 1928, mathematician Edwin Bidwell Wilson[^2] developed this score interval above to estimate the successful (or positive, in our case) probability:

$$ \hat{p} $$

[^2]: [Binomial proportion confidence interval](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval).

Here's my R script to simulate a comment ranking situation possibly happening every day on Reddit.

```r
library(dplyr)

Wilson <- function(n, k, alpha=0.95) {
    phat <- k/n
    score <- qnorm(1-(1-alpha)/2)
    lbound <- (phat+score**2/(2*n)-score*sqrt((phat*(1-phat)+score**2/(4*n))/n))/(1+score**2/n)
    return(lbound)
}

seed(2019)
x <- sample(1:500,200,replace=T)
y <- sample(1:500,200,replace=T)

votes <- data.frame(x,y)
votes %>%
    rowwise() %>%
    mutate(pos=min(x,y),total=max(x,y)) %>%
    select(total, pos) %>%
    mutate(Wilson=Wilson(total,pos)) %>%
    arrange(desc(Wilson))
```

And the pseudo-results are:

```
# A tibble: 200 x 3
   total   pos Wilson
   <int> <int>  <dbl>
 1   473   467  0.973
 2   494   487  0.971
 3   390   385  0.970
 4   234   232  0.969
 5   354   347  0.960
 6   237   233  0.957
 7   123   121  0.943
 8   391   376  0.938
 9   296   285  0.935
10   415   397  0.932
# ... with 190 more rows
```
