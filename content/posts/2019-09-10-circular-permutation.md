---
title: "Circular permutation"
date: 2019-09-10T23:10:51+10:00
tags: ["maths"]
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

A colleague had a coding interview for Huawei last Sunday. I heard the second question was quite “mathematical”. Let me rephrase it here a little bit.

> A hero summoner in a MOBA game has an ability to manipulate three elements. By controlling the order of releasing these elements, he can cast different spells accordingly. For example, casting in the order of _fire, water, lightening_ can be treated as a spell. But there are some limitations as well.

> Consider fitting the elements of that spell in a cycle. Then turning the cycle clockwise or counterclockwise does not produce any new spells. Additionally, inverting the cycle will not generate new ones either. The question is, if n is the number of elements he is capable of mastering, m is the number of elements consisting a spell, then what is the value of the number of different spells modulo 1000000007?

Typically, the mathematical term that describes the way of ordering elements, is called [Circular Permutation](http://mathworld.wolfram.com/CircularPermutation.html).

The number of ways to arrange `n` distinct objects along a fixed (i.e., cannot be picked up out of the plane and turned over) circle is

$$ P_n=(n-1)!$$

The reason why it is the factorial of `n-1` instead of $n$ is all cycle rotation.

![perm](https://mathworld.wolfram.com/images/eps-gif/CircularPermutations_950.gif)

If we consider a stricter definition, there will be only three free permutations (i.e., inequivalent when flipping the circle is allowed).

$$P’_n=\frac{1}{2} (n-1)!, n\geq 3$$

In our problem, the number would be

$${n \choose m} \frac{1}{2} (m-1)!$$

{{< math.inline >}}
<p>
Since \(1\leq m \leq 10000, 1\leq n \leq 20000\), direct calculation of factorial is suicidal for a computer. The hack here should be using modulo arithmetic, namely, we only keep the mod of \(10^9 + 7\) in intermediate steps.
</p>
{{</ math.inline >}}

```r
fact <- function(n) {
  res <- 1
  for (i in 1:n) {
    res <- (res * i) %% 1000000007
  }
  return(res)
}
```

Although `factorial(203)` will give us `Inf` as a result, `fact(203` won’t. It will give us an exact answer of `572421883`.
