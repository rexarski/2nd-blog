---
title: "Pokémon recoloring"
date: 2019-04-17T23:32:22+10:00
tags: ["R", "viz"]
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

Inspired by Alex Onsager and his wacky web app [Pokemon Fusion](https://pokemon.alexonsager.net/), I was thinking about improving his approaches myself. In his post _[Pokemon Fusion: Behind the Scenes](https://www.alexonsager.net/2013/06/04/behind-the-scenes-pokemon-fusion.html)_, he explains how he makes the head-swapping. Basically this can be included in three steps:

![img](/image/viz/pokemon-1.png)

1. Manually prepare separate parts (a head and a body) of a Pokémon.
2. Determine the face size.
3. Determine the main palette of a Pokémon.
4. _Transplant_ the head to its new owner and unify the torso color.

> Introducing the King in the Soil: Dugking!

![Introducing the King in the Soil: Dugking!](/image/viz/pokemon-dugking.png)

The resizing part is definitely a highlight. But some weird combinations could be easily noticed due to the "manual separation", such as this Magikarp head on a Gyarados body:

![img](/image/viz/pokemon-4.png)

Nevertheless, it is some brilliant work done by Alex.

## Crawler

Alex mentions he plans only to support the first 151 Pokémons for his site. Understandable, because it's really hard to automate the head/torso separation. To be honest, I really wish there could be some magical _facical recognition_ algorithm for Pokémons.

The first thing is to scrape the sprites of Pokémons. My target website is [Pokémon Database](https://pokemondb.net/sprites) which has a pretty organized gallery of all 809 Pokémons from 7 generations.

Scraping with `rvest` is not the hard part. The only thing tricky worth mentioning is the set the sleeping time between batches. How do I know this? Learn from trials.

```r
t0 <- Sys.time()
# do things here
t1 <- Sys.time()
Sys.sleep(0.5*as.numeric(t1-t0))
```

> Sleeping time is proportional to response time.

## Color picker

What's next?

1. Load the images of Pokémon A and Pokémon B.
2. Use [**K-means**](https://en.wikipedia.org/wiki/K-means_clustering) to find the main palette of the two.
3. Simply swap the colors of two palettes. In fact, we are swapping two vectors of clustering assignments.
4. Reconstruct the Pokémon A with main palette of Pokémon B. (I forgot to make the other way around, sorry.)

At this point, I've realized that facial recognition with data so far is not practical since all images are only of size 30x40 pixels. In other words, the data is too limited.

Anyways, there are still something I want to point out during the four steps above.

- Although the scraped images are in PNG format with transparent background, once loaded with `load.image()`, the RGB of these transparent pixels will "turn to the darkside".
- I am pretty sure in most of the cases, the top two clusterings are the background "Mr Black" and some darkish grey boundary. You can alwasy check their sizes to confirm.
  - When swapping the palettes between two Pokémons, we can just ignore these two unnecessary colors.
- Due to the randomness in K-means (`km()`), the generated clusterings differ slightly from time to time.
  - The number of clusterings `k` I pick here is 7. Just do not set it too large.

---

Are [Red Gyarados](https://bulbapedia.bulbagarden.net/wiki/Red_Gyarados) extremely rare? _Maybe_.

But ladies and gentlemen, I present you the _Elite Four of Blue Magikarps_!

![*Elite Four of Blue Magikarps.*](/image/viz/pokemon-4xkarp.JPG)

And four mediocore red worms. Meh!

![*And four mediocore red worms. Meh!*](/image/viz/pokemon-4xgyara.JPG)

As I mentioned above, the head swapping process cannot be fulfilled with this scale of data. The least I can do is to implement the color swapping. You can find my codes on [GitHub](https://github.com/rexarski/pokeswapR). Have fun playing with it!
