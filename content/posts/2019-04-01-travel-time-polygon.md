---
title: "Travel time polygon with osrm"
date: 2019-04-01T23:26:51+10:00
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

OpenStreetMap's [OSRM (Open Source Routing Machine)](http://project-osrm.org/) provides not only routing service, but a feature that enables drawing multiple routes to form a time-based polygon range as well. This is also called an isochrone.

An isochrone, according to Wikipedia, is defined as "a line drawn on a map connecting points at which something occurs or arrives at the same time". In other words, it is just like contour lines measuring travel time in urban design.

[Timothée Giraud](https://github.com/rCarto) has a repo [osrm](https://github.com/rCarto/osrm) that connects OpenStreetMap's service and R.

In the README file of this repo, there is also a vanilla example of how to draw `osrmIsochrone`.

```r
library(osrm)
library(sp)
library(cartography)
library(leaflet)

# Get isochones with a SpatialPointsDataFrame, custom breaks
iso <- osrmIsochrone(loc = c(149.1212457, -35.2766747), breaks = seq(from = 0,to = 30, by = 5))

# Map
osm <- getTiles(x = iso, crop = FALSE, type = "osm", zoom = 13)
tilesLayer(x = osm)
bks <- sort(c(unique(iso$min), max(iso$max)))
cols <- paste0(carto.pal("turquoise.pal", n1 = length(bks)-1), 80)
choroLayer(spdf = iso,
           var = "center", breaks = bks,
           border = NA, col = cols,
           legend.pos = "topleft",legend.frame = TRUE,
           legend.title.txt = "Isochrones\n(min)",
           add = TRUE)
plot(apotheke.sp[10,], add=TRUE, col ="red", pch = 20)
```

![img](/image/viz/canberra-osrm.png)

However, the processing time of this plot is not very satisfying. What about drawing the polygons with `leaflet`?

```r
# Map as leaflet
iso@data$drive_times <- factor(paste(iso@data$min, "to", iso@data$max, "min"))
factpal <- colorFactor("RdYlBu", iso@data$drive_times)

leaflet() %>%
    setView(149.1212457, -35.2766747, zoom = 11) %>%
    addProviderTiles("CartoDB.Positron", group="Greyscale") %>%
    addMarkers(lng = 149.1212457, lat = -35.2766747, popup = "Starting Point") %>%
    addPolygons(fill=TRUE, stroke=TRUE, color = "black",
                fillColor = ~factpal(iso@data$drive_times),
                weight=0.5, fillOpacity=0.2,
                data = iso, popup = iso@data$drive_times,
                group = "Drive Time") %>%
    addLegend("bottomright", pal = factpal, values = iso@data$drive_time,   title = "Drive Time")
```

![img](/image/viz/canberra-leaflet.png)

An interactive map is not a bad idea. And can use `proc.time()` to track the processing time of both methods.

```r
ptm <- proc.time()
# plotting method
proc.time() - ptm
```

Compare the processing of two plotting above. `osrm` takes around 50 seconds, while `leaflet()` only requires less than 7 seconds.

```
   user  system elapsed
 28.081  13.457  48.232

   user  system elapsed
  5.692   0.173   6.532
```
