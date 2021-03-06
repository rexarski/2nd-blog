---
title: "A sneak peek at Apple Watch Health data"
date: 2018-11-26T23:17:51+10:00
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

I've always been wondering if Apple could utilize its collected health data in a more elegant way. The Activity app is decent, especially the achievements inside are definitely highlights in an _iPhone x Apple Watch_ dynamic duo. On the other hand, the Health app serves nothing but a centralized information hub for various types data collected by either the sensor in your iPhone or your Apple Watch.

Well, that's basically my perspective and I don't have any constructive suggestions yet. But one thing that always interests me is that if I can play some magic around those hidden data. The answer is an absolute yes. The data export from iPhone is quite easy, and kudos to Apple. The problem is, the data comes in JSON format. [Ryan Praskievicz](http://www.ryanpraski.com/apple-health-data-how-to-export-analyze-visualize-guide/#2) provided a Python script to convert those extracted JSON to csv, but also he attached a link to an online converter [here](http://ericwolter.com/projects/health-export.html).

![](/image/viz/apple-health-to-csv.png)

I initially had the very first model of Apple Watch, aka the "Apple Watch before Series 1", aka "Series 0", which should be considered as a crappy smart watch even back in a scenario 4 years ago. The battery life could hardly power through one day, and launching a third-party app could literally take half a minute. So the only two functions I used were time keeping and activaity monitoring. Unfortunately, the watch encountered a serious this May and I still haven't figured out what happened to it. But in short, the watch stopped syncing with my iPhone and did not track any movements either. I had to perform a hard reset and thus lost all data prior to that day. So far, the data sitting on my MacBook is my Apple Watch health data collected from early this May to October.

The size of the data is around 90 mb and it wouldn't take too long to convert them to csv. But this really depends. I mean if you have a load of several years' data, I strongly suggest you make yourself a cup of tea during this time. I haven't checked the data structure inside at that moment, but a vague goal is to mimic the "major three" on Apple Watch: the activities, the exercise time and the standing hours. Surprisingly, the exported data has no standing hours, so that I have to use daily steps as a substitute.

```r
library(tidyverse)
library(lubridate)
library(gridExtra)
library(RColorBrewer)

read_one_field <- function(filename) {
    file <- read_delim(paste0("./", filename), delim=";", na = c("", "NA"))
    # glimpse(file)

    if (str_detect(file$type[1],
                   regex("ActiveEnergyBurned", ignore_case = TRUE))) {
        core_data <- file %>%
            select(creationDate, value) %>%  # unit: kcal
            mutate(creationDate = as_datetime(creationDate)) %>%
            rename(active = value) %>%
            group_by(day = date(creationDate)) %>%
            summarise(active = sum(active))
    } else if (str_detect(file$type[1],
                          regex("AppleExerciseTime", ignore_case = TRUE))) {
        core_data <- file %>%
            select(creationDate, value) %>%  # unit: min
            mutate(creationDate = as_datetime(creationDate)) %>%
            rename(exercise = value) %>%
            group_by(day = date(creationDate)) %>%
            summarise(exercise = sum(exercise))
    } else if (str_detect(file$type[1],
                          regex("StepCount", ignore_case = TRUE))) {
        core_data <- file %>%
            select(creationDate, value) %>%  # unit: count
            mutate(creationDate = as_datetime(creationDate)) %>%
            rename(step = value) %>%
            group_by(day = date(creationDate)) %>%
            summarise(step = sum(step))
    }

}

active <- read_one_field("/data/apple_health_export/HKQuantityTypeIdentifierActiveEnergyBurned.csv")
exercise <- read_one_field("/data/apple_health_export/HKQuantityTypeIdentifierAppleExerciseTime.csv")
step <- read_one_field("/data/apple_health_export/HKQuantityTypeIdentifierStepCount.csv")
```

The csv files are pretty neat overall, only a few NAs to deal with. Afterwards, I notice that each entry is responsible for an time elapse ranging from several seconds to minutes. Hence aggregation is unavoidable. The most natural way to do so is to group them by days.

```r
# standard: active = 400, exercise = 30, step = 10000

full_data <- active %>%
    full_join(exercise) %>%
    full_join(step) %>%
    replace_na(list(active = 0, exercise = 0, step = 0)) %>%
    mutate(active = active / 400,
           exercise = exercise / 30,
           step = step / 10000) %>%
    gather(key = type, value = value, -day)
```

Soon I come up with a line plot using a cutomized theme, which seems pretty similar to Apple's Activity color palette. The theme is based on a pure black `ggplot2` theme on [GitHub](https://gist.github.com/jslefche/eff85ef06b4705e6efbc).

```r
theme_black = function(base_size = 12, base_family = "") {

    theme_grey(base_size = base_size, base_family = base_family) %+replace%

        theme(
            # Specify axis options
            axis.line = element_blank(),
            axis.text.x = element_text(size = base_size*0.8, color = "white", lineheight = 0.9),
            axis.text.y = element_text(size = base_size*0.8, color = "white", lineheight = 0.9),
            axis.ticks = element_line(color = "white", size  =  0.2),
            axis.title.x = element_text(size = base_size, color = "white", margin = margin(0, 10, 0, 0)),
            axis.title.y = element_text(size = base_size, color = "white", angle = 90, margin = margin(0, 10, 0, 0)),
            axis.ticks.length = unit(0.3, "lines"),
            # Specify legend options
            legend.background = element_rect(color = NA, fill = "black"),
            legend.key = element_rect(color = "white",  fill = "black"),
            legend.key.size = unit(1.2, "lines"),
            legend.key.height = NULL,
            legend.key.width = NULL,
            legend.text = element_text(size = base_size*0.8, color = "white"),
            legend.title = element_text(size = base_size*0.8, face = "bold", hjust = 0, color = "white"),
            legend.position = "right",
            legend.text.align = NULL,
            legend.title.align = NULL,
            legend.direction = "vertical",
            legend.box = NULL,
            # Specify panel options
            panel.background = element_rect(fill = "black", color  =  NA),
            panel.border = element_rect(fill = NA, color = "white"),
            panel.grid.major = element_line(color = "grey35"),
            panel.grid.minor = element_line(color = "grey20"),
            panel.margin = unit(0.5, "lines"),
            # Specify facetting options
            strip.background = element_rect(fill = "grey30", color = "grey10"),
            strip.text.x = element_text(size = base_size*0.8, color = "white"),
            strip.text.y = element_text(size = base_size*0.8, color = "white",angle = -90),
            # Specify plot options
            plot.background = element_rect(color = "black", fill = "black"),
            plot.title = element_text(size = base_size*1.2, color = "white"),
            plot.margin = unit(c(0.35, 0.2, 0.3, 0.35), "cm")
        )
}
```

```r
ggplot(full_data, aes(x=day, y=value)) +
    geom_line(aes(color=type),size=1) +
    scale_color_manual(labels = c("Active Energy", "Exercise Time", "Steps"),
                       values=c("#FF0054", "#8CFF00", "#00FFCA")) +
    theme_black() +
    facet_grid(type ~ .) +
    labs(title="Apple Watch Health Data",
         subtitle="what to write here",
         caption="Source: my retired Apple Watch (1st gen)",
         y="Complete 100%",
         color=NULL)
```

![](/image/viz/apple-health-1.png)

But I'm not satisfied with theme somehow, so I decide to experiment a little bit on a [FiveThirtyEight-style theme](https://github.com/jrnold/ggthemes/blob/master/R/fivethirtyeight.R). It is a more furnished theme, but does not match the Activity app palette. Well, I just cannot reject the magic of minimalism.

```r
fte_theme <- function() {

    # Generate the colors for the chart procedurally with RColorBrewer

    palette <- brewer.pal("Greys", n=9)
    color.background = palette[2]
    color.grid.major = palette[3]
    color.axis.text = palette[6]
    color.axis.title = palette[7]
    color.title = palette[9]

    # Begin construction of chart

    theme_bw(base_size=9) +

        # Set the entire chart region to a light gray color

        theme(panel.background=element_rect(fill=color.background, color=color.background)) +
        theme(plot.background=element_rect(fill=color.background, color=color.background)) +
        theme(panel.border=element_rect(color=color.background)) +

        # Format the grid

        theme(panel.grid.major=element_line(color=color.grid.major,size=.25)) +
        theme(panel.grid.minor=element_blank()) +
        theme(axis.ticks=element_blank()) +

        # Format the legend, but hide by default

        theme(legend.position="none") +
        theme(legend.background = element_rect(fill=color.background)) +
        theme(legend.text = element_text(size=7,color=color.axis.title)) +

        # Set title and axis labels, and format these and tick marks

        theme(plot.title=element_text(color=color.title, size=10, vjust=1.25)) +
        theme(axis.text.x=element_text(size=7,color=color.axis.text)) +
        theme(axis.text.y=element_text(size=7,color=color.axis.text)) +
        theme(axis.title.x=element_text(size=8,color=color.axis.title, vjust=0)) +
        theme(axis.title.y=element_text(size=8,color=color.axis.title, vjust=1.25)) +

        # Plot margins

        theme(plot.margin = unit(c(0.35, 0.2, 0.3, 0.35), "cm"))
}
```

```r
ggplot(full_data, aes(x=day, y=value)) +
    geom_line(aes(color=type),size=1) +
    fte_theme() +
    facet_grid(type ~ .) +
    labs(title="Apple Watch Health Data",
         subtitle="what to write here",
         caption="Source: my retired Apple Watch (1st gen)",
         y="Complete 100%",
         color=NULL)
```

![](/image/viz/apple-health-2.png)

But even now I'm not so sure about what I can really do with these data, except for probably plotting them with lines and dots. Meanwhile, I'm not okay with the variable on y-axis as well. I tried to combine them in one plot at the very beginning, but it turned out that the three types of data were not on the same scale. So I made them the percentages of the completeness of daily goals by setting activity to 400, exercise time to 30 minutes and steps to 10,000. But still, the y-axis should better be "the ratio to the goal" instead of a bunch of misleading integers (1%? 2%?)

Well, at least I can conclude that all these match the title which says "a sneak peek", not a thorough digging.
