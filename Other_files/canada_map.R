library(tidyverse)
library(ggmap)
setwd("~/GitHub/rest_recs/Other_files")
dta <- read_csv("~/Data/project121/c_business.csv") %>%
  select(city, state, latitude, longitude, review_count) %>%
  filter(review_count >50)
on_map <- get_map(location = "toronto", maptype = "roadmap",
                  source = "google", zoom=9)
ggmap(on_map) + geom_jitter(aes(longitude, latitude), data=dta, color="red", alpha=.1) +
  theme_void()
mt_map <- get_map(location = "montreal", maptype = "roadmap",
                  source = "google", zoom=11)
ggmap(mt_map) + geom_jitter(aes(longitude, latitude), data=dta, color="red", alpha=.5) +
  theme_void()
