library(rlang)
library(wesanderson)
library(dplyr)
library(tidyverse)
library(magrittr)
library(ggplot2)
library(gmodels)
library(tidyverse)
library(xtable)
library(tidyselect)
library(rJava)
library(ggpubr)
library(rgdal)
library(descr, quietly = TRUE)
library(pander)
library(kableExtra)
library(gmodels)
library(foreign)
library(summarytools)
library(knitr)
library(kableExtra)
library(grid)
library(gridExtra)
library(haven)
library(weights)
library(latex2exp)
library(magick)
library(webshot)
library(ggridges)
library(xlsx)
library(stargazer)
library(plm)
library(stringr)
library(gdata)
library(purrr)
library(foreign)
library(googledrive)
library(gdata)
library(ggrepel)
library(tidytext)


## Import data: 
eurbrm_data = read.csv('./Data_figure1/Eurobarometer.csv') %>% 
                filter(ï..Country %in% c("Greece", "Spain", "Portugal", "Ireland", "Cyprus", "Denmark", "Netherlands", "Latvia")) %>% 
                separate(Date, c("Day", "Month", "Year"), "-")  %>% 
                mutate(Country = ï..Country) %>% 
                group_by(Year, Country) %>% 
                mutate(mean_trust = mean(Tend.to.trust, na.rm=T)) %>% 
                filter(!duplicated(Year)) %>% ungroup()  %>% 
                mutate(Year = case_when(Year == "00" ~ "2000", 
                                        Year == "01" ~ "2001",
                                        Year == "02" ~ "2002",
                                        Year == "03" ~ "2003",
                                        Year == "04" ~ "2004",
                                        Year == "05" ~ "2005",
                                        Year == "06" ~ "2006",
                                        Year == "07" ~ "2007",
                                        Year == "08" ~ "2008",
                                        Year == "09" ~ "2009",
                                        Year == "10" ~ "2010",
                                        Year == "11" ~ "2011",
                                        Year == "12" ~ "2012",
                                        Year == "13" ~ "2013",
                                        Year == "14" ~ "2014",
                                        Year == "15" ~ "2015",
                                        Year == "16" ~ "2016",
                                        Year == "17" ~ "2017",
                                        Year == "18" ~ "2018",
                                        Year == "19" ~ "2019")) %>% 
                mutate(Year = as.numeric(Year)) %>% group_by(Country) %>%
                arrange(desc(-Year)) %>% ungroup() %>% 
                select(Country, Year, mean_trust)

write.csv(eurbrm_data, './Data_figure1/Eurobarometer_final.csv')
eurbrm_data_pigs = eurbrm_data %>% 
                   filter(Country %in% c("Greece", "Spain", "Ireland", "Cyprus", "Portugal"))

write.csv(eurbrm_data_pigs, './Data_figure1/Eurobarometer_final_pigs.csv')


