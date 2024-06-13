library(ggplot2)
library(dplyr)

args <- commandArgs(trailingOnly = TRUE)
bandwidth_data <- read.csv(paste(args[1], "bandwidth.csv", sep=""))
bandwidth_data$bandwidth <- bandwidth_data$bandwidth / 1000 / 1000
round_data <- read.csv(paste(args[1], "round_duration.csv", sep=""))
round_data$time <- round_data$time / 1000000

merged_data <- inner_join(bandwidth_data, round_data, by = "client")

p <- ggplot(merged_data, aes(x=bandwidth, y=time)) +
     geom_point() +
     theme_bw() +
     xlab("Bandwidth [MB/s.]") +
     ylab("Round Duration [s.]")

ggsave(paste(args[1], "bandwidth_vs_round_duration.pdf", sep=""), p, width=4.5, height=3)
