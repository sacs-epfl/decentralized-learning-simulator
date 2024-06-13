library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)
dat <- read.csv(paste(args[1], "round_duration.csv", sep = ""))
dat$time <- dat$time / 1000000

p <- ggplot(dat, aes(x=time)) +
     stat_ecdf() +
     theme_bw() +
     xlab("Round duration [s.]") +
     ylab("ECDF")

ggsave(paste(args[1], "ecdf_round_duration.pdf", sep = ""), p, width=4.5, height=2.5)
