library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)
dat <- read.csv(paste(args[1], "bandwidth.csv", sep = ""))
dat$bandwidth <- dat$bandwidth / 1000 / 1000

p <- ggplot(dat, aes(x=bandwidth)) +
     stat_ecdf() +
     theme_bw() +
     xlab("Bandwidth [MB/s.]") +
     ylab("ECDF")

ggsave(paste(args[1], "ecdf_bandwidth.pdf", sep = ""), p, width=4.5, height=2.5)
