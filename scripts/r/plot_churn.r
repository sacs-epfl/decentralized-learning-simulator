library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: plot_churn.r <experiment directory>")
}

dat <- read.csv(paste(args[1], "churn.csv", sep="/"))
dat$time <- dat$time / 3600

p <- ggplot(dat, aes(x=time, y=active_clients)) +
     geom_step() +
     theme_bw() +
     xlab("Time into Experiment [h.]") +
     ylab("Online Clients")

ggsave(paste(args[1], "churn.pdf", sep="/"), p, width=5, height=3)
