library(ggplot2)
library(dplyr)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: plot_conflux.r <experiment directory>")
}

# Plot histogram with the number of contributions per reconstructed model
dat <- read.csv(paste(args[1], "contributions_per_reconstructed_model.csv", sep="/"))

ggplot(dat, aes(x = num_clients_in_model)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 1, fill = "skyblue", color = "black") +
  xlab("Clients in reconstructed model") +
  ylab("Fraction") +
  theme_bw()

ggsave(paste(args[1], "contributions_per_reconstructed_model.pdf", sep="/"), width = 6, height = 4)


# Plot the contributions of client, as function of their compute/network speed
dat <- read.csv(paste(args[1], "contributions.csv", sep="/"))

# Normalize network and compute speeds relative to the slowest node
dat$normalized_network_speed <- dat$network_speed / min(dat$network_speed)
dat$normalized_compute_speed <- dat$compute_speed / min(dat$compute_speed)
dat$combined_normalized_speed <- 0.5 * dat$normalized_compute_speed + 0.5 * dat$normalized_network_speed

client_stats <- dat %>%
  group_by(client) %>%
  summarise(avg_coverage = mean(coverage),
            sd_coverage = sd(coverage),
            mean_combined_speed = mean(combined_normalized_speed)) %>%
  arrange(mean_combined_speed) %>%
  mutate(client_order = row_number())

ggplot(client_stats, aes(x = client_order, y = avg_coverage)) +
  geom_point() +
  geom_errorbar(aes(ymin = avg_coverage - sd_coverage, ymax = avg_coverage + sd_coverage),
                width = 0.4, color = "gray50") +
  xlab("Client (sorted by increasing speed)") +
  ylab("Coverage") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))

ggsave(paste(args[1], "coverage_vs_normalized_speed.pdf", sep="/"), width = 10, height = 5)
