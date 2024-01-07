library(ggplot2)

# Function to merge CSV files
merge_csv_files <- function(directory, pattern) {
  files <- list.files(path = directory, pattern = pattern, full.names = TRUE)

  # Read each file and create a list of data frames
  data_frames <- lapply(files, read.csv)

  # Combine all data frames into one
  combined_data_frame <- do.call(rbind, data_frames)

  # Return the combined data frame
  return(combined_data_frame)
}

# Plot broker resource usage
args <- commandArgs(trailingOnly = TRUE)
dat <- merge_csv_files(args[1], "^resources_broker_.*\\.csv")
dat$phys_mem_usage <- dat$phys_mem_usage / 1000 / 1000
dat$shared_mem_usage <- dat$shared_mem_usage / 1000 / 1000

# Items in the worker queues
p <- ggplot(dat, aes(x=time, y=queue_items, group=broker, color=broker)) +
     geom_line() +
     theme_bw() +
     xlab("Time into Experiment [s.]") +
     ylab("Queue Size") +
     theme(legend.position="bottom")
ggsave(paste(args[1], "queue_size.pdf"), p, width=4.5, height=3)

# Number of models in memory
p <- ggplot(dat, aes(x=time, y=num_models, group=broker, color=broker)) +
     geom_line() +
     theme_bw() +
     xlab("Time into Experiment [s.]") +
     ylab("Models in Memory") +
     theme(legend.position="bottom")
ggsave(paste(args[1], "models_in_mem.pdf"), p, width=4.5, height=3)

# CPU Usage
p <- ggplot(dat, aes(x=time, y=cpu_percent, group=broker, color=broker)) +
     geom_line() +
     theme_bw() +
     xlab("Time into Experiment [s.]") +
     ylab("CPU Utilization [%]") +
     theme(legend.position="bottom")
ggsave(paste(args[1], "cpu_usage_broker.pdf"), p, width=4.5, height=3)

# Physical Memory Usage
p <- ggplot(dat, aes(x=time, y=phys_mem_usage, group=broker, color=broker)) +
     geom_line() +
     theme_bw() +
     xlab("Time into Experiment [s.]") +
     ylab("Phys. Mem. Utilization [MB]") +
     theme(legend.position="bottom")
ggsave(paste(args[1], "phys_mem_usage.pdf"), p, width=4.5, height=3)

# Shared Memory Usage
p <- ggplot(dat, aes(x=time, y=shared_mem_usage, group=broker, color=broker)) +
     geom_line() +
     theme_bw() +
     xlab("Time into Experiment [s.]") +
     ylab("Shared Mem. Utilization [MB]") +
     theme(legend.position="bottom")
ggsave(paste(args[1], "shared_mem_usage.pdf"), p, width=4.5, height=3)


# Torch multiprocessing shared cache size
p <- ggplot(dat, aes(x=time, y=mp_torch_cache_items, group=broker, color=broker)) +
     geom_line() +
     theme_bw() +
     xlab("Time into Experiment [s.]") +
     ylab("Shared Cache Size") +
     theme(legend.position="bottom")
ggsave(paste(args[1], "mp_torch_cache_items.pdf"), p, width=4.5, height=3)


# Plot task statistics
dat <- merge_csv_files(args[1], "tasks_broker_.*\\.csv")
dat$func <- dat$`function`

# Total completion time for tasks, per broker
p <- ggplot(dat, aes(x=total_time, group=broker, color=broker)) +
     stat_ecdf(geom="step") +
     theme_bw() +
     xlab("Task Execution Time [s.]") +
     ylab("ECDF") +
     theme(legend.position="bottom")
ggsave(paste(args[1], "tasks_total_time_per_broker_ecdf.pdf"), p, width=4.5, height=3)

# Total completion time for tasks, per broker
p <- ggplot(dat, aes(x=total_time, group=func, color=func)) +
     stat_ecdf(geom="step") +
     theme_bw() +
     xlab("Task Execution Time [s.]") +
     ylab("ECDF") +
     theme(legend.position="bottom")
ggsave(paste(args[1], "tasks_total_time_per_function_ecdf.pdf"), p, width=4.5, height=3)


# Plot worker statistics
dat <- merge_csv_files(args[1], "worker_resources_broker_.*\\.csv")

# CPU Usage
p <- ggplot(dat, aes(x=time, y=cpu_percent, group=worker, color=worker)) +
     geom_line() +
     theme_bw() +
     xlab("Time into Experiment [s.]") +
     ylab("CPU Utilization [%]") +
     theme(legend.position="bottom")
ggsave(paste(args[1], "cpu_usage_workers.pdf"), p, width=4.5, height=3)