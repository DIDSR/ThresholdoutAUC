library(readr)
library(dplyr)

path_to_results <- "./glmnet_random/"
result_files <- list.files(path_to_results)
result_files <- grep("_\\d+\\.csv$", result_files, value = TRUE)
result_files <- paste0(path_to_results, result_files)

path_to_results <- "./xgb_random/"
tmp <- list.files(path_to_results)
tmp <- grep("_\\d+\\.csv$", tmp, value = TRUE)
tmp <- paste0(path_to_results, tmp)
result_files <- c(result_files, tmp)

num_files <- length(result_files)

if (num_files != 200) {
  stop("Simulation runs missing?")
}

results <- read_csv(result_files[1],
                    col_types = "_iicdciiiidddc") %>%
  mutate(replication = 1)
for (i in 2:num_files) {
  results_i <- read_csv(result_files[i], col_types = "_iicdciiiidddc") %>%
    mutate(replication = i)
  results <- bind_rows(results, results_i)
}

source("../../set_params_RSNA-ICH.R")
if (!all(results$round <= p$n_adapt_rounds)) {
  warning("Mismatch with the number of adaptivity rounds!")
}

getwd()
write_csv(results, path = "./naive_random_all_results.csv")
