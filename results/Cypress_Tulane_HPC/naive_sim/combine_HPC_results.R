library(readr)
library(dplyr)

path_to_results <- "./"
result_files <- list.files(path_to_results)
result_files <- grep("_\\d+\\.csv$", result_files, value = TRUE)
result_files <- paste0(path_to_results, result_files)
num_files <- length(result_files)

if (num_files != 100) {
  stop("Simulation runs missing?")
}

results <- read_csv(result_files[1],
                    col_types = "_iicdciiiiddd") %>%
  mutate(replication = 1)
for (i in 2:num_files) {
  results_i <- read_csv(result_files[i], col_types = "_iicdciiiiddd") %>%
    mutate(replication = i)
  results <- bind_rows(results, results_i)
}
# sanity check:
#dim(results)
#31*6*5*num_files
#table(complete.cases(results))

source("../../../set_params.R")
if (!all(results$round <= p$n_adapt_rounds)) {
  warning("Mismatch with the number of adaptivity rounds!")
}

getwd()
write_csv(results, path = "../cypress_naive_all_results.csv")
