library(caret)
library(tidyverse)
library(ROCR)
library(tictoc)
library(xgboost)

# for HPC:
setwd("/home/alexej.gossmann/ThresholdoutAUC/RSNA_ICH_data_example/thresholdout_simulation")

#--- load functions

# functions to calculate p-values
source("../../functions/p-values.R")
# a function to fit classification models for every subset of the most significant predictors
source("../../functions/fit_model_for_every_subset.R")
# set simulation parameters (such as number of cases, number of features, etc.)
source("../set_params_RSNA-ICH.R")
# functions to run the main simulation one time for a specified method
source("general_simulation.R")
# thresholdout algorithm
source("../../functions/thresholdout_auc.R")

#--- constants

num_cpus <- as.numeric(Sys.getenv("NSLOTS"))

task_id <- Sys.getenv("SGE_TASK_ID")
print(paste("SGE_TASK_ID:", task_id))
iter_i <- as.integer(task_id)

ran <- 2020 + iter_i
print(paste("Random seed for this instance:", ran))

out_path <- "./results/xgb_random/"

#--- run the simulation

set.seed(ran)
tic("xgboost")
tryCatch({
  results <- run_sim("xgbTree", iter_i, p,
                     RData_save_path = paste0(out_path, "thresholdout_", iter_i, ".RData"),
                     nthread = num_cpus)
  write.csv(file = paste0(out_path, "thresholdout_", iter_i, ".csv"), x = results)
}, error = function(e) {
  print(paste("I caught an error!\n", e, "\n",
              "Random seed =", ran, "\n",
              "Classifier: xgboost"))
})
toc()
