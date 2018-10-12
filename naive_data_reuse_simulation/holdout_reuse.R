library(caret)
library(plyr)
library(dplyr)
library(tidyr)
library(ROCR)
library(tictoc)

#--- load functions

# functions to calculate p-values
source("../functions/p-values.R")
# functions to generate datasets
source("../functions/data_generation.R")
# a function to fit classification models for every subset of the most significant predictors
source("../functions/fit_model_for_every_subset.R")
# set simulation parameters (such as number of cases, number of features, etc.)
source("../set_params.R")
# functions to run the main simulation one time for a specified method
source("general_simulation.R")

#--- run the simulation

# use a unique random seed for each repetition in the array of jobs
array_task_id  <- Sys.getenv("SLURM_ARRAY_TASK_ID")
print(paste("SLURM_ARRAY_TASK_ID:", array_task_id))
ran <- as.integer(array_task_id)
print(paste("Random seed for this instance:", ran))

#--- prepare a function to run simulations in parallel with the same random seed for
# different classifiers. Namely, the AdaBoost simulation runs in a separate parallel
# process because it's super slow.

run_in_parallel <- function(selection, ran) {

  if ( !(selection == 1 | selection == 2) ) {
    stop(paste("Invalid `selection` parameter:", selection))
  }

  if (selection == 1) {

    #--- AdaBoost
    set.seed(ran)
    tic("AdaBoost")
    tryCatch({
      adaboost_results <- run_sim("adaboost", p)
    }, error = function(e) {
      print(paste("I caught an error!\n", e, "\n",
                  "Random seed =", ran, "\n",
                  "Classifier: AdaBoost"))
    })
    toc()
    print("--- AdaBoost DONE ---")

    results <- adaboost_results

  } else {

    #--- glm
    set.seed(ran)
    tic("GLM")
    tryCatch({
      glm_results <- run_sim("glm", p)
    }, error = function(e) {
      print(paste("I caught an error!\n", e, "\n",
                  "Random seed =", ran, "\n",
                  "Classifier: glm"))
    })
    toc()
    print("--- glm DONE ---")

    #--- glmnet
    set.seed(ran)
    tic("glmnet")
    tryCatch({
      glmnet_results <- run_sim("glmnet", p)
    }, error = function(e) {
      print(paste("I caught an error!\n", e, "\n",
                  "Random seed =", ran, "\n",
                  "Classifier: glmnet"))
    })
    toc()
    print("--- glmnet DONE ---")

    #--- SVM linear kernel
    set.seed(ran)
    tic("SVM")
    tryCatch({
      svmLinear_results <- run_sim("svmLinear2", p)
    }, error = function(e) {
      print(paste("I caught an error!\n", e, "\n",
                  "Random seed =", ran, "\n",
                  "Classifier: svmLinear2"))
    })
    toc()
    print("--- SVM DONE ---")

    #--- Random Forest
    set.seed(ran)
    tic("RF")
    tryCatch({
      rf_results <- run_sim("rf", p)
    }, error = function(e) {
      print(paste("I caught an error!\n", e, "\n",
                  "Random seed =", ran, "\n",
                  "Classifier: rf"))
    })
    toc()
    print("--- RF DONE ---")

    # combine results
    results <- bind_rows(glm_results,
                         glmnet_results,
                         svmLinear_results,
                         rf_results)
  }

  return(results)
}

#--- Run the simulation

library(fastAdaboost)
library(randomForest)
library(e1071)
library(glmnet)
library(doParallel)
registerDoParallel(cores = as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK")))

r <- foreach (i = 1:2) %dopar% {
  run_in_parallel(i, ran)
}

#--- save all results in one .csv
results <- bind_rows(r[[1]], r[[2]])
filename <- paste0("naive_", array_task_id, ".csv")
out_path <- "/lustre/project/wyp/agossman/holdout_reuse/naive_sim/"
write.csv(file = paste0(out_path, filename), x = results)
