library(caret)
library(plyr)
library(dplyr)
library(tidyr)
#library(VGAM) # for laplace distribution
library(ROCR)
library(tictoc)

#--- load functions

# functions to calculate p-values
source("../functions/p-values.R")
# functions to generate datasets
source("../functions/data_generation.R")
# a function to fit classification models for every subset of the most significant predictors
source("../functions/refactor_fit_model_for_every_subset.R")
# set simulation parameters (such as number of cases, number of features, etc.)
source("../set_params.R")
# functions to run the main simulation one time for a specified method
source("general_simulation.R")
# thresholdout algorithm
source("../functions/thresholdout_auc.R")

getMlrTask = function() {
  n_all = 208
  n_train = 50  ## begin
  n_train_increase = 5
  n_train_total = 100
  n_holdout = 80
  n_test = 28
  ind_all = sample(n_all)
  ind_tr_begin = ind_all[1:n_train]
  ind_tr_total = ind_all[1:n_train_total]
  ind_val = ind_all[((n_train_total + 1) : (n_train_total + n_holdout))]
  ind_test = ind_all[((n_train_total + n_holdout + 1L) : (n_all))]
  require(mlr)
  require(dplyr)
  sonar.task
  dfpair = getTaskData(sonar.task, target.extra = T)
  names(dfpair)
  tname = mlr::getTaskTargetNames(sonar.task)
# matrices
  p = getTaskNFeats(sonar.task)
  classifier = "glm"
  xy_train_total = getTaskData(sonar.task)[ind_tr_total, ]
  x_train_total <- as.matrix(dfpair$data[ind_tr_total, ])
  #n_train_total <- n_train + n_train_increase * n_adapt_rounds
  #n <- n_train_total + n_holdout + n_test
  x_holdout <- as.matrix(dfpair$data[ind_val, ])
  x_test <- as.matrix(dfpair$data[ind_test, ])

  # vectors
  y_train_total <- xy_train_total[, tname]
  y_holdout <- dfpair$target[ind_val]
  y_test <- dfpair$target[ind_test]

  ##
  #xy_holdout <- dplyr::slice(xy_full, (n_train_total+1):(n_train_total + n_holdout))
  xy_holdout = getTaskData(sonar.task)[ind_val, ]

  #xy_test <- dplyr::slice(xy_full, (n_train_total + n_holdout + 1):(n_train_total + n_holdout + n_test))
  xy_test <- getTaskData(sonar.task)[ind_test, ]

  return(list(n_train = n_train, x_train_total = x_train_total, y_train_total = y_train_total, ind_test = ind_test, ind_val = ind_val, x_train_total = x_train_total, x_holdout = x_holdout, y_holdout = y_holdout, x_test = x_test, y_test = y_test, n_train_increase = n_train_increase, n_holdout = n_holdout))
}



#--- A function that successively fits classifiers of specified type on variables selected via
# 2-sample t-tests. Each model is fit with an increased number of cases,
# while retaining all variables selected in the previous model.
fit_models = function(fun = getMlrTask, tname, bname, classifier, p, n_adapt_rounds, n_signif, signif_level, thresholdout_threshold, thresholdout_sigma, thresholdout_noise_distribution, verbose = FALSE, sanity_checks = TRUE) {
  tuple = fun()
  x_train_total = tuple$x_train_total
  ind_test = tuple$ind_test
  y_train_total = tuple$y_train_total
  x_holdout = tuple$x_holdout
  x_test = tuple$x_test
  y_test = tuple$y_test
  y_holdout = tuple$y_holdout
  n_train = tuple$n_train
  n_train_increase = tuple$n_train_increase
  #p_train_total <- xy_train_total$p
  #p_holdout <- xy_holdout$p
  #p_test <- xy_test$p
  p_train_total <- 1
  p_holdout <- 1
  p_test <- 1

  features_to_keep <- c() # this is updated in every round to store names of the selected features
  #--- train the first model without looking at the holdout
  if (verbose) { print("Fitting initial model") }
  # define train data for this iteration
  x_train <- x_train_total[1:n_train, ]
  y_train <- y_train_total[1:n_train]

  # fit only one model: the one with the two most significant features
  model_fit_results <- fit_model_for_every_subset(tname = tname, bname = "R", classifier = classifier,
                                                  x_train = x_train, y_train = y_train,
                                                  x_holdout = x_holdout, y_holdout = y_holdout,
                                                  p_holdout = p_holdout, x_test = x_test,
                                                  y_test = y_test, p = p,
                                                  features_to_keep = features_to_keep,
                                                  signif_level = 0, # (this forces the function to consider 2 most significant features only)
                                                  verbose = F, sanity_checks = T)

  if (length(model_fit_results$fitted_models) > 1) { stop("Something went wrong when fitting initial model!") }

  auc <- model_fit_results$auc
  fitted_models <- model_fit_results$fitted_models
  selected_features <- model_fit_results$selected_features
  p_values <- model_fit_results$p_values
  rm(model_fit_results)

  # package results
  features_to_keep <- union(features_to_keep, selected_features[[1]])
  auc_by_round_df <- auc[1, ] %>% mutate(round = 0)
  num_features_by_round <- length(features_to_keep)
  holdout_access_by_round <- 0
  cum_budget_decrease_by_round <- 0

  #--- train subsequent models by selecting the one performing the best on the holdout utilizing thresholdout

  # intitialize Thresholdout parameters.
  thresholdout_params <- initialize_thresholdout_params(threshold = thresholdout_threshold,
                                                        sigma = thresholdout_sigma,
                                                        # set gamma = 0, because the initial gamma in the Thresholdout algorithm
                                                        # may be way too large if you're unlucky, in which case the test and
                                                        # train AUC will _never_ be close enough for the algorithm
                                                        # to return any information about the test data...
                                                        gamma = 0, #rlaplace(1, 2*thresholdout_sigma),
                                                        budget_utilized = 0,
                                                        noise_distribution = thresholdout_noise_distribution)
#                                                        noise_distribution = "norm")

  for (round_ind in 1:n_adapt_rounds) {
    if (verbose) { print(paste0("Round: ", round_ind)) }

    # define train data for this iteration
    x_train <- x_train_total[1:(n_train + round_ind * n_train_increase), ]
    y_train <- y_train_total[1:(n_train + round_ind * n_train_increase)]

    # fit models with different numbers of features
    model_fit_results <- fit_model_for_every_subset(tname = tname, bname = "R", classifier = classifier,
                                                    x_train = x_train, y_train = y_train,
                                                    x_holdout = x_holdout, y_holdout = y_holdout,
                                                    p_holdout = p_holdout, x_test = x_test,
                                                    y_test = y_test, p = p,
                                                    features_to_keep = features_to_keep,
                                                    signif_level = signif_level,
                                                    verbose = verbose, sanity_checks = sanity_checks)
    auc <- model_fit_results$auc
    fitted_models <- model_fit_results$fitted_models
    selected_features <- model_fit_results$selected_features
    p_values <- model_fit_results$p_values
    rm(model_fit_results)

    # get the thresholdout auc
    th_auc_vec <- rep(NA, nrow(auc))
    for (model_ind in 1:nrow(auc)) {
      temp <- thresholdout_auc(thresholdout_params = thresholdout_params,
                               train_auc = auc$repeatedcv_auc[model_ind],
                               holdout_auc = auc$holdout_auc[model_ind])
      th_auc_vec[model_ind] <- temp$thresholdout_auc
      thresholdout_params <- temp$params
    }
    auc <- mutate(auc, thresholdout_auc = th_auc_vec)

    # package results
    best_model_ind <- which.max(auc$thresholdout_auc)
    # (the recorded thresholdout_auc is not useful for model evaluation,
    # because it captures an instantiation where the added noise happened
    # to be extremely large and positive. So recalculate the Thresholdout
    # score for the best model)
    temp <- thresholdout_auc(thresholdout_params = thresholdout_params,
                             train_auc = auc$repeatedcv_auc[best_model_ind],
                             holdout_auc = auc$holdout_auc[best_model_ind])
    auc$thresholdout_auc[best_model_ind] <- temp$thresholdout_auc
    thresholdout_params <- temp$params
    features_to_keep <- union(features_to_keep, selected_features[[best_model_ind]])
    num_features_by_round <- c(num_features_by_round, length(features_to_keep))
    holdout_access_by_round <- c(holdout_access_by_round, nrow(auc)+1)
    cum_budget_decrease_by_round <- c(cum_budget_decrease_by_round,
                                      thresholdout_params$budget_utilized)
    auc_by_round_df <- bind_rows(auc_by_round_df,
                                 mutate(auc[best_model_ind, ], round = round_ind))
  }

  auc_by_round_df <- auc_by_round_df %>% gather(dataset, auc, -round, -n_train)

  return(list(selected_features = features_to_keep,
              num_features_by_round = num_features_by_round,
              auc_by_round_df = auc_by_round_df,
              holdout_access_by_round = holdout_access_by_round,
              cum_holdout_access = cumsum(holdout_access_by_round),
              cum_budget_decrease_by_round = cum_budget_decrease_by_round))
}

#--- a function to run the simulation one time
run_sim <- function(method, p) {
  method = "glm"
  bname = "R"
  tname = "Class"
  conf = list(n_signif = 10                 # number of predictors that have an effect on the response
  ,p = 60
  , n_adapt_rounds = 10
  ,signif_level = 0.01           # cutoff level used to determine which predictors to consider in each round based on their p-values
  ,thresholdout_threshold = 0.02 # T in the Thresholdout algorithm
  ,thresholdout_sigma = 0.03     # sigma in the Thresholdout algorithm
  ,thresholdout_noise_distribution = "norm" # choose between "norm" and "laplace"
  ,verbose = TRUE
  ,sanity_checks = FALSE
  )
  sim_out <- fit_models(tname = tname, bname = bname, classifier = method,
                        p = conf$p, n_adapt_rounds = conf$n_adapt_rounds, n_signif = conf$n_signif,
                        signif_level = conf$signif_level,
                        thresholdout_threshold = conf$thresholdout_threshold,
                        thresholdout_sigma = conf$thresholdout_sigma,
                        thresholdout_noise_distribution = conf$thresholdout_noise_distribution,
                        verbose = conf$verbose,
                        sanity_checks = conf$sanity_checks)
  results <- mutate(sim_out$auc_by_round_df, method = method)



  num_features_df <- data_frame(round = 0:conf$n_adapt_rounds,
                                num_features = sim_out$num_features_by_round)

  results <- left_join(results, num_features_df, by = "round")

  holdout_access_count_df <- data_frame(round = 0:conf$n_adapt_rounds,
                                        holdout_access_count = sim_out$holdout_access_by_round,
                                        cum_holdout_access_count = sim_out$cum_holdout_access,
                                        cum_budget_decrease_by_round = sim_out$cum_budget_decrease_by_round)
  results <- left_join(results, holdout_access_count_df, by = "round")
#  results <- results %>%
#    mutate(pos_prop_train = sim_out$pos_prop_train,
#           pos_prop_holdout = sim_out$pos_prop_holdout,
#           pos_prop_test = sim_out$pos_prop_test)

  return(results)
}
