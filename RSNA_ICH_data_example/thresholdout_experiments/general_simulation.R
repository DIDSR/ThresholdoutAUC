#library(tidyverse)
#library(caret)
#library(ROCR)
#library(tictoc)
#source("../set_params_RSNA-ICH.R")
#list2env(p, globalenv())
#ls()
#source("../set_params_RSNA-ICH.R")
#source("../../functions/p-values.R")
#source("../../functions/data_generation.R")
#source("../../functions/fit_model_for_every_subset.R")
#source("../../functions/thresholdout_auc.R")
#iter_i <- 49
#classifier <- "xgbTree"
#ran <- 2020 + iter_i
#set.seed(ran)

#--- A function that successively fits classifiers of specified type on variables selected via
# 2-sample t-tests. Each model is fit with an increased number of cases,
# while retaining all variables selected in the previous model.
fit_models_rsna_ich <- function(classifier = "glmnet", iter_i, n_train,
                                n_train_increase, n_adapt_rounds, n_holdout,
                                signif_level, n_models, max_features = NULL,
                                t_tests = TRUE, thresholdout_threshold,
                                thresholdout_sigma,
                                thresholdout_noise_distribution,
                                verbose = FALSE, sanity_checks = TRUE,
                                RData_save_path = NULL, ...) {

  #--- load data

  # (note the adjustment of iter_i for 0-based indexing)
  train_df <- read.csv(paste0("../data/png_256_resnet50features/50_subj_per_split/train_", iter_i - 1, ".csv"))
  holdout_df <- read.csv(paste0("../data/png_256_resnet50features/50_subj_per_split/test_", iter_i - 1, ".csv"))
  test_df <- read.csv("../data/png_256_resnet50features/50_subj_per_split/lockbox_small.csv")

  if (!all(train_df$group == iter_i - 1) | !all(holdout_df$group == iter_i - 1)) {
    stop("There must have been a mistake in the data splitting...")
  }

  # identify images by patient in the total training set, and shuffle the patients
  train_patient_IDs <- as.character(unique(train_df$patient_ID))
  train_patient_ind <- vector(mode="list", length=length(train_patient_IDs))
  repeat {
    shuffled_order <- sample(1:length(train_patient_IDs))
    for (j in 1:length(train_patient_IDs)) {
      shuf <- shuffled_order[j]
      train_patient_ind[[shuf]] <- which(train_df$patient_ID == train_patient_IDs[j])
    }
    # the number of positive labels in the initial training set in round 0 should be greater than zero:
    num_TP <- sum(sapply(train_patient_ind[1:n_train],
                         function(idx) { sum(train_df$any[idx]) }))
    if (num_TP != 0) {
      break
    }
  }

  # matrices
  x_train_total <- as.matrix(select(train_df, starts_with("x")))
  x_holdout_total <- as.matrix(select(holdout_df, starts_with("x")))
  x_test <- as.matrix(select(test_df, starts_with("x")))

  # vectors
  y_train_total <- train_df$any
  y_holdout_total <- holdout_df$any
  y_test <- test_df$any

  # shuffle holdout_total data
  repeat {
    holdout_ordering <- sample(1:length(y_holdout_total))
    x_holdout_total <- x_holdout_total[holdout_ordering, ]
    y_holdout_total <- y_holdout_total[holdout_ordering]
    if (sum(y_holdout_total[1:n_holdout]) != 0) {
      break
    }
  }

  x_holdout <- x_holdout_total[1:n_holdout, ]
  y_holdout <- y_holdout_total[1:n_holdout]
  p_holdout <- rep(0.5, n_holdout)
  # p_holdout is a placeholder, since we don't actually know p_holdout for real data (it's only available for simulated data, where p_holdout are basically simulation parameters):

  features_to_keep <- c() # this is updated in every round to store names of the selected features

  # define train data for this iteration
  ind_list <- train_patient_ind[1:n_train]
  train_ind <- unlist(ind_list)
  x_train <- x_train_total[train_ind, ]
  y_train <- y_train_total[train_ind]
  p <- dim(x_train)[2]

  pos_prop_train <- sum(y_train_total) / nrow(x_train_total)
  pos_prop_train_initial <- sum(y_train) /  nrow(x_train)
  pos_prop_holdout <- sum(y_holdout) / n_holdout
  pos_prop_test <- sum(y_test) / nrow(x_test)

  #--- train the first model without looking at the holdout

  if (verbose) {
    print("Fitting initial baseline model")
    print(paste("Total training images available:", nrow(x_train_total)))
    print(paste("Initial training data size:", nrow(x_train)))
    print(paste("Initial training data prevalence:", sum(y_train) / length(y_train)))
    print(paste("Holdout data size:", nrow(x_holdout)))
    print(paste("Holdout data prevalence:", sum(y_holdout) / length(y_holdout)))
  }

  # standardize
  preProcValues <- preProcess(x_train, method = c("center", "scale"))

  # fit only one model: the one with the two most significant features
  # Always use `t_tests = TRUE` and `signif_level = 0` in the initial model fit, in order to obtain a model that uses the 2 most significant features only.
  model_fit_results <- fit_model_random(
    classifier = classifier, x_train = predict(preProcValues, x_train),
    y_train = y_train, x_holdout = predict(preProcValues, x_holdout),
    y_holdout = y_holdout, p_holdout = p_holdout,
    x_test = predict(preProcValues, x_test), y_test = y_test, p = p,
    features_to_keep = features_to_keep, signif_level = 0,
    verbose = verbose, sanity_checks = sanity_checks, n_models = 1,
    max_features = max_features, t_tests = TRUE, ...
  )

  if (length(model_fit_results$fitted_models) > 1) { stop("Something went wrong when fitting initial model!") }

  auc <- model_fit_results$auc
  fitted_models <- model_fit_results$fitted_models
  selected_features <- model_fit_results$selected_features
  p_values <- model_fit_results$p_values
  rm(model_fit_results)

  # package results
  features_to_keep <- union(features_to_keep, selected_features[[1]])  # note: this will be reset back to be an empty vector a few lines below
  auc_by_round_df <- auc[1, ] %>% mutate(round = 0)
  num_features_by_round <- length(features_to_keep)
  holdout_access_by_round <- 0
  cum_budget_decrease_by_round <- 0

  if (verbose) { print(paste0("Initial baseline model - done: The model uses ",
                              length(features_to_keep), " features.")) }

  #--- train subsequent models by selecting the one performing the best on the holdout utilizing thresholdout

  features_to_keep <- c()  # reset the set of selected features

  # intitialize Thresholdout parameters.
  thresholdout_params <- initialize_thresholdout_params(
    threshold = thresholdout_threshold,
    sigma = thresholdout_sigma,
    # set gamma = 0, because the initial gamma in the Thresholdout algorithm
    # may be way too large if you're unlucky, in which case the test and
    # train AUC will _never_ be close enough for the algorithm
    # to return any information about the test data...
    gamma = 0, #rlaplace(1, 2*thresholdout_sigma),
    budget_utilized = 0,
    noise_distribution = thresholdout_noise_distribution
  )

  for (round_ind in 1:n_adapt_rounds) {
    if (verbose) { print(paste0("Round: ", round_ind)) }

    # define train data for this iteration
    ind_list <- train_patient_ind[1:(n_train + (round_ind-1) * n_train_increase)]
    train_ind <- unlist(ind_list)
    x_train <- x_train_total[train_ind, ]
    y_train <- y_train_total[train_ind]

    if (!is.null(max_features) & round_ind > 1) {
      # update maximal number of features to include in the model; used for logistic regression
      new_patients_ind <- ind_list[(n_train + (round_ind-2) * n_train_increase + 1):(n_train + (round_ind-1) * n_train_increase)]
      max_features <- max_features + sum(sapply(new_patients_ind, length)) / 2
    }

    # standardize
    preProcValues <- preProcess(x_train, method = c("center", "scale"))

    # fit models with different numbers of features
    model_fit_results <- fit_model_random(
      classifier = classifier, x_train = predict(preProcValues, x_train),
      y_train = y_train, x_holdout = predict(preProcValues, x_holdout),
      y_holdout = y_holdout, p_holdout = p_holdout,
      x_test = predict(preProcValues, x_test), y_test = y_test, p = p,
      features_to_keep = features_to_keep, signif_level = signif_level,
      verbose = verbose, sanity_checks = sanity_checks, n_models = n_models,
      max_features = max_features, t_tests = t_tests, ...
    )

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

    if (verbose) {
      print(paste0("Round ", round_ind, " done."))
      print(paste0("Training data consists of ", nrow(x_train), " images."))
      print(paste0("The model uses ", length(features_to_keep), " features."))
    }

    # save intermediate results:
    if (!is.null(RData_save_path)) {
      intermediate_results <- list(selected_features = features_to_keep,
                num_features_by_round = num_features_by_round,
                auc_by_round_df = auc_by_round_df,
                cum_budget_decrease_by_round = cum_budget_decrease_by_round,
                holdout_access_by_round = holdout_access_by_round)
      save(intermediate_results, file = RData_save_path)
    }
  }

  auc_by_round_df <- auc_by_round_df %>% gather(dataset, auc, -round, -n_train)

  return(list(selected_features = features_to_keep,
              num_features_by_round = num_features_by_round,
              auc_by_round_df = auc_by_round_df,
              holdout_access_by_round = holdout_access_by_round,
              cum_holdout_access = cumsum(holdout_access_by_round),
              cum_budget_decrease_by_round = cum_budget_decrease_by_round,
              pos_prop_train = pos_prop_train,
              pos_prop_holdout = pos_prop_holdout,
              pos_prop_test = pos_prop_test))
}

#--- a function to run the simulation one time
run_sim <- function(method="glmnet", iter_i, p, RData_save_path = NULL, ...) {
  sim_out <- fit_models_rsna_ich(classifier = method, iter_i = iter_i,
                                 n_train = p$n_train,
                                 n_train_increase = p$n_train_increase,
                                 n_adapt_rounds = p$n_adapt_rounds,
                                 n_holdout = p$n_holdout,
                                 signif_level = p$signif_level,
                                 n_models = p$n_models,
                                 max_features = p$max_features,
                                 t_tests = p$t_tests,
                                 thresholdout_threshold = p$thresholdout_threshold,
                                 thresholdout_sigma = p$thresholdout_sigma,
                                 thresholdout_noise_distribution = p$thresholdout_noise_distribution,
                                 verbose = p$verbose,
                                 sanity_checks = p$sanity_checks,
                                 RData_save_path = RData_save_path, ...)
  results <- mutate(sim_out$auc_by_round_df, method = method)
  num_features_df <- data_frame(round = 0:p$n_adapt_rounds,
                                num_features = sim_out$num_features_by_round)
  results <- left_join(results, num_features_df, by = "round")
  holdout_access_count_df <- data_frame(round = 0:p$n_adapt_rounds,
                                        holdout_access_count = sim_out$holdout_access_by_round,
                                        cum_holdout_access_count = sim_out$cum_holdout_access,
                                        cum_budget_decrease_by_round = sim_out$cum_budget_decrease_by_round)
  results <- left_join(results, holdout_access_count_df, by = "round")
  results <- results %>%
    mutate(pos_prop_train = sim_out$pos_prop_train,
           pos_prop_holdout = sim_out$pos_prop_holdout,
           pos_prop_test = sim_out$pos_prop_test)
  results$features <- rep("not recorded", nrow(results))
  results$features[which(results$round == p$n_adapt_rounds)] <- paste(sim_out$selected_features,
                                                                      collapse=",")

  return(results)
}
