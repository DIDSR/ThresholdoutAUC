#--- A function that successively fits classifiers of specified type on variables selected via
# 2-sample t-tests. Each model is fit with an increased number of cases,
# while retaining all variables selected in the previous model.
fit_models <- function(classifier, n_train, n_train_increase, n_adapt_rounds, n_holdout,
                       n_test, p, n_signif, signif_level, verbose = FALSE, sanity_checks = TRUE) {

  n_train_total <- n_train + n_train_increase * n_adapt_rounds
  n <- n_train_total + n_holdout + n_test

  #--- generate train, holdout, and test data

  # generate data frames, while making sure that the distribution of y is sufficiently balanced
  pos_prop_train <- 0
  pos_prop_train_initial <- 0
  pos_prop_holdout <- 0
  pos_prop_test <- 0
  while(!( (pos_prop_train >= 0.2 & pos_prop_train <= 0.8) &
           (pos_prop_train_initial >= 0.2 & pos_prop_train_initial <= 0.8) &
           (pos_prop_holdout >= 0.2 & pos_prop_holdout <= 0.8) &
           (pos_prop_test >= 0.2 & pos_prop_test <= 0.8) )) {
    xy_full <- generate_iid_gaussian_data(n = n, p = p, n_signif = n_signif)
    xy_train_total <- dplyr::slice(xy_full, 1:n_train_total)
    xy_holdout <- dplyr::slice(xy_full, (n_train_total+1):(n_train_total + n_holdout))
    xy_test <- dplyr::slice(xy_full,
                            (n_train_total + n_holdout + 1):(n_train_total + n_holdout + n_test))

    pos_prop_train <- sum(xy_train_total$y) / nrow(xy_train_total)
    pos_prop_train_initial <- sum(xy_train_total[1:n_train, ]$y) / nrow(xy_train_total[1:n_train, ])
    pos_prop_holdout <- sum(xy_holdout$y) / nrow(xy_holdout)
    pos_prop_test <- sum(xy_test$y) / nrow(xy_test)
  }

  # matrices
  x_train_total <- as.matrix(select(xy_train_total, -y, -p))
  x_holdout <- as.matrix(select(xy_holdout, -y, -p))
  x_test <- as.matrix(select(xy_test, -y, -p))

  # vectors
  y_train_total <- xy_train_total$y
  y_holdout <- xy_holdout$y
  y_test <- xy_test$y

  p_train_total <- xy_train_total$p
  p_holdout <- xy_holdout$p
  p_test <- xy_test$p

  features_to_keep <- c() # this is updated in every round to store names of the selected features

  #--- train the first model without looking at the holdout
  if (verbose) { print("Fitting initial model") }

  # define train data for this iteration
  x_train <- x_train_total[1:n_train, ]
  y_train <- y_train_total[1:n_train]

  # fit only one model: the one with the two most significant features
  model_fit_results <- fit_model_for_every_subset(classifier = classifier,
                                                  x_train = x_train, y_train = y_train,
                                                  x_holdout = x_holdout, y_holdout = y_holdout,
                                                  p_holdout = p_holdout, x_test = x_test,
                                                  y_test = y_test, p = p,
                                                  features_to_keep = features_to_keep,
                                                  signif_level = 0, # (this forces the function to consider 2 most significant features only)
                                                  verbose = verbose, sanity_checks = sanity_checks)

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

  #--- train subsequent models by selecting the one performing the best on the holdout

  for (round_ind in 1:n_adapt_rounds) {
    if (verbose) { print(paste0("Round: ", round_ind)) }

    # define train data for this iteration
    x_train <- x_train_total[1:(n_train + round_ind * n_train_increase), ]
    y_train <- y_train_total[1:(n_train + round_ind * n_train_increase)]

    # fit models with different numbers of features
    model_fit_results <- fit_model_for_every_subset(classifier = classifier,
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

    # package results
    best_model_ind <- which.max(auc$holdout_auc)
    features_to_keep <- union(features_to_keep, selected_features[[best_model_ind]])
    num_features_by_round <- c(num_features_by_round, length(features_to_keep))
    holdout_access_by_round <- c(holdout_access_by_round, nrow(auc))
    auc_by_round_df <- bind_rows(auc_by_round_df,
                                 mutate(auc[best_model_ind, ], round = round_ind))
  }

  auc_by_round_df <- auc_by_round_df %>% gather(dataset, auc, -round, -n_train)

  return(list(selected_features = features_to_keep,
              num_features_by_round = num_features_by_round,
              auc_by_round_df = auc_by_round_df,
              holdout_access_by_round = holdout_access_by_round,
              cum_holdout_access = cumsum(holdout_access_by_round),
              cum_budget_decrease_by_round = cumsum(holdout_access_by_round),
              pos_prop_train = pos_prop_train,
              pos_prop_holdout = pos_prop_holdout,
              pos_prop_test = pos_prop_test))
              # here the number of budget decreases is the same as the number of holdout queries;
              # it is included for consistency in the reported values with
              # the Thresholdout simulation.
}

#--- a function to run the simulation one time
run_sim <- function(method, p) {
  sim_out <- fit_models(classifier = method, n_train = p$n_train,
                        n_train_increase = p$n_train_increase,
                        n_adapt_rounds = p$n_adapt_rounds,
                        n_holdout = p$n_holdout, n_test = p$n_test,
                        p = p$p, n_signif = p$n_signif,
                        signif_level = p$signif_level,
                        verbose = p$verbose,
                        sanity_checks = p$sanity_checks)
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

  return(results)
}
