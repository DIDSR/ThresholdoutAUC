#--- A function that successively fits classifiers of
# a specified type on the subset of significant
# variables among a large number of variables.
# Each model is fit with an increased number of cases.
fit_models <- function(classifier, n_train, n_train_increase,
                       n_adapt_rounds, n_holdout, n_test, p,
                       n_signif, verbose = FALSE) {

  n_train_total <- n_train + n_train_increase * n_adapt_rounds
  n <- n_train_total + n_holdout + n_test

  #--- generate train, holdout, and test data

  # generate data frames, while making sure that the distribution of y is somewhat balanced
  pos_prop_train <- 0
  pos_prop_train_initial <- 0
  pos_prop_holdout <- 0
  pos_prop_test <- 0
  while(!( (pos_prop_train >= 0.2 & pos_prop_train <= 0.8) &
           (pos_prop_train_initial >= 0.2 & pos_prop_train_initial <= 0.8) &
           (pos_prop_holdout >= 0.2 & pos_prop_holdout <= 0.8) &
           (pos_prop_test >= 0.2 & pos_prop_test <= 0.8) )) {
    data_gen_results <- generate_iid_gaussian_data(n = n, p = p, n_signif = n_signif,
                                                   return_ind_signif = TRUE)
    ind_signif <- data_gen_results$ind_signif
    xy_full <- data_gen_results$x_df
    xy_train_total <- dplyr::slice(xy_full, 1:n_train_total)
    xy_holdout <- dplyr::slice(xy_full, (n_train_total+1):(n_train_total + n_holdout))
    xy_test <- dplyr::slice(xy_full,
                            (n_train_total + n_holdout + 1):(n_train_total + n_holdout + n_test))

    pos_prop_train <- sum(xy_train_total$y) / nrow(xy_train_total)
    pos_prop_train_initial <- sum(xy_train_total[1:n_train, ]$y) / nrow(xy_train_total[1:n_train, ])
    pos_prop_holdout <- sum(xy_holdout$y) / nrow(xy_holdout)
    pos_prop_test <- sum(xy_test$y) / nrow(xy_test)
  }

  # columns of x_* are named as "x1", "x2", etc.
  features_to_keep <- paste0("x", ind_signif)

  # matrices
  x_train_total <- as.matrix(select(xy_train_total, one_of(features_to_keep)))
  x_holdout <- as.matrix(select(xy_holdout, one_of(features_to_keep)))
  x_test <- as.matrix(select(xy_test, one_of(features_to_keep)))

  # vectors
  y_train_total <- xy_train_total$y
  y_holdout <- xy_holdout$y
  y_test <- xy_test$y

  p_train_total <- xy_train_total$p
  p_holdout <- xy_holdout$p
  p_test <- xy_test$p

  #--- train models

  auc_by_round_df <- NULL

  for (round_ind in 0:n_adapt_rounds) {
    if (verbose) { print(paste0("Round: ", round_ind)) }

    # define train data for this iteration
    x_train <- x_train_total[1:(n_train + round_ind * n_train_increase), ]
    y_train <- y_train_total[1:(n_train + round_ind * n_train_increase)]

    # fit model with significant features only
    model_fit_results <- fit_single_model(classifier = classifier,
                                          x_train = x_train, y_train = y_train,
                                          x_holdout = x_holdout, y_holdout = y_holdout,
                                          x_test = x_test, y_test = y_test,
                                          p_holdout = p_holdout)
    auc <- model_fit_results$auc
    fitted_models <- model_fit_results$fitted_models
    rm(model_fit_results)

    # package results
    if (is.null(auc_by_round_df)) {
      auc_by_round_df <- auc %>% mutate(round = 0)
    } else {
      auc_by_round_df <- bind_rows(auc_by_round_df, mutate(auc, round = round_ind))
    }
  }

  auc_by_round_df <- auc_by_round_df %>% gather(dataset, auc, -round, -n_train)

  return(list(auc_by_round_df = auc_by_round_df,
              pos_prop_train = pos_prop_train,
              pos_prop_holdout = pos_prop_holdout,
              pos_prop_test = pos_prop_test))
}

#--- a function to run the simulation one time
run_sim <- function(method, p) {
  sim_out <- fit_models(classifier = method, n_train = p$n_train,
                        n_train_increase = p$n_train_increase,
                        n_adapt_rounds = p$n_adapt_rounds,
                        n_holdout = p$n_holdout, n_test = p$n_test,
                        p = p$p, n_signif = p$n_signif,
                        verbose = p$verbose)
  results <- mutate(sim_out$auc_by_round_df, method = method)
  results <- results %>%
    mutate(pos_prop_train = sim_out$pos_prop_train,
           pos_prop_holdout = sim_out$pos_prop_holdout,
           pos_prop_test = sim_out$pos_prop_test)

  return(results)
}
