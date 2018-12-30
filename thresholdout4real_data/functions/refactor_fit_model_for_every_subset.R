# An auxilliary function used in fit_model_for_every_subset below in order to
# compute AUC from a fitted model on training, holdout, and testing datasets
# refactor comment:
# bname is the second factor level in character form of the target
# refactor is done as less as possible to keep a good comparision with original code
get_auc <- function(tname, bname, fitted_model, features, x_train, y_train,
                    x_holdout, y_holdout, p_holdout, x_test, y_test) { cv_auc <- max(fitted_model$results$ROC) 
  # get a new CV AUC for the model with the best parameters using several
  # repeats to make it more reliable (also the best CV score of fitted_model
  # may be inflated due to randomness)
  best_params <- fitted_model$bestTune
  fitControl <- trainControl(method = "repeatedcv",
                             number = 5,
                             repeats = 10,
                             classProb = TRUE,
                             summaryFunction = twoClassSummary,
                             allowParallel = FALSE)
  x_sub <- as.matrix(x_train[, features])
  train_df <- as.data.frame(x_sub)
  # turn y into a factor b/c caret does not considered 0-1-valued
  # response variables categorical
  #train_df[tname] <- factor(letters[y_train+1])
  train_df[tname] <- y_train
  refitted_model <- caret::train(as.formula(paste(tname, " ~ .")), data = train_df,
                          method = fitted_model$method,
                          trControl = fitControl,
                          metric = "ROC",
                          # Now specify the exact model to evaluate:
                          tuneGrid = best_params)
  repeatedcv_auc <- refitted_model$results$ROC
  pred_train <- predict(fitted_model, newdata = as.data.frame(x_train), type = "prob")
  pred_train <- ROCR::prediction(predictions = pred_train[bname], labels = y_train)
  train_auc <- as.numeric(ROCR::performance(pred_train, "auc")@y.values)

  pred_holdout <- predict(fitted_model, newdata = as.data.frame(x_holdout), type = "prob")
  pred_holdout <- ROCR::prediction(pred_holdout[bname], y_holdout)
  holdout_auc <- as.numeric(ROCR::performance(pred_holdout, "auc")@y.values)

  pred_test <- predict(fitted_model, newdata = as.data.frame(x_test), type = "prob")
  pred_test <- ROCR::prediction(pred_test[bname], y_test)
  test_auc <- as.numeric(ROCR::performance(pred_test, "auc")@y.values)

  # the best possible AUC score that can be achieved on the holdout dataset
 # perfect_pred <- prediction(p_holdout, y_holdout)
 # perfect_auc <- as.numeric(performance(perfect_pred, "auc")@y.values)
  perfect_pred <- NA    # for real data, there is no perfect prediction
  perfect_auc <- NA



  return(data_frame(cv_auc = cv_auc,
                    repeatedcv_auc = repeatedcv_auc,
                    train_auc = train_auc,
                    holdout_auc = holdout_auc,
                    test_auc = test_auc,
                    perfect_auc = perfect_auc,
                    n_train = nrow(x_train)))
}

# A function to fit classification models for every subset of the the most
# significant predictors. Any classifier available in the caret package can be used.
fit_model_for_every_subset <- function(tname, bname, classifier, x_train, y_train,
                                       x_holdout, y_holdout, p_holdout,
                                       x_test, y_test, p, features_to_keep,
                                       signif_level, verbose, sanity_checks) {
  # compute 2-sample t-statistics on train data
  labelvec = unique(y_train)
  p_values <- rep(NA, p)
  t_stat <- rep(NA, p)
  t_degf <- length(y_train) - 2
  for (i in 1:p) {
    feature <- x_train[, i]
    a <- feature[y_train == labelvec[1L]]
    b <- feature[y_train == labelvec[2L]]
    t_stat[i] <- get_t_stat(a, b)
    p_values[i] <- pt(abs(t_stat[i]), df = t_degf, lower.tail = FALSE)
  }

  # initialize some parameters
  fitControl <- trainControl(method = "cv",
                             number = 5,
                             classProb = TRUE,
                             summaryFunction = twoClassSummary,
                             allowParallel = FALSE)

  #--- determine significant predictors, and fit a model including each
  # subset of newly identified significant predictors

  # the new significant variables are the ones not already present in features_to_keep
  new_signif_ind <- which(p_values <= signif_level)
  if (length(new_signif_ind) < 2) {
    # use the two most significant predictors in this case
    new_signif_ind <- which(p_values <= sort(p_values)[2])
  }
  new_signif_vars <- colnames(x_train)[new_signif_ind]
  new_signif_vars <- setdiff(new_signif_vars, features_to_keep)

  fitted_models <- list()
  selected_features <- list()
  auc <- NULL

  if (length(new_signif_vars) == 0) {
    # i.e., no new features were selected
    # in this case just fit a single model with the features in features_to_keep
    if (verbose) {
      print("No new features identified!")
    }
    x_sub <- as.matrix(x_train[, features_to_keep])
    train_df <- as.data.frame(x_sub)
    train_df[tname] <- y_train
    single_model <- caret::train(as.formula(paste(tname, " ~ .")), data = train_df,
                          method = classifier,
                          trControl = fitControl,
                          metric = "ROC")
    fitted_models[[1]] <- single_model
    selected_features[[1]] <- features_to_keep
    auc <- get_auc(tname = tname, bname = bname, fitted_model = single_model, features = features_to_keep, x_train = x_train, y_train = y_train, x_holdout = x_holdout, y_holdout = y_holdout, p_holdout = p_holdout, x_test = x_test, y_test = y_test)
  } else {
    # consider all subsets of new features
    if (length(features_to_keep) == 0) {
      # using subsets of size at least 2 in this case, because for example
      # glmnet requires at least two predictors
      all_subsets <- lapply(2:length(new_signif_vars),
                            function(x) combn(new_signif_vars, x))   # all combinations
      # # # all_subsets  are list
    } else if (length(features_to_keep) == 1) {
      # using subsets of size at least 1 in this case, because for example
      # glmnet requires at least two predictors
      all_subsets <- lapply(1:length(new_signif_vars),
                            function(x) combn(new_signif_vars, x))
    } else {
      # using all subsets in this case, because features_to_keep supplies
      # at least two additional predictors
      all_subsets <- lapply(0:length(new_signif_vars),
                            function(x) combn(new_signif_vars, x))
    }
    # all_subsets is a list
    for (n_vars in 1:length(all_subsets)) {
      subsets <- all_subsets[[n_vars]]  # matrix
      print(subsets)
      for (i in 1:ncol(subsets)) {
        new_features <- subsets[, i]
        features <- union(features_to_keep, new_features)
        x_sub <- as.matrix(x_train[, features])

        train_df <- as.data.frame(x_sub)
        #train_df$y <- factor(letters[y_train+1])
        train_df[tname] <- y_train
        fitted_model_i <- caret::train(as.formula(paste(tname, " ~ .")), data = train_df,
                                method = classifier,
                                trControl = fitControl,
                                metric = "ROC")
        fitted_models[[ length(fitted_models) + 1]] <- fitted_model_i
        selected_features[[ length(selected_features) + 1]] <- features
        auc_i <- get_auc(tname = tname, bname = bname, fitted_model = fitted_model_i, features = features,
                         x_train = x_train, y_train = y_train,
                         x_holdout = x_holdout, y_holdout = y_holdout,
                         p_holdout = p_holdout, x_test = x_test, y_test = y_test)

        if (is.null(auc)) {
          auc <- auc_i
        } else {
          auc <- bind_rows(auc, auc_i)
        }
      }
    }
  }
  print("begining sanity check")
  # sanity checks
  if (sanity_checks) {
    if (length(fitted_models) != nrow(auc)) {
      stop(paste("Indexing error! (different lengths)", "round_ind =", round_ind))
    }
    for (i in 1:nrow(auc)) {
      pred_train <- predict(fitted_models[[i]],
                            newdata = as.data.frame(x_train), type = "prob")
      pred_train <- ROCR::prediction(pred_train[bname], y_train)
      train_auc <- as.numeric(ROCR::performance(pred_train, "auc")@y.values)
      if (abs(auc$train_auc[i] - train_auc) > 1e-12) {
        stop(paste("Indexing error in", "round_ind =", round_ind, "i =", i))
      }
    }
  }

  return(list("auc" = auc, "fitted_models" = fitted_models,
              "selected_features" = selected_features, "p_values" = p_values))
}
