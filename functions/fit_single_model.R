# An auxilliary function used in fit_model_for_every_subset below in order to
# compute AUC from a fitted model on training, holdout, and testing datasets
get_auc <- function(fitted_model, x_train, y_train, x_holdout,
                    y_holdout, p_holdout, x_test, y_test) {
  cv_auc <- max(fitted_model$results$ROC)

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
  train_df <- as.data.frame(x_train)
  # turn y into a factor b/c caret does not considered 0-1-valued
  # response variables categorical
  train_df$y <- factor(letters[y_train+1])
  refitted_model <- train(y ~ ., data = train_df,
                          method = fitted_model$method,
                          trControl = fitControl,
                          metric = "ROC",
                          # Now specify the exact model to evaluate:
                          tuneGrid = best_params)
  repeatedcv_auc <- refitted_model$results$ROC

  pred_train <- predict(fitted_model, newdata = as.data.frame(x_train), type = "prob")
  pred_train <- prediction(pred_train$b, y_train)
  train_auc <- as.numeric(performance(pred_train, "auc")@y.values)

  pred_holdout <- predict(fitted_model, newdata = as.data.frame(x_holdout), type = "prob")
  pred_holdout <- prediction(pred_holdout$b, y_holdout)
  holdout_auc <- as.numeric(performance(pred_holdout, "auc")@y.values)

  pred_test <- predict(fitted_model, newdata = as.data.frame(x_test), type = "prob")
  pred_test <- prediction(pred_test$b, y_test)
  test_auc <- as.numeric(performance(pred_test, "auc")@y.values)

  # the best possible AUC score that can be achieved on the holdout dataset
  perfect_pred <- prediction(p_holdout, y_holdout)
  perfect_auc <- as.numeric(performance(perfect_pred, "auc")@y.values)

  return(data_frame(cv_auc = cv_auc,
                    repeatedcv_auc = repeatedcv_auc,
                    train_auc = train_auc,
                    holdout_auc = holdout_auc,
                    test_auc = test_auc,
                    perfect_auc = perfect_auc,
                    n_train = nrow(x_train)))
}


# A function to fit a single classification model with tuning via CV.
# Any classifier available in the caret package can be used.
fit_single_model <- function(classifier, x_train, y_train,
                             x_holdout, y_holdout, p_holdout,
                             x_test, y_test) {

  fitControl <- trainControl(method = "cv",
                             number = 5,
                             classProb = TRUE,
                             summaryFunction = twoClassSummary,
                             allowParallel = FALSE)

  train_df <- as.data.frame(x_train)
  # turn y into a factor b/c caret does not considered 0-1-valued
  # response variables categorical
  train_df$y <- factor(letters[y_train+1])

  single_model <- train(y ~ ., data = train_df,
                        method = classifier,
                        trControl = fitControl,
                        metric = "ROC")

  auc <- get_auc(fitted_model = single_model,
                 x_train = x_train, y_train = y_train,
                 x_holdout = x_holdout, y_holdout = y_holdout,
                 p_holdout = p_holdout, x_test = x_test, y_test = y_test)

  return(list("auc" = auc, "fitted_models" = list(single_model)))
}
