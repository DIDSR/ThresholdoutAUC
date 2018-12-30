initialize_thresholdout_params <- function(threshold, sigma, gamma,
                                           budget_utilized, noise_distribution) {
  if ( !(noise_distribution %in% c("norm", "laplace")) ) {
    stop("noise_distribution needs to be specified as either \"norm\" or \"laplace\"")
  }

  params <- structure(list(threshold = threshold,
                           sigma = sigma, gamma = gamma,
                           budget_utilized = budget_utilized,
                           noise_distribution = noise_distribution),
                      class = "ThresholdoutParams")
  return(params)
}


thresholdout_auc <- function(thresholdout_params, train_auc, holdout_auc) {
  if (class(thresholdout_params) != "ThresholdoutParams" |
      !all.equal(names(thresholdout_params),
                 c("threshold", "sigma", "gamma",
                   "budget_utilized", "noise_distribution"))) {
    stop("invalid thresholdout_params")
  }

  if (thresholdout_params$noise_distribution == "norm") {
    xi <- rnorm(1, sd = thresholdout_params$sigma)
    eta <- rnorm(1, sd = 4*thresholdout_params$sigma)
  } else if (thresholdout_params$noise_distribution == "laplace") {
    xi <- rlaplace(1, scale = thresholdout_params$sigma / sqrt(2))
    eta <- rlaplace(1, scale = 4*thresholdout_params$sigma / sqrt(2))
  }

  noisy_threshold <- thresholdout_params$threshold + thresholdout_params$gamma

  if (abs(holdout_auc - train_auc) > noisy_threshold + eta) {
    out <- holdout_auc + xi
    thresholdout_params$budget_utilized <- thresholdout_params$budget_utilized + 1
    # regenerate noise added to the threshold
    # (set gamma = 0, because the initial gamma in the Thresholdout algorithm
    # may be way too large if you're unlucky, in which case the test and
    # train AUC will _never_ be close enough for the algorithm
    # to return any information about the test data...)
    thresholdout_params$gamma <- 0#rlaplace(1, scale = 2*thresholdout_params$sigma)
  } else {
    out <- train_auc
  }

  return(list(params = thresholdout_params, thresholdout_auc = out))
}

