p <- list(n_train = 5,                   # initial number of training *patients*; note that each patient has many images (slices) and the classification task is on a per-image basis
          n_train_increase = 5,          # number *patients* added to train data in each round
          n_adapt_rounds = 10,            # number rounds
          n_holdout = 100,               # number of *images* in the holdout dataset (*note*: this is the number of images or slices, not patients)
          signif_level = 0.01,           # cutoff level used to determine which predictors to consider in each round based on their p-values
          thresholdout_threshold = 0.02, # T in the Thresholdout algorithm
          thresholdout_sigma = 0.03,     # sigma in the Thresholdout algorithm
          thresholdout_noise_distribution = "norm", # choose between "norm" and "laplace"
          n_models = 100,                # applies only to fit_model_random(); number of models to train in each adaptive round of training.
          max_features = NULL,           # applies only to fit_model_random(); restrict the maximal number of features; useful for logistic regression without regularization; if not NULL, increases by n_train_increase in each round
          t_tests = TRUE,                # applies only to fit_model_random(); whether to perform univariate t-tests at level `signif_level` on the training data to identify candidate features, which are considered for the adaptive feature selection process; if FALSE then all available features will be considered
          verbose = TRUE,
          sanity_checks = FALSE)
