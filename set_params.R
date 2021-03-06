p <- list(n_train = 100,                 # number of training samples
          n_train_increase = 10,         # number samples added to train data in each round
          n_adapt_rounds = 30,           # number rounds
          n_holdout = 100,               # number of samples in the holdout dataset
          n_test = 5000,                 # number of samples in the test dataset
          p = 300,                       # number of predictors
          n_signif = 10,                 # number of predictors that have an effect on the response
          signif_level = 0.01,           # cutoff level used to determine which predictors to consider in each round based on their p-values
          thresholdout_threshold = 0.02, # T in the Thresholdout algorithm
          thresholdout_sigma = 0.03,     # sigma in the Thresholdout algorithm
          thresholdout_noise_distribution = "norm", # choose between "norm" and "laplace"
          verbose = TRUE,
          sanity_checks = FALSE)
