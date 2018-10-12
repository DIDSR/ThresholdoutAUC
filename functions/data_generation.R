generate_iid_gaussian_data <- function(n, p, n_signif, return_ind_signif = FALSE) {

  if (n_signif < 0) { stop("need n_signif can't be negative.") }
  if ( (n_signif > 0) & (n_signif <= 2) ) { stop("need n_signif=0 or n_signif>2.") }

  # generate i.i.d. Gaussian predictors

  x <- matrix(rnorm(n*p), n, p)
  colnames(x) <- paste0("x", 1:p)

  # generate a non-linear binary response using logit^(-1) of a third order polynomial

  if (n_signif > 0) {
    # only keep n_signif of those
    ind_signif <- sample(1:ncol(x), n_signif)
    x_signif <- x[ , ind_signif]

    # add a few interactions
    x_signif_inter <- model.matrix(y ~ .*.,
                                   bind_cols(data_frame(y = rnorm(n)),
                                             as_data_frame(x_signif)))
    # (remove intercept and non-interaction effects)
    x_signif_inter <- x_signif_inter[ , (n_signif+2):ncol(x_signif_inter)]
    x_signif_inter <- x_signif_inter[ , sample(1:ncol(x_signif_inter), n_signif)]

    # include non-linear effects
    x_signif_nonlinear <- cbind(x_signif^2, exp(x_signif))
    colnames(x_signif_nonlinear) <- c(paste0(colnames(x_signif), "squared"),
                                      paste0(colnames(x_signif), "exp"))

    # combine linear, non-linear, and interaction effects in one matrix
    x_signif_trans <- cbind(x_signif, x_signif_nonlinear, x_signif_inter)

    # generate the response
    b <- 0.5 * rep(c(1, -1), 2*n_signif)
    # (using *consecutive* positive and negative coefficients is important here, because that makes the distribution of `probs` less skewed; otherwise, the problem stems from the fact that quadratic and exponential terms are always positive)
    xb <- x_signif_trans %*% b
    probs <- 1 / (1 + exp(-xb))
  } else {
    probs <- rep(0.5, n)
  }

  x_df <- as_data_frame(x)
  x_df$p <- probs
  x_df$y <- rbinom(n = n, size = 1, p = probs)

  if (return_ind_signif) {
    return(list("x_df" = x_df, "ind_signif" = ind_signif))
  } else {
    return(x_df)
  }
}
