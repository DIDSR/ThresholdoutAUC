pooled_sd <- function(a, b) {
  n1 <- length(a)
  n2 <- length(b)
  pooled_estimate <- ( (n1 - 1) * sd(a) + (n2 - 1) * sd(b) ) / (n1 + n2 - 2)
  return(pooled_estimate)
}

get_t_stat <- function(a, b) {
  n1 <- length(a)
  n2 <- length(b)
  t_stat <- (mean(a) - mean(b)) / (pooled_sd(a, b) * sqrt(1/n1 + 1/n2))
  return(t_stat)
}
