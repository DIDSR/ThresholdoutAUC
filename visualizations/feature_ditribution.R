library(tidyverse)
library(RColorBrewer)

n <- 5000
p <- 20
n_signif <- 10

set.seed(20180328)

# generate i.i.d. Gaussian predictors

x <- matrix(rnorm(n*p), n, p)
colnames(x) <- paste0("x", 1:p)
x_df <- as_data_frame(x)

# generate a non-linear binary response using logit^(-1) of a third order polynomial

# only keep n_signif of those
x_signif <- x[ , 1:n_signif]

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
xb <- x_signif_trans %*% b
probs <- 1 / (1 + exp(-xb))

x_df$probs <- probs
x_df$y <- rbinom(n = n, size = 1, p = probs)

# histogram of probs
x_df %>% ggplot(aes(probs)) +
  geom_histogram(bins = 60) +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank()) +
  xlab(expression(P(y==1))) +
  ylab("Count")

ggsave("./img/probs_hist.pdf",
       width = 3.4, height = 3.19, units = "in")

# distribution of probs conditional on y
x_df %>% mutate(y = factor(y, levels = c(0, 1),
                           labels = c("y = 0 (i.e., \"negative\" cases)",
                                      "y = 1 (i.e., \"positive\" cases)"))) %>%
  ggplot(aes(x = probs, fill = y)) +
    geom_histogram(color = "black") +
    facet_wrap(~y, nrow = 2) +
    scale_fill_manual(values = brewer.pal(n = 3, name = "Paired")) +
    xlab(expression(P(y==1))) +
    ylab("Count (out of 5000 cases total)") +
    scale_y_sqrt() +
    theme(legend.position = "none")

ggsave("./img/probs_distribution_by_y.pdf",
       width = 3.4, height = 3.19, units = "in")

# distribution of entries of x conditional on y
x_df %>% gather(x, value, -y, -probs) %>%
  mutate(significant = as.integer(gsub("x", "", x))) %>%
  mutate(significant = ifelse(significant <= 10, TRUE, FALSE)) %>%
  mutate(y = factor(y)) %>%
  mutate(x = factor(x, levels = paste0("x", 1:20),
           labels = c(paste0("x[", 1:10, "]~(beta[", 1:10, "]%~~%(", round(b[1:10], 2), ")~(", round(b[11:20], 2), ")~(", round(b[21:30], 2), "))"),
                               paste0("x[", 11:20, "]~(beta[", 11:20, "]==0)")),
                    ordered = TRUE)) %>%
  mutate(significant = ifelse(significant, "Relevant to the response",
                              "No relationship to response")) %>%
  ggplot(aes(y, value, fill = significant)) +
    geom_violin(draw_quantiles = c(0.25, 0.5, 0.75)) +
    scale_fill_manual(values = brewer.pal(n = 3, name = "Paired")) +
    facet_wrap(~x, ncol = 5, labeller = label_parsed) +
    theme(legend.position = "bottom",
          legend.title = element_blank()) +
    labs(x = "Class membership",
         y = "Explanatory variable distribution")

ggsave("./img/features_by_y.pdf")
