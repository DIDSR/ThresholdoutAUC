library(tidyverse)
library(RColorBrewer)
library(xtable)

# load simulation parameters (such as number of cases, number of features, etc.)
source("../set_params.R")
source("../functions/thresholdout_auc.R")

naive_csv <- "../results/Cypress_Tulane_HPC/cypress_naive_all_results.csv"
thresholdout_csv <- "../results/Cypress_Tulane_HPC/cypress_thresholdout_all_results.csv"
naive_holdout <- read_csv(naive_csv)
thresholdout <- read_csv(thresholdout_csv)

naive_holdout <- mutate(naive_holdout, holdout_reuse = "Naive test data reuse")
thresholdout <- mutate(thresholdout, holdout_reuse = "Thresholdout")

results <- bind_rows(naive_holdout, thresholdout)

#--- Table of the error between reported and true AUC

error_df <- results %>%
  filter(round != 0) %>% # Not taking error in round 0, because only error in the *adaptive* rounds is of interest.
  select(-n_train, -num_features, -holdout_access_count) %>%
  filter((dataset == "holdout_auc" & holdout_reuse == "Naive test data reuse") |
         (dataset == "thresholdout_auc" & holdout_reuse == "Thresholdout") |
         dataset == "test_auc") %>%
  mutate(dataset = ifelse(dataset == "thresholdout_auc" | dataset == "holdout_auc",
                          "Reported performance", "True performance")) %>%
  spread(dataset, auc) %>%
  mutate(error = `Reported performance` - `True performance`) %>%
  mutate(abs_error = abs(error)) %>%
  mutate(sq_error = error^2) %>%
  filter(!is.na(error)) %>%
  mutate(holdout_reuse = factor(holdout_reuse)) %>%
  mutate(method = factor(method,
                         levels = c("glm", "glmnet", "svmLinear2",
                                    "rf", "adaboost"),
                         labels = c("Logistic regression (GLM)", "L1- and L2-regularized GLM",
                                    "SVM with linear kernel", "Random Forest",
                                    "AdaBoost")))

# (signed & absolute error)
tab <- error_df %>% group_by(method, holdout_reuse) %>%
  summarize("Mean signed error" = mean(error),
            "Median signed error" = median(error),
            "SD signed error" = sd(error),
            "Mean absolute error" = mean(abs_error),
            "Median absolute error" = median(abs_error),
            "SD absolute error" = sd(abs_error)) %>%
  xtable(digits = 3)

print(tab, file = "signed_and_absolute_error_LaTeX_table.txt")

#--- Plot error distribution (density) by classifier

# (GLM and Adaboost only - PDF)
error_df %>% left_join(avg_error_df) %>%
  filter(method == "Logistic regression (GLM)" | method == "AdaBoost") %>%
  ggplot(aes(x = error, y = ..density..,
             color = holdout_reuse, linetype = holdout_reuse)) +
    geom_freqpoly(bins = 30, position = "identity", size = 1) +
    geom_vline(aes(xintercept = error_mean, color = holdout_reuse,
                   linetype = holdout_reuse)) +
    facet_wrap(~method) +
    scale_color_brewer(palette = "Dark2") +
    theme_bw() +
    xlab("Error (reported AUC minus true AUC)") +
    ylab("Density") +
    theme(legend.position = "none")

ggsave("./img/report_AUC_minus_true_AUC_Gaussian_noise_GLM_and_Adaboost_only.pdf",
       width = 5.8, height = 3.0, units = "in")

#--- Plot the CDF of the absolute error by classifier

# (only glm and adaboost)
pal <- brewer.pal(3, "Dark2")
error_df %>% select(method, holdout_reuse, abs_error) %>%
  filter(method == "Logistic regression (GLM)" | method == "AdaBoost") %>%
  ggplot(aes(x = abs_error, color = holdout_reuse, linetype = holdout_reuse)) +
    stat_ecdf(size = 0.7) +
    facet_wrap(~method) +
    scale_linetype_manual(values = c("Naive test data reuse" = 1,
                                     "Thresholdout" = 2),
                          name = "",
                          labels = c(bquote(~"Naive test data reuse"),
                                     bquote(~Thresholdout[AUC]))) +
    scale_color_manual(values = c("Naive test data reuse" = pal[1],
                                  "Thresholdout" = pal[2]),
                       name = "",
                       labels = c(bquote(~"Naive test data reuse"),
                                  bquote(~Thresholdout[AUC]))) +
    xlim(0, 0.25) +
    theme_bw() +
    xlab(expression(group("|", ~"Error", "|") == group("|", ~"reported AUC" - ~"true AUC", "|"))) +
    ylab("Cumulative distribution function (CDF)") +
    theme(legend.title = element_blank(),
          legend.background = element_rect(size = 0.4, color = "black"),
          legend.position = c(0.8, 0.2))

ggsave("./img/absolote_error_CDF_Thresholdout_Gaussian_noise_GLM_and_Adaboost_only.pdf",
       width = 5.8, height = 3.0, units = "in")
