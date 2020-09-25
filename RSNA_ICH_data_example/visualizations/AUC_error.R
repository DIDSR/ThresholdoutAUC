library(tidyverse)
library(RColorBrewer)
library(xtable)

# load simulation parameters (such as number of cases, number of features, etc.)
source("../set_params_RSNA-ICH.R")
source("../../functions/thresholdout_auc.R")

thresholdout_csv <- "../thresholdout_simulation/results/thresh_random_all_results.csv"
naive_csv <- "../naive_data_reuse_simulation/results/naive_random_all_results.csv"

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
                         levels = c("glmnet", "xgbTree"),
                         labels = c("L1- and L2-regularized GLM", "XGBoost")))

# (signed & absolute error)
tab <- error_df %>% group_by(method, holdout_reuse) %>%
  summarize("Mean signed error" = mean(error),
            "Median signed error" = median(error),
            "SD signed error" = sd(error),
            "Mean absolute error" = mean(abs_error),
            "Median absolute error" = median(abs_error),
            "SD absolute error" = sd(abs_error)) %>%
  xtable(digits = 4)

print(tab, file = "signed_and_absolute_error_LaTeX_table.txt")

# overall across all methods
overall_error_summary_df <- error_df %>% group_by(holdout_reuse) %>%
  summarize("Mean signed error" = mean(error),
            "Median signed error" = median(error),
            "SD signed error" = sd(error),
            "Mean absolute error" = mean(abs_error),
            "Median absolute error" = median(abs_error),
            "SD absolute error" = sd(abs_error))
tab <- overall_error_summary_df %>% xtable(digits = 4)

print(tab, file = "signed_and_absolute_error_overall_LaTeX_table.txt")

overall_error_summary_df %>% glimpse()
# Rows: 2
# Columns: 7
# $ holdout_reuse           <fct> Naive test data reuse, Th…
# $ `Mean signed error`     <dbl> 0.10370827, 0.05116094
# $ `Median signed error`   <dbl> 0.10770799, 0.04478623
# $ `SD signed error`       <dbl> 0.06940799, 0.08709853
# $ `Mean absolute error`   <dbl> 0.10924786, 0.07684883
# $ `Median absolute error` <dbl> 0.10874684, 0.06060606
# $ `SD absolute error`     <dbl> 0.06030903, 0.06554429

#--- Plot error distribution (density) by classifier

avg_error_df <- error_df %>%
  select(error, holdout_reuse, method) %>%
  group_by(method, holdout_reuse) %>%
  summarize(error_mean = mean(error), error_sd = sd(error), n_reps = n()) %>% tbl_df()

# (penalized GLM and XGBoost only - PDF)
pal <- brewer.pal(4, "Dark2")
error_df %>% left_join(avg_error_df) %>%
  filter(method == "L1- and L2-regularized GLM" | method == "XGBoost") %>%
  ggplot(aes(x = error, y = ..density..,
             color = holdout_reuse, linetype = holdout_reuse)) +
    geom_freqpoly(bins = 30, position = "identity", size = 1) +
    geom_vline(aes(xintercept = error_mean, color = holdout_reuse,
                   linetype = holdout_reuse)) +
    facet_wrap(~method, nrow=2) +
    #scale_color_brewer(palette = "Dark2") +
    scale_linetype_manual(values = c("Naive test data reuse" = 1,
                                     "Thresholdout" = 2),
                          name = "",
                          labels = c(bquote(~"Naive test data reuse"),
                                     bquote(~Thresholdout[AUC]))) +
    scale_color_manual(values = c("Naive test data reuse" = pal[2],
                                  "Thresholdout" = pal[3]),
                       name = "",
                       labels = c(bquote(~"Naive test data reuse"),
                                  bquote(~Thresholdout[AUC]))) +
    theme_bw() +
    xlab("Error (reported AUC minus true AUC)") +
    ylab("Density") +
    theme(legend.title = element_blank(),
          legend.position = "bottom")

ggsave("./img/fig-RSNA-reported_AUC_minus_true_AUC.pdf",
       width = 3.5, height = 6.0, units = "in")

#--- Plot the CDF of the absolute error by classifier

# (penalized GLM and XGBoost only - PDF)
pal <- brewer.pal(3, "Dark2")
error_df %>% select(method, holdout_reuse, abs_error) %>%
  filter(method == "L1- and L2-regularized GLM" | method == "XGBoost") %>%
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

ggsave("./img/fig-RSNA-absolote_error_CDF_ThresholdoutAUC.pdf",
       width = 5.8, height = 3.0, units = "in")
