library(tidyverse)
library(RColorBrewer)

# load simulation parameters (such as number of cases, number of features, etc.)
source("../set_params_RSNA-ICH.R")
source("../../functions/thresholdout_auc.R")

thresholdout_csv <- "../thresholdout_experiments/results/thresh_random_all_results.csv"
naive_csv <- "../naive_data_reuse_experiments/results/naive_random_all_results.csv"

thresholdout <- read_csv(thresholdout_csv)
naive_holdout <- read_csv(naive_csv)

thresholdout <- mutate(thresholdout, holdout_reuse = "Thresholdout")
naive_holdout <- mutate(naive_holdout, holdout_reuse = "Naive test data reuse")

results <- bind_rows(naive_holdout, thresholdout)

#--- look at the average AUCs across replications

auc_df <- results %>%
  select(-n_train, -num_features, -holdout_access_count) %>%
  group_by(round, dataset, method, holdout_reuse) %>%
  summarize(auc_mean = mean(auc), auc_sd = sd(auc), n_reps = n()) %>% as_tibble()

auc_df <- auc_df %>%
  mutate(method = factor(method,
                         levels = c("glm", "glmnet", "svmLinear2",
                                    "rf", "adaboost", "xgbTree"),
                         labels = c(bquote(~"Logistic regression (GLM)"),
                                    bquote(~"L1- and L2-regularized GLM"),
                                    bquote(~"SVM with linear kernel"),
                                    bquote(~"Random Forest"),
                                    bquote(~"AdaBoost"),
                                    bquote(~"XGBoost")))) %>%
  # keep only the 10-times repeated 5-fold CV AUC on train data (though the unrepeated 5-fold CV AUC measures are essentially the same, only slightly more overestimating).
  filter(dataset != "cv_auc") %>%
  mutate(dataset = factor(dataset,
                          levels = c("repeatedcv_auc", "train_auc",
                                     "holdout_auc", "thresholdout_auc",
                                     "test_auc", "perfect_auc"),
                          labels = c("Training performance (cross-validation)",
                                     "Resubstitution",
                                     "Test performance", "Thresholdout output",
                                     "True performance", "Perfect classifier"),
                          ordered = TRUE))

labs1 <- c("Training performance (cross-validation)",
           "Test performance (naive test data reuse)",
           expression(Thresholdout["AUC"]~output),
           "True performance (large independent dataset)") # NEW and the related places
guide1 <- guide_legend(nrow = 2, byrow = TRUE) # NEW and the related places

auc_df %>%
  filter(!(holdout_reuse == "Thresholdout" & dataset == "Test performance")) %>%
  filter(dataset != "Resubstitution", dataset != "Perfect classifier") %>%
  mutate(holdout_reuse = factor(holdout_reuse,
                                levels = c("Naive test data reuse", "Thresholdout"),
                                labels = c(bquote(~"Naive test data reuse"),
                                           bquote(~Thresholdout[AUC])))) %>%
  filter(round > 0) %>%  # NEW
  ggplot() +
    geom_line(aes(round, auc_mean, color = dataset, linetype = dataset)) +
    geom_ribbon(aes(x = round, ymin = auc_mean - 2*auc_sd/sqrt(n_reps),
                    ymax = auc_mean + 2*auc_sd/sqrt(n_reps), fill = dataset),
                alpha = 0.25) +
    scale_color_brewer(palette = "Dark2", labels = labs1, guide = guide1) +
    scale_fill_brewer(palette = "Dark2", labels = labs1, guide = guide1) +
    scale_linetype_discrete(labels = labs1, guide = guide1) +
    scale_x_continuous("Round of adaptivity", breaks=1:10) +  # NEW
    facet_grid(method ~ holdout_reuse, labeller = label_parsed) + # NEW
    theme_bw() +
    theme(legend.position = "bottom", legend.title = element_blank(),
          legend.text.align = 0) + #NEW align left
    xlab("Round of adaptivity") + ylab("Mean AUC +/- 2SE")

ggsave("./img/fig-RSNA-naive_holdout_reuse_vs_thresholdout_AUC.pdf",
       width = 6.0, height = 5.5, units="in") # NEW

# plot for GLMnet only

auc_df %>%
  filter(method == "~\"L1- and L2-regularized GLM\"") %>%
  filter(!(holdout_reuse == "Thresholdout" & dataset == "Test performance")) %>%
  filter(dataset != "Resubstitution", dataset != "Perfect classifier") %>%
  mutate(holdout_reuse = factor(holdout_reuse,
                                levels = c("Naive test data reuse", "Thresholdout"),
                                labels = c(bquote(~"Naive test data reuse"),
                                           bquote(~Thresholdout[AUC])))) %>%
  filter(round > 0) %>%  # NEW
  ggplot() +
    geom_line(aes(round, auc_mean, color = dataset, linetype = dataset)) +
    geom_ribbon(aes(x = round, ymin = auc_mean - 2*auc_sd/sqrt(n_reps),
                    ymax = auc_mean + 2*auc_sd/sqrt(n_reps), fill = dataset),
                alpha = 0.25) +
    scale_color_brewer(palette = "Dark2") +
    scale_fill_brewer(palette = "Dark2") +
    facet_wrap(~holdout_reuse, labeller = label_parsed) +
    theme_bw() +
    theme(legend.title = element_blank()) +
    scale_x_continuous("Round of adaptivity", breaks=1:10) +  # NEW
    ylab("Mean AUC +/- 2SE")

ggsave("./img/fig-RSNA-naive_holdout_reuse_vs_thresholdout_AUC_landscape_GLMnet_only.png", width = 10, height = 5.5, units="in")


#--- Thresholdout budget decrease count

# The budget decrease count is contrasted with the holdout access count of
# the Thresholdout simulation and the naive data reuse simulation

holdout_queries_combined_df <- results %>%
  filter(dataset == "test_auc") %>%
  select(round, holdout_reuse,
         cum_holdout_access_count, cum_budget_decrease_by_round) %>%
  gather(what, value, -round, -holdout_reuse) %>%
  mutate(what = gsub("cum_([a-z]+)_.+$", "\\1", what)) %>%
  group_by(round, holdout_reuse, what) %>%
  summarize(avg_cum_count = mean(value),
            sd_cum_count = sd(value),
            n_reps = n()) %>% tbl_df()

holdout_queries_combined_df <- holdout_queries_combined_df %>%
  filter(!(holdout_reuse == "Naive test data reuse" & what == "budget"))

holdout_queries_combined_df %>%
  filter(round > 0) %>%
  mutate(what = factor(what, levels = c("holdout", "budget"),
                       labels = c("Test data queries", "Budget reduction"))) %>%
  rename("Cumulative count" = what) %>%
  mutate(holdout_reuse = factor(holdout_reuse,
                                levels = c("Naive test data reuse", "Thresholdout"),
                                labels = c(bquote(~"Naive test data reuse"),
                                           bquote(Thresholdout[AUC])))) %>%
  ggplot() +
    geom_line(aes(round, avg_cum_count, linetype = `Cumulative count`,
                  color = `Cumulative count`)) +
    geom_point(aes(round, avg_cum_count, shape = `Cumulative count`,
                   color = `Cumulative count`)) +
    geom_ribbon(aes(x = round, ymin = avg_cum_count - 2*sd_cum_count/sqrt(n_reps),
                    ymax = avg_cum_count + 2*sd_cum_count/sqrt(n_reps),
                    fill = `Cumulative count`),
                alpha = 0.25) +
    scale_x_continuous(breaks = c(1, 10, 20, 30)) +
    scale_color_brewer(palette = "Dark2") +
    scale_fill_brewer(palette = "Dark2") +
    facet_wrap(~holdout_reuse, scales = "free", labeller = label_parsed) +
    xlab("Round of adaptivity") +
    ylab("Mean cumulative test data access count") +
    theme_bw() +
    theme(legend.position = c(0.85, 0.2),
          legend.background = element_rect(size = 0.4, color = "black"))

ggsave("./img/fig-RSNA-naive_holdout_reuse_vs_thresholdout_avg_cum_number_of_holdout_queries_and_budget_reduction_Combined.pdf", width = 5.8, height = 3.0, units = "in")

#--- some budget decrease count statistics

cumsum_inv <- function(vec) {
  return(c(vec[1], diff(vec)))
}

prop_budget_decrease_df <- results %>%
  filter(dataset == "test_auc" & holdout_reuse == "Thresholdout") %>%
  select(round, holdout_access_count, cum_budget_decrease_by_round) %>%
  mutate(budget_decrease_by_round = cumsum_inv(cum_budget_decrease_by_round)) %>%
  mutate(budget_decrease_by_round = ifelse(round == 0, 0,
                                           budget_decrease_by_round)) %>%
  select(-cum_budget_decrease_by_round) %>%
  mutate(prop_budget = budget_decrease_by_round / holdout_access_count) %>%
  filter(round > 0)

mean(prop_budget_decrease_df$prop_budget)
# [1] 0.9283471
sd(prop_budget_decrease_df$prop_budget)
# [1] 0.06685119

prop_budget_decrease_df %>%
  select(-holdout_access_count, -budget_decrease_by_round) %>%
  group_by(round) %>%
  summarize(prop_budget_avg = mean(prop_budget),
            prop_budget_sd = sd(prop_budget)) %>%
  filter(round == 1 | round == 10)
# # A tibble: 2 x 3
#   round prop_budget_avg prop_budget_sd
#   <dbl>           <dbl>          <dbl>
# 1     1           0.920         0.0802
# 2    10           0.911         0.0727

#--- compare the true performance between the naive and ThresholdoutAUC strategies

df_to_plot <- auc_df %>% filter(dataset == "True performance") %>%
  filter(round > 0) %>%  # NEW
  mutate(holdout_reuse = factor(holdout_reuse,
                                levels = c("Naive test data reuse",
                                           "Thresholdout"),
                                labels = c("Naive test data reuse",
                                           "Thresholdout")))

# the following averages over data reuse methods and classiers
overall_auc_mean <- df_to_plot %>%
  filter(round == 1 | round == 10) %>%
  group_by(round) %>%
  summarize(auc = mean(auc_mean))
print.data.frame(overall_auc_mean, digits = 4)
#   round    auc
# 1     1 0.6292
# 2    10 0.6982

# the following averages over data reuse methods
auc_mean_by_method <- df_to_plot %>%
  filter(round == 1 | round == 10) %>%
  group_by(round, method) %>%
  summarize(auc = mean(auc_mean))
print.data.frame(auc_mean_by_method, digits = 4)
# (I permuted the rows here for clarity:)
#   round                        method    auc
#
# 1     1 ~"L1- and L2-regularized GLM" 0.6315
# 3    10 ~"L1- and L2-regularized GLM" 0.6909
#
# 2     1                    ~"XGBoost" 0.6269
# 4    10                    ~"XGBoost" 0.7056
0.7056 - 0.6909
# [1] 0.0147

# the following averages over classifiers
auc_mean_by_holdoutreuse <- df_to_plot %>%
  filter(round == 1 | round == 10) %>%
  group_by(round, holdout_reuse) %>%
  summarize(auc = mean(auc_mean))
print.data.frame(auc_mean_by_holdoutreuse, digits = 4)
# (I permuted the rows here for clarity:)
#   round         holdout_reuse    auc
#
# 1     1 Naive test data reuse 0.6316
# 3    10 Naive test data reuse 0.6981
#
# 2     1          Thresholdout 0.6267
# 4    10          Thresholdout 0.6984
0.6984 - 0.6981
# [1] 3e-04

auc_mean_by_method_holdoutreuse <- df_to_plot %>%
  filter(round == 1 | round == 10) %>%
  select(-dataset, -n_reps)
print.data.frame(auc_mean_by_method_holdoutreuse, digits = 4)
#   round                        method         holdout_reuse  auc_mean  auc_sd
# 1     1 ~"L1- and L2-regularized GLM" Naive test data reuse    0.6321 0.03547
# 2     1 ~"L1- and L2-regularized GLM"          Thresholdout    0.6309 0.03864
# 3     1                    ~"XGBoost" Naive test data reuse    0.6311 0.03749
# 4     1                    ~"XGBoost"          Thresholdout    0.6226 0.04591
# 5    10 ~"L1- and L2-regularized GLM" Naive test data reuse    0.6937 0.02682
# 6    10 ~"L1- and L2-regularized GLM"          Thresholdout    0.6881 0.02701
# 7    10                    ~"XGBoost" Naive test data reuse    0.7025 0.02635
# 8    10                    ~"XGBoost"          Thresholdout    0.7087 0.02775

# (I permuted the rows here for clarity:)
#   round                        method         holdout_reuse     auc
#
# 1     1 ~"L1- and L2-regularized GLM" Naive test data reuse  0.6321
# 5    10 ~"L1- and L2-regularized GLM" Naive test data reuse  0.6937
#
# 2     1 ~"L1- and L2-regularized GLM"          Thresholdout  0.6309
# 6    10 ~"L1- and L2-regularized GLM"          Thresholdout  0.6881
#
# 3     1                    ~"XGBoost" Naive test data reuse  0.6311
# 7    10                    ~"XGBoost" Naive test data reuse  0.7025
#
# 4     1                    ~"XGBoost"          Thresholdout  0.6226
# 8    10                    ~"XGBoost"          Thresholdout  0.7087
#


pal <- brewer.pal(3, "Dark2")
g <- guide_legend(nrow = 2, byrow = TRUE)
df_to_plot %>%
  ggplot() +
    geom_line(aes(round, auc_mean, color = holdout_reuse, linetype = holdout_reuse)) +
    geom_ribbon(aes(x = round, ymin = auc_mean - 2*auc_sd/sqrt(n_reps),
                    ymax = auc_mean + 2*auc_sd/sqrt(n_reps), fill = holdout_reuse), alpha = 0.25) +
    scale_linetype_manual(values = c("Naive test data reuse" = 2,
                                     "Thresholdout" = 4),
                          name = "",
                          labels = c(bquote(~"Naive test data reuse"),
                                     bquote(~Thresholdout[AUC]))) +
    scale_color_manual(values = c("Naive test data reuse" = pal[2],
                                  "Thresholdout" = pal[3]),
                       name = "",
                       labels = c(bquote(~"Naive test data reuse"),
                                  bquote(~Thresholdout[AUC]))) +
    scale_fill_manual(values = c("Naive test data reuse" = pal[2],
                                 "Thresholdout" = pal[3]),
                      name = "",
                      labels = c(bquote(~"Naive test data reuse"),
                                 bquote(~Thresholdout[AUC]))) +
    facet_wrap(~ method, nrow = 1, labeller = label_parsed) +
    theme_bw() +
    theme(legend.position = c(0.84, 0.24),
          legend.background = element_rect(size = 0.4, color = "black"),
          legend.title = element_blank()) +
    guides(color = g, linetype = g) +
    xlab("Round of adaptivity") + ylab("Mean AUC +/- 2SE")

ggsave("./img/fig-RSNA-naive_holdout_reuse_vs_thresholdout_AUC_true_performance.pdf", width = 5.8, height = 3.0, units = "in")
