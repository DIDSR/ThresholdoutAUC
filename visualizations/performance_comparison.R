library(tidyverse)
library(RColorBrewer)

# load simulation parameters (such as number of cases, number of features, etc.)
source("../set_params.R")
source("../functions/thresholdout_auc.R")

thresholdout_csv <- "../results/Cypress_Tulane_HPC/cypress_thresholdout_all_results.csv"
naive_csv <- "../results/Cypress_Tulane_HPC/cypress_naive_all_results.csv"
oracle_csv <- "../results/Cypress_Tulane_HPC/cypress_oracle_all_results.csv"

thresholdout <- read_csv(thresholdout_csv)
naive_holdout <- read_csv(naive_csv)
oracle_holdout <- read_csv(oracle_csv)

naive_holdout <- mutate(naive_holdout, holdout_reuse = "Naive test data reuse")
thresholdout <- mutate(thresholdout, holdout_reuse = "Thresholdout")
oracle_holdout <- mutate(oracle_holdout, holdout_reuse = "Oracle")

results <- bind_rows(naive_holdout, thresholdout)
results_with_oracle <- bind_rows(naive_holdout, thresholdout, oracle_holdout)

#--- look at the average AUCs across replications

auc_df <- results %>%
  select(-n_train, -num_features, -holdout_access_count) %>%
  group_by(round, dataset, method, holdout_reuse) %>%
  summarize(auc_mean = mean(auc), auc_sd = sd(auc), n_reps = n()) %>% tbl_df()

auc_df <- auc_df %>%
  mutate(method = factor(method,
                         levels = c("glm", "glmnet", "svmLinear2",
                                    "rf", "adaboost"),
                         labels = c(bquote(~"Logistic regression (GLM)"),
                                    bquote(~"L1- and L2-regularized GLM"),
                                    bquote(~"SVM with linear kernel"),
                                    bquote(~"Random Forest"),
                                    bquote(~"AdaBoost")))) %>%
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

auc_df %>%
  filter(!(holdout_reuse == "Thresholdout" & dataset == "Test performance")) %>%
  filter(dataset != "Resubstitution", dataset != "Perfect classifier") %>%
  mutate(holdout_reuse = factor(holdout_reuse,
                                levels = c("Naive test data reuse", "Thresholdout"),
                                labels = c(bquote(~"Naive test data reuse"),
                                           bquote(~Thresholdout[AUC])))) %>%
  ggplot() +
    geom_line(aes(round, auc_mean, color = dataset, linetype = dataset)) +
    geom_ribbon(aes(x = round, ymin = auc_mean - 2*auc_sd/sqrt(n_reps),
                    ymax = auc_mean + 2*auc_sd/sqrt(n_reps), fill = dataset),
                alpha = 0.25) +
    scale_color_brewer(palette = "Dark2") +
    scale_fill_brewer(palette = "Dark2") +
    facet_grid(holdout_reuse ~ method, labeller = label_parsed) +
    theme_bw() +
    theme(legend.position = "bottom", legend.title = element_blank()) +
    xlab("Round of adaptivity") + ylab("Mean AUC +/- 2SE")

ggsave("./img/naive_holdout_reuse_vs_thresholdout_AUC_landscape.pdf",
       width = 11, height = 5.5, units="in")

#--- Maximal achievable AUC based on how the data is generated

perfect_auc_df <- results %>%
  select(-n_train, -num_features, -holdout_access_count) %>%
  filter(dataset == "perfect_auc") %>%
  group_by(replication) %>%
  summarize(auc = unique(auc))
mean(perfect_auc_df$auc)
# [1] 0.9633105
sd(perfect_auc_df$auc)
# [1] 0.01555172

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

ggsave("./img/naive_holdout_reuse_vs_thresholdout_avg_cum_number_of_holdout_queries_and_budget_reduction_Combined.pdf", width = 5.8, height = 3.0, units = "in")

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
# [1] 0.7269255
sd(prop_budget_decrease_df$prop_budget)
# [1] 0.2752433

prop_budget_decrease_df %>%
  select(-holdout_access_count, -budget_decrease_by_round) %>%
  group_by(round) %>%
  summarize(prop_budget_avg = mean(prop_budget),
            prop_budget_sd = sd(prop_budget)) %>%
  filter(round == 1 | round == 30)
# # A tibble: 2 x 3
#   round prop_budget_avg prop_budget_sd
#   <int>           <dbl>          <dbl>
# 1     1           0.861          0.112
# 2    30           0.623          0.312


#--- look at the average AUCs across replications, including the "oracle" results

# The "oracle" simulation describes the case where we know a priori which
# small subset of variables are predictive of the outcome of interest.
# Of course this method is unimplementable in practice.

auc_df <- results_with_oracle %>%
  select(-n_train, -num_features, -holdout_access_count,
         -cum_holdout_access_count, -cum_holdout_access_count) %>%
  group_by(round, dataset, method, holdout_reuse) %>%
  summarize(auc_mean = mean(auc), auc_sd = sd(auc), n_reps = n()) %>% tbl_df()

auc_df <- auc_df %>%
  mutate(method = factor(method,
                         levels = c("glm",
                                    "glmnet", "svmLinear2",
                                    "rf", "adaboost"),
                         labels = c("Logistic regression (GLM)",
                                    "L1- and L2-regularized GLM",
                                    "SVM with linear kernel",
                                    "Random Forest",
                                    "AdaBoost"))) %>%
  # keep only the 10-times repeated 5-fold CV AUC on train data (though the unrepeated 5-fold CV AUC measures are essentially the same, only slightly more overestimating).
  filter(dataset != "cv_auc") %>%
  mutate(dataset = factor(dataset,
                          levels = c("repeatedcv_auc", "train_auc",
                                     "holdout_auc", "thresholdout_auc",
                                     "test_auc", "perfect_auc"),
                          labels = c("Training performance", "Resubstitution",
                                     "Test performance", "Thresholdout output",
                                     "True performance", "Perfect classifier"),
                          ordered = TRUE))

#--- compare the true performance between the naive, ThresholdoutAUC, and the "oracle" simulations

df_to_plot <- auc_df %>% filter(dataset == "True performance") %>%
  filter(method == "Logistic regression (GLM)" | method == "AdaBoost") %>%
  mutate(holdout_reuse = factor(holdout_reuse,
                                levels = c("Oracle",
                                           "Naive test data reuse",
                                           "Thresholdout"),
                                labels = c("Oracle",
                                           "Naive test data reuse",
                                           "Thresholdout")))

pal <- brewer.pal(3, "Dark2")
g <- guide_legend(nrow = 3, byrow = TRUE)
df_to_plot %>%
  ggplot() +
    geom_line(aes(round, auc_mean, color = holdout_reuse, linetype = holdout_reuse)) +
    geom_ribbon(aes(x = round, ymin = auc_mean - 2*auc_sd/sqrt(n_reps),
                    ymax = auc_mean + 2*auc_sd/sqrt(n_reps), fill = holdout_reuse), alpha = 0.25) +
    scale_linetype_manual(values = c("Oracle" = 1,
                                     "Naive test data reuse" = 2,
                                     "Thresholdout" = 4),
                          name = "",
                          labels = c(bquote(~"Oracle model (with\na priori knowledge)"),
                                     bquote(~"Naive test data reuse"),
                                     bquote(~Thresholdout[AUC]))) +
    scale_color_manual(values = c("Oracle" = pal[1],
                                  "Naive test data reuse" = pal[2],
                                  "Thresholdout" = pal[3]),
                       name = "",
                       labels = c(bquote(~"Oracle model (with\na priori knowledge)"),
                                  bquote(~"Naive test data reuse"),
                                  bquote(~Thresholdout[AUC]))) +
    scale_fill_manual(values = c("Oracle" = pal[1],
                                 "Naive test data reuse" = pal[2],
                                 "Thresholdout" = pal[3]),
                      name = "",
                      labels = c(bquote(~"Oracle model (with\na priori knowledge)"),
                                 bquote(~"Naive test data reuse"),
                                 bquote(~Thresholdout[AUC]))) +
    facet_wrap(~ method, nrow = 1) +
    theme_bw() +
    theme(legend.position = c(0.84, 0.24),
          legend.background = element_rect(size = 0.4, color = "black"),
          legend.title = element_blank()) +
    guides(color = g, linetype = g) +
    xlab("Round of adaptivity") + ylab("Mean AUC +/- 2SE")

ggsave("./img/naive_holdout_reuse_vs_thresholdout_AUC_true_performance_Combined_with_oracle_simulation_GLM_and_AdaBoost_only.pdf", width = 5.8, height = 3.0, units = "in")
