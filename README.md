# ThresholdoutAUC

The ThresholdoutAUC procedure (based on [the Thresholdout or reusable holdout method of Dwork et al., 2015](https://arxiv.org/abs/1506.02629)) allows the repeated evaluation of the area under the receiver operating characteristic curve (AUC) using a single fixed dataset for the performance assessment of continuously evolving classification algorithms.

This repository includes source code that can be used to perform extensive simulation studies, which show that ThresholdoutAUC substantially reduces the problem of overfitting to the test data under the simulation conditions, at the cost of a mild additional uncertainty on the reported test performance.

In the following we provide a general overview of the code contained in this repository.

## The simulations

There are three general types of simulations corresponding to three ways in which the test dataset is reused. These are organized in the following three directories:

1. `naive_data_reuse_simulation`: the same test dataset is reused for each model evaluation, and the best model is selected based on the AUC values on the test data in each round of adaptive learning.
2. `thresholdout_simulation`: the same test dataset is reused for each model evaluation, but the evaluation metrics run through the ThresholdoutAUC procedure; the best model is selected based on the ThresholdoutAUC estimates of test data AUC in each round of adaptive learning. This simulation can be run with either Gaussian noise or Laplace noise applied within the ThresholdoutAUC procedure.
3. `oracle_classifier_simulation`: the small subset of truly predictive variables is known a priori, and thus, the best model is known a priori and no model selection needs to be performed. While the "oracle" is unimplementable in practice, it gives an upper bound on the maximal performance that can be achieved with a specific classification method.

Directories 1, 2, 3 are referred to as "simulation directories" in the following.

## How to change parameter settings before running the simulations

Before the simulations are run, many simulation settings (such as sample size, number of features, number of rounds of adaptive learning, ThresholdoutAUC hyperparamenters, etc.) can be set/changed by modifying `set_params.R` file.

Moreover, the file paths where the simulation results should be saved can be specified at the bottom of the `holdout_reuse.R` script within each simulation directory.
The coded naming conventions of all saved files are based on environment variables provided by the SLURM HPC workload manager (see below). If you are not using SLURM please replace the respective variables (all starting with `SLURM_`) with something suitable for your system, in order to save the results at a desired location with unique file names.

## Running simulations on a high performance computing cluster

We ran all simulations on the Tulane University high performance computing system Cypress (https://wiki.hpc.tulane.edu/trac/wiki/cypress/about).
Cypress uses SLURM to manage job submission (for documentation, see the SLURM website: http://slurm.schedmd.com/documentation.html).
Within each simulation directory we provide the `.srun` scripts that we used to submit hundreds of simulation runs to SLURM as job arrays.
For example, 100 runs of the "naive data reuse" simulations can be performed with:

```
sbatch ./naive_data_reuse_simulation/holdout_reuse.srun
sbatch ./naive_data_reuse_simulation/holdout_reuse_long_jobs.srun
```

It should not be difficult to adjust the code for using an HPC job scheduling system other than SLURM, since each of the `.srun` scripts simply repeatedly (100 times) submits the same `holdout_reuse.R` script with a different random number generator seed.
Please let us know under the Issues tab of this repository if you require assistance.

## Performing a single simulation run on your local machine

When a simulation script, named `holdout_reuse.R` in each of the three simulation directories, is run, a single simulation run is performed.
A single simulation run includes training five types of classification algorithms (logistics regression, regularized logistic regression, linear SVM, random forest, and AdaBoost) in an adaptive fashion through 30 rounds of adaptive learning using a single fixed test dataset.
This code can be run interactively in R.

The results of a single simulation run are saved to a `.csv` file using the file path specified at the bottom of the respective `holdout_reuse.R` script.
The coded naming conventions of all saved files are based on environment variables provided by the SLURM HPC workload manager (e.g., the job number within the job array). If you are not using SLURM please replace the respective variables (all starting with `SLURM_`) with something suitable for your system to save the results at the desired location.

## Visualization of the results

The results of individual simulation runs are combined into a large `.csv` file using the scripts in the `results` directory.

Subsequently, a number of visualization can be generated from the simulation results using the R scripts included in the directory `visualizations`.

The resulting figures can be found in the directory `visualizations/img`.
