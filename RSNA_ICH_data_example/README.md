The following steps were used to perform the real data experiments on the publicly available dataset from the [RSNA 2019 Brain CT Hemorrhage Challenge](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection).

All Python code was run in a Docker container environment, whose image is specified by [docker-jupyter/Dockerfile](docker-jupyter/Dockerfile).

0. **Data source**:
    - The data has to be downloaded from <https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection>, and extracted in the directory `RSNA_ICH_data_example/data`.

1. **Image preprocessing**:
    - The Python source code and Jupyter notebooks, which are used for image preprocessing and deep learning feature extraction, are found in the subdirectory `feature_extraction`.
    - Run the Python script [preproc_and_save_2D_256.py](feature_extraction/preproc_and_save_2D_256.py). The resulting preprocessed images will be saved in PNG format in a subdirectory within the directory `data`.
    - (Optional) The preprocessed images can be visually inspected with the Jupyter notebook [visualize_preprocessed_images.ipynb](feature_extraction/visualize_preprocessed_images.ipynb) 

2. Data splitting (by patient ID):
    - The Python source code and Jupyter notebooks, which are used for image preprocessing and deep learning feature extraction, are found in the subdirectory `feature_extraction`.
    - Run the Jupyter notebook [split_by_parient.ipynb](feature_extraction/split_by_parient.ipynb) in order to define many data subsets (to be used in experiments) which do not overlap with respect to image or patient ID.

3. **Deep learning feature extraction** with an ImageNet-trained CNN:
    - The Python source code and Jupyter notebooks, which are used for image preprocessing and deep learning feature extraction, are found in the subdirectory `feature_extraction`.
    - Run the Jupyter notebook [feature_extraction.ipynb](feature_extraction/feature_extraction.ipynb) in order to extract 2048 ResNet50 features for each image. The feature vectors will be saved as rows in CSV files which correspond to different data splits (in the same directory as the preprocessed PNG images).

4. Run the **adaptive data analysis experiments**:
    - These experiments were run on the HPC at FDA/CDRH, making heavy use of array job parallelism. You will have to adjust the `.sh` scripts for your specific HPC environment.
    - (On HPC) In the folder `naive_data_reuse_experiments` execute the scripts `array_job_glmnet.sh` and `array_job_xgboost.sh`.
    - (On HPC) In the folder `thresholdout_experiments` execute the scripts `array_job_glmnet.sh` and `array_job_xgboost.sh`.
    - For additional detail see the [README](../README.md) sections describing the simulation studies, because this part of the real data experiments is very similar to the simulation studies, with the main difference being the use of real medical imaging data instead of simulated data.

5. Visualization of the results
    - The results of individual simulation runs are into large `.csv` files using the scripts in the `naive_data_reuse_experiments/results` and `thresholdout_experiments/results` subdirectories.
    - Subsequently, a number of visualization can be generated from the simulation results using the R scripts included in the subdirectory `visualizations`.
    - The resulting figures can be found in the subdirectory `visualizations/img`.

6. (Optional) Compute upper bounds on the classifier performance achievable in the adaptive data analysis experiments, by training a logistic regression and XGBoost *non-adaptively* on the *combination of all 100 training sets*. Logistic regression achieves AUC of about 0.82 on the combined testing dataset (and the lock-box dataset), and XGBoost an AUC of 0.83. This is not included in the published journal paper in SIMODS for reasons of brevity.
    - Run the Jupyter notebook [classifiers.ipynb](feature_extraction/classifiers.ipynb).
