The following steps were used to perform the real data experiments on the RSNA ICH dataset.
All python code was run in a Docker container environment, whose image is specified by `docker-jupyter/Dockerfile`.

0. **Data source**:
  - The data has to be downloaded from <https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection>, and extracted in the directory `data`.

1. **Image preprocessing**:
  - The source code and Jupyter notebooks, which are used for image preprocessing and deep learning feature extraction, are found in the folder `feature_extraction`.
  - Run the Jupyter notebook `preproc_and_save_2D_256.ipynb`. The resulting preprocessed images will be saved in PNG format in a subdirectory within the directory `data`.
  - (Optional) The preprocessed images can be visually inspected with the Jupyter notebook `visualize_preprocessed_images.ipynb` 

2. Data splitting (by patient ID):
  - The source code and Jupyter notebooks, which are used for image preprocessing and deep learning feature extraction, are found in the folder `feature_extraction`.
  - Run the Jupyter notebook `split_by_parient.ipynb` in order to define many data subsets which do not overlap with respect to image or patient ID.

3. **Deep learning feature extraction** with an ImageNet-trained CNN:
  - The source code and Jupyter notebooks, which are used for image preprocessing and deep learning feature extraction, are found in the folder `feature_extraction`.
  - Run the Jupyter notebook `feature_extraction.ipynb` in order to extract 2048 ResNet50 features for each image. The feature vectors will be saved as rows in CSV files which correspond to different data splits (in the same directory as the preprocessed PNG images).

4. Run the **adaptive data analysis experiments**:
  - These experiments were run on the HPC at FDA/CDRH, making heavy use of array job parallelism.
  - (On HPC) In the folder `naive_data_reuse_experiments` execute the scripts `array_job_glmnet.sh` and `array_job_xgboost.sh`.
  - (On HPC) In the folder `thresholdout_experiments` execute the scripts `array_job_glmnet.sh` and `array_job_xgboost.sh`.

4. (Optional) Compute upper bounds on the classifier performance achievable in the adaptive data analysis experiments, by training a logistic regression and XGBoost *non-adaptively* on the *combination of all 100 training sets*. Logistic regression achieves AUC of about 0.82 on the combined testing dataset (and the lock-box dataset), and XGBoost an AUC of 0.83. This is not included in the SIMODS paper draft for reasons of brevity.
  - Run the Jupyter notebook `classifiers.ipynb`.
