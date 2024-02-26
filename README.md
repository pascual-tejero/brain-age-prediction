# Brain Age Prediction

This repository contains the practical task project for the course of AI in Medicine I at the Technical University of Munich (TUM). The main objective of this project is to implement two distinct supervised learning approaches for age regression using brain MRI data.

## Project Overview

In this project, we aim to predict the age of individuals based on their brain MRI scans. We are provided with a dataset comprising 652 brain MRIs, each accompanied by the corresponding age of the subject. The ultimate goal is to develop and compare the performance of two supervised learning models for age regression.

## Dataset

The dataset consists of 652 brain MRI images along with the age of each subject. This data serves as the foundation for training and evaluating our machine learning models. Access to the dataset is provided within this repository for academic and research purposes. 

## Approach
To gain insights into our dataset, we conducted exploratory data analysis. This script provides essential information about the population statistics, including gender distribution, age distribution, and provides an example MRI image. All these information can be seen in `./results/data_familiarisation`.

- Gender distribution: We analyzed the gender distribution within our dataset to understand the representation of different genders among the subjects.
- Age distribution: Exploring the age distribution allowed us to observe the range and distribution of ages present in the dataset.
- MRI image as example: The script also presents an example MRI image from the dataset, providing a visual representation of the brain scans used in our analysis.

We explore two different supervised learning approaches for age regression:

1. **Feature-based linear regression using brain structure segmentation**: In this approach, we used various linear regression models available in the `sklearn` library, including Lasso, Ridge, SGDRegressor, ElasticNetCV, and BayesianRidge. The objective is to predict age by utilizing cross-validation to assess model performance. 
   - **Feature Extraction**: Initially, we extract features from brain segmentations, which serve as input for our machine learning models. Specifically, we quantify the number of voxels (3D pixels) corresponding to different brain regions or tags and calculate the volume per tag.
   - **Cross-Validation and Evaluation**: Subsequently, we conduct cross-validation for each model, partitioning the dataset into five folds. During each iteration, the model is trained on four folds and evaluated on the remaining fold. Mean Squared Error (MSE) is computed as the evaluation metric, providing insights into the model's predictive accuracy.
   - **Performance Analysis**: We report the mean accuracy and standard deviation of MSE for each model across the folds. This analysis offers a comprehensive view of the models' performance and enables us to compare their effectiveness in predicting age from brain MRI data.


2. **Image-based brain age regression using CNNs**: In this second approach, we employ Convolutional Neural Networks (CNNs) to directly regress age from brain MRIs. The visualization below depicts the results on test data, with red representing the true age and blue indicating the predicted age.
![Brain age regression on test data](https://github.com/pascutc98/brain-age-prediction/blob/main/results/age_regression_cnn/plot_results.png)

This CNN-based method offers an alternative approach to age regression, leveraging the power of deep learning models to directly analyze image data and predict age.




## Instructions 

1. Clone this repository to your local machine:
```bash
git clone https://github.com/pascutc98/brain-age-prediction/tree/main
cd brain-age-prediction
```

2. Create and activate a conda environment:
```bash
conda create -n img_class python=3.8
conda activate img_class
```

3. Install the required dependencies by using the provided requirements.txt file:
```bash
pip install -r requirements.txt
```

4. Run the different Python scripts:
```bash
# Data familiarisation on MRI data
python data_familiarisation.py

# Feature-based linear brain age regression using brain structure segmentation
python age_regression_seg_features.py

# Image-based brain age regression using CNNs
python age_regression_cnn.py
```

## Conclusion

Through this project, we aim to demonstrate the application of supervised learning techniques in the domain of medical image analysis, specifically in predicting age from brain MRI data. The insights gained from this endeavor contribute to the broader field of AI in Medicine, with potential implications for diagnostic and prognostic applications.





