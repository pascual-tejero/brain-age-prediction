import glob
from utils.data_utils import load_segmentations, load_nii
from utils.plot_utils import plot_segmentations
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer, max_error


def main():
    # Load segmentations
    paths = sorted(glob.glob('./data/brainage-data/segs_refs/*'))
    filenames, segmentations = load_segmentations(paths)

    # Plot segmentations
    im = load_nii(f'./data/brainage-data/images/sub-{filenames[1]}_T1w_unbiased.nii.gz')
    plot_segmentations(im, segmentations[1], i=40)

    # Load labels
    df_labels = pd.read_csv('data/brainage-data/meta/meta_data_all.csv')
    labels = list(df_labels[df_labels['subject_id'].isin(filenames)].age)
    assert list(df_labels[df_labels['subject_id'].isin(filenames)]['subject_id']) == filenames

    # Extract features from segmentations (number of voxels per tag)
    count_tags = np.zeros((segmentations.shape[0],4))
    count_tags[:,0] = np.count_nonzero(segmentations == 1, axis=(1,2,3))
    count_tags[:,1] = np.count_nonzero(segmentations == 2, axis=(1,2,3))
    count_tags[:,2] = np.count_nonzero(segmentations == 3, axis=(1,2,3))
    count_tags[:,3] = count_tags[:,0] + count_tags[:,1] + count_tags[:,2]
    print("Number of voxels per tag (column) and per patient (row): ", count_tags)

    # Extract features from segmentations (volume per tag)
    X = count_tags
    y = np.array(labels)
    print("\n")
    print("="*120)

    model1 = linear_model.Lasso()
    score1 = cross_validation(model1, X, y, folds = 5)
    print(f"{score1.mean()} accuracy with a standard deviation of {score1.std()} for the model {model1}")
    print(score1, "\n")

    model2 = linear_model.Ridge(alpha=.5)
    score2 = cross_validation(model2, X, y, folds = 5)
    print(f"{score2.mean()} accuracy with a standard deviation of {score2.std()} for the model {model2}")
    print(score2, "\n")

    model3 = linear_model.SGDRegressor()
    score3 = cross_validation(model3, X, y, folds = 5)
    print(f"{score3.mean()} accuracy with a standard deviation of {score3.std()} for the model {model3}")
    print(score3, "\n")

    model4 = linear_model.ElasticNetCV()
    score4 = cross_validation(model4, X, y, folds = 5)
    print(f"{score4.mean()} accuracy with a standard deviation of {score4.std()} for the model {model4}")
    print(score4, "\n")

    model5 = linear_model.BayesianRidge()
    score5 = cross_validation(model5, X, y, folds = 5)
    print(f"{score5.mean()} accuracy with a standard deviation of {score5.std()} for the model {model5}")
    print(score5, "\n")
    print("="*120)

def cross_validation(model, X, y, folds = 5):
    """Cross validation function"""
    scoring_method1 = make_scorer(mean_squared_error)
    score = cross_val_score(model, X, y, cv=folds, scoring=scoring_method1)

    return score


if __name__ == '__main__':
    main()