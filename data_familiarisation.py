import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
import numpy as np

def data_familiarisation(data_path, save_path):
    # Read the meta data using pandas
    meta_data_all = pd.read_csv(data_path)
    meta_data_all.head()  
    print(meta_data_all.head())  

    # Get population statistics
    get_population_statistics(meta_data_all, save_path)

    # Get image example
    get_image_example(save_path)


def get_population_statistics(meta_data_all, save_path):
    # Get population statistics
    sns.catplot(x="gender_text", data=meta_data_all, kind="count") 
    plt.title('Gender distribution')
    plt.xlabel('Gender')
    plt.ylabel('Frequency')
    plt.savefig(save_path + "/gender_distribution.png")
    # plt.savefig('./results/data_familiaristation/gender_distribution.png')
    plt.show() 
    plt.close()

    sns.distplot(meta_data_all['age'], bins=[10, 20, 30, 40, 50, 60, 70, 80, 90])
    plt.title('Age distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig(save_path + "/age_distribution.png")
    plt.show()
    plt.close() 

    plt.scatter(range(len(meta_data_all['age'])), meta_data_all['age'], marker='.')
    plt.grid()
    plt.xlabel('Subject')
    plt.ylabel('Age')
    plt.title('Age distribution')
    plt.savefig(save_path + "/age_distribution_scatter.png")
    plt.show()
    plt.close()

def get_image_example(save_path):
    file = './data/brainage-data/images/sub-CC110033_T1w_unbiased.nii.gz'

    image = nib.load(file).get_fdata()
    print(image.shape)

    f, axarr = plt.subplots(1, 3)
    H, W, D = image.shape
    axarr[0].imshow(np.flip(image[H // 2, :, :].T, axis=0), cmap='gray') # x is the height (AP view)
    axarr[1].imshow(np.flip(image[:, W // 2, :].T, axis=0), cmap='gray') # y is the width (AP view)
    axarr[2].imshow(image[:, :, D // 2].T, cmap='gray')
    plt.savefig(save_path + "/image_example.png")
    plt.show()
    plt.close()

if __name__ == '__main__':
    data_path = './data/brainage-data/meta/meta_data_all.csv'
    save_path = './results/data_familiaristation'

    data_familiarisation(data_path, save_path)