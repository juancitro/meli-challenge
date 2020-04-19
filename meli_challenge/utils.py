from sklearn.utils import class_weight as sk_class_weight
import pandas as pd
import numpy as np

# "The sample weighting rescales the C parameter in cost function formula,
# which means that the classifier puts more emphasis on getting these points right."
def build_sample_weight(df, label_data, quality_data, factor=4):
    # df['sample_weight'] = sk_class_weight.compute_sample_weight(class_weight='balanced', y=df[label_data].values)
    # df['sample_weight'] = np.where(df[quality_data] == 'reliable',
    #                                        df['sample_weight'] * factor,
    #                                        df['sample_weight'])
    # return df
    np_sample_weight = sk_class_weight.compute_sample_weight(class_weight='balanced', y=df[label_data].values)
    np_sample_weight = np.where(df[quality_data] == 'reliable',
                                   np_sample_weight * factor,
                                   np_sample_weight)
    return np_sample_weight


if __name__ == '__main__':
    FILE = '../data/train_spanish_cleaned_lemmatized.csv'
    df = pd.read_csv(FILE)
    df = build_sample_weight(df, 'category', 'label_quality', 4)
