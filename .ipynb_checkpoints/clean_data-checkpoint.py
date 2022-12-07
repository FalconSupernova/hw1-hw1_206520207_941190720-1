# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
pd.options.mode.chained_assignment = None

def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg

rm_ext_and_nan will remove the extra feature `DR` (ignore the feature), and **all** non-numeric values (ignore the samples). 
Notice that real data can have missing values in many different ways and thus your implementation has to be generic. 
This function should return a dictionary of features where the values of each feature are the clean excel columns 
without the `DR` feature. **Hint**: In order to eliminate every cell that is non-numeric, you will have to transform 
it first to NaN and only then eliminate them.

Implement the function in a single line of code using dictionary comprehensions.

"""
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    CTG_copy = CTG_features.copy()
    # removing the DR feature
    if extra_feature in CTG_copy.columns:
        CTG_copy.drop([extra_feature], inplace = True, axis = 1)
    
    for i in CTG_copy.columns:
        for j in CTG_copy.index:
            if not isinstance(CTG_copy[i][j], (int, float)):
                CTG_copy.replace(CTG_copy[i][j], np.nan)
    c_ctg = {y: [x for x in CTG_copy[y] if not pd.isna(x)] for y in CTG_copy.columns if y != extra_feature}
    # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas dataframe of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe c_cdf containing the "clean" features
    """

    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_cdf = CTG_features.copy()

    # removing the DR feature
    if extra_feature in c_cdf.columns:
        c_cdf.drop([extra_feature], inplace = True, axis = 1)

    # cleaning the data and replacing with NaN
    for i in range(0,len(c_cdf.columns)):
        c_cdf[c_cdf.columns[i]]= pd.to_numeric(c_cdf[c_cdf.columns[i]], errors='coerce')
    
    # sampling the rows with NaN values (random values)  
    for i in range(0, len(c_cdf.columns)):
        for j in range(0, len(c_cdf)):
            if math.isnan(c_cdf.iloc[j,i]) == True:
                rand_idx = np.random.choice(len(c_cdf))
                c_cdf.iloc[j,i] = c_cdf.iloc[rand_idx,i]
    # -------------------------------------------------------------------------

    return c_cdf





def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_samp
    :return: Summary statistics as a dictionary of dictionaries (called d_summary) as explained in the notebook

    c_feat is the dataframe object
    We need to calculate the min, Q1, median, Q3, max of each column of data frame.
    In order to do so we will create a for loop and calculuate these values for each coulumn and store them in dictionary of dictionaries.

    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    bp = c_feat.boxplot(return_type = 'dict', figsize=(70, 70))
    medians = [item.get_ydata()[0] for item in bp['medians']]
    """to obtain the minimum values for all the boxplots, we have to use list slicing to get the odd ndarrays 
    (and selecting the first item from each), to obtain the maximum values — the even ones."""
    minimums = [item.get_ydata()[0] for item in bp['caps']][::2] 
    maximums = [item.get_ydata()[0] for item in bp['caps']][1::2]
    q1 = [min(item.get_ydata()) for item in bp['boxes']]
    q3 = [max(item.get_ydata()) for item in bp['boxes']]
    d_summary = {}
    values = ['min','Q1','median','Q3','max']
    for i in range(0,len(c_feat.columns)):
        d_summary[c_feat.columns[i]] = {}
        d_summary[c_feat.columns[i]][values[0]] = minimums[i]
        d_summary[c_feat.columns[i]][values[1]] = q1[i]
        d_summary[c_feat.columns[i]][values[2]] = medians[i]
        d_summary[c_feat.columns[i]][values[3]] = q3[i]
        d_summary[c_feat.columns[i]][values[4]] = maximums[i]
    # -------------------------------------------------------------------------

    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_samp
    :param d_summary: Output of sum_stat
    :return: Dataframe containing c_feat with outliers removed
    """
    c_no_outlier = c_feat.copy()
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    for i in range(0,len(c_no_outlier.columns)):
        max1 = d_summary[c_no_outlier.columns[i]]['max']
        min1 = d_summary[c_no_outlier.columns[i]]['min']
        for j in range(0, len(c_no_outlier)):
            if (c_no_outlier.iloc[j][i] >= max1) or (c_no_outlier.iloc[j][i] <= min1):
                c_no_outlier.iloc[j][i] = np.nan
    # -------------------------------------------------------------------------

    return c_no_outlier


def phys_prior(c_samp, feature, thresh):
    """

    :param c_samp: Output of nan2num_samp
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    mean = c_samp.apply(np.mean)
    columns = c_samp.columns
    i = columns.get_loc(feature)

    filt_feature = []

    for j in range(0,len(c_samp)):
        if c_samp.iloc[j][i] > thresh or c_samp.iloc[j][i] < 0:
                filt_feature.append(mean[i]) 
        else:
            filt_feature.append(c_samp.iloc[j][i]) 
    # -------------------------------------------------------------------------

    return np.array(filt_feature)


class NSD:

    def __init__(self):
        self.max = np.nan
        self.min = np.nan
        self.mean = np.nan
        self.std = np.nan
        self.fit_called = False
    
    def fit(self, CTG_features):
        # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
        self.mean = CTG_features.apply(np.mean)
        self.max = CTG_features.apply(np.max)
        self.min = CTG_features.apply(np.min)
        self.std = CTG_features.std()
        # -------------------------------------------------------------------------

        self.fit_called = True

    def transform(self, CTG_features, mode='none', selected_feat=('LB', 'ASTV'), flag=False):
        """
        Note: x_lbl should only be either: 'Original values [N.U]', 'Standardized values [N.U.]', 'Normalized values [N.U.]' or 'Mean normalized values [N.U.]'
        :param mode: A string determining the mode according to the notebook
        :param selected_feat: A two elements tuple of strings of the features for comparison
        :param flag: A boolean determining whether or not plot a histogram
        :return: Dataframe of the normalized/standardized features called nsd_res
        """
        ctg_features = CTG_features.copy()
        if self.fit_called:
            if mode == 'none':
                nsd_res = ctg_features
                x_lbl = 'Original values [N.U]'
            # ------------------ IMPLEMENT YOUR CODE HERE (for the remaining 3 methods using elif):------------------------------
            elif mode == 'standard':
                nsd_res = ctg_features
                for i in range(0,len(ctg_features.columns)):
                    for j in range(0,len(ctg_features)):
                        nsd_res.iloc[j,i] = ((nsd_res.iloc[j,i] - self.mean[i])/self.std[i])

                x_lbl = 'Standardized values [N.U.]'

            elif mode == 'MinMax':
                nsd_res = ctg_features
                for i in range(0,len(ctg_features.columns)):
                    a = (self.max[i]-self.min[i])
                    for j in range(0,len(ctg_features)):
                        nsd_res.iloc[j,i] = ((nsd_res.iloc[j,i] - self.min[i])/a)
                x_lbl = 'Normalized values [N.U.]'

            elif mode == 'mean':        
                nsd_res = ctg_features
                for i in range(0,len(ctg_features.columns)):
                    b = (self.max[i]-self.min[i])
                    for j in range(0,len(ctg_features)):
                        nsd_res.iloc[j,i] = ((nsd_res.iloc[j,i] - self.mean[i])/b)
                x_lbl = 'Mean normalized values [N.U.]'
            # -------------------------------------------------------------------------

            if flag:
                self.plot_hist(nsd_res, mode, selected_feat, x_lbl)
            return nsd_res
        else:
            raise Exception('Object must be fitted first!')

    def fit_transform(self, CTG_features, mode='none', selected_feat=('LB', 'ASTV'), flag=False):
        self.fit(CTG_features)
        return self.transform(CTG_features, mode=mode, selected_feat=selected_feat, flag=flag)

    def plot_hist(self, nsd_res, mode, selected_feat, x_lbl):
        x, y = selected_feat
        if mode == 'none':
            bins = 50
            plt.hist(nsd_res[x], color='blue', edgecolor='black', bins=bins)
            plt.hist(nsd_res[y], color='red', edgecolor='black', bins=bins)
            plt.xlabel(x_lbl)
        else:
            bins = 80
            # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
            plt.hist(nsd_res[x], color='blue', edgecolor='black', bins=bins)
            plt.xlabel(x_lbl)
            plt.hist(nsd_res[y], color='red', edgecolor='black', bins=bins)
            plt.xlabel(x_lbl)
            # -------------------------------------------------------------------------


# Debugging block!
if __name__ == '__main__':
    from pathlib import Path
    file = Path.cwd().joinpath(
        'messed_CTG.xls')  # concatenates messed_CTG.xls to the current folder that should be the extracted zip folder
    CTG_dataset = pd.read_excel(file, sheet_name='Raw Data')
    CTG_features = CTG_dataset[['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DR', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV',
                                'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance',
                                'Tendency']]
    CTG_morph = CTG_dataset[['CLASS']]
    fetal_state = CTG_dataset[['NSP']]

    extra_feature = 'DR'
    c_ctg = rm_ext_and_nan(CTG_features, extra_feature)