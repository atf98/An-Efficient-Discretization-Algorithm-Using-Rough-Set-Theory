from tooldev import RST
from tooldev.sample import bootstrap

from itertools import combinations

import os
import json
import numpy as np
import pandas as pd
import collections
from pathlib import Path


from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KernelDensity

import matplotlib.pyplot as plt


class Error(Exception):
    """Base class for other exceptions"""
    pass


class NotMatchResult(Error):
    """Raised when the input value is too small"""
    pass


class DRST:
    '''
        An Efficient Discretization Algorithm Using Rough Set Theory

        ...

        Attributes
        ----------
        checker_type : str='topN'
            Type of the checker of data to determind wether data column is Continous or Discrete

        Methods
        -------
        fit(x: DataFrame, continous_columns: list)

        Note
        -------

        Example
        -------


    '''

    def __init__(
        self,
        comb_max_depth: int = None,
        decision_column_name: str = 'class',
        output_loction=None,
        save_output=True,
        checker_type: str = 'topN',
        topN_thrshold: str = 0.5,
        ratio_thrshold: str = 0.5,
    ):

        # Primary Data Information
        self.data = None
        self.prepared_data = None
        self.columns = None
        self.continuous_columns = None
        self.discrete_columns = None
        self.scaled_data = None
        self.decision_column_name = decision_column_name

        # Secondary Data infromation
        self.data_after_INI = None
        self.silhouette_scores = None
        self.lower_edge_value = {}
        self.upper_edge_value = {}
        self.outer_edge_value = {}
        
        # check_discrete
        self.dis_check_threshold = 0.5
        self.dis_checker = checker_type  # ratio
        self.topN_thrshold = topN_thrshold  # topN_thrshold
        self.ratio_thrshold = ratio_thrshold  # ratio_thrshold
        self.checker_top_n = 10  # defult number of top N for topN checker

        # Config Tools
        self.rst = None
        self.encoder = LabelEncoder()
        self.encoder_dict = dict()
        self.scaler = StandardScaler()
        self.natural_interval_model = 'kmeans'
        self.NIM_message = 'Model for initiat the natural intervals cannot be blank'
        self.comb_max_depth = comb_max_depth
        self.all_combinations = None
        
        self.output_loction = os.getcwd() if not output_loction else os.getcwd() + '\\' + output_loction
        self.save_output = save_output

    def fit(
            self,
            x: pd.DataFrame,
            continous_columns: list = [],
            natural_interval_model: str = 'kmeans',
            ensamble_threshold: int = 0.9
    ):
        """Discretization using Rough Sets Theory (RST).

        Fit `x` to an efficient intervals using the concept of RST 
        * asd
        Parameters
        ----------
        x : DataFrame
            Full DataFrame (default is None)

        continous_columns : list
            Determind names of column(s) in the DataFarme as Continuous data. If `None` a function _check_continuous triger to determind names of column(s).

        Raises
        ------
        NotMatchResult
            If continous_columns `~None` model check the input with the checker determation if didn't match the result

        """

        # Reduct Data Intervals phase: Pre-process of discretization, intiate classification
        # Pre-process of discretization
        self.ensamble_theshold = ensamble_threshold
        self.data = x
        self.columns = self.data.columns
        self.continuous_columns = continous_columns if continous_columns else self._check_continuous()
        self.discrete_columns = list(set(self.columns).difference(self.continuous_columns))
        self.discrete_columns.remove(self.decision_column_name)
        self.scaled_data = self._scaling_continuous()
        self.silhouette_scores = self._get_silhouette_score()

        # Intiate classification
        self.data_after_INI = getattr(self, '_%s_model' % natural_interval_model, lambda: self.NIM_message)()
        self.all_combinations = self.get_combinations()
        print(self.data_after_INI.nunique())

        data = self.data.copy()  # local data vriable to save change in
        data = self._labeling_discrete(data)
        self.data_after_INI.columns = map((lambda x: x + '_AFTER'), self.data_after_INI.columns)

        data = pd.concat([
            data.reset_index(drop=True),
            self.data_after_INI.reset_index(drop=True)
        ], axis=1)

        dependency_original_dict = {}
        dependency_dict = {}
        
        # Duplicate Continuous columns
        for continuous_att in self.continuous_columns:
            att_AFTER = continuous_att + '_AFTER'
            label_simulator = self.silhouette_scores[continuous_att]

            self.prepared_data = data.copy()
            self.prepared_data.sort_values([continuous_att], inplace=True, ignore_index=True)
            temp_data = self.prepared_data.copy()
            for n_label, o_label in enumerate(temp_data[att_AFTER].unique()):
                indices_change_label = temp_data[temp_data[att_AFTER] == o_label].index.tolist()
                self.prepared_data.loc[indices_change_label, (att_AFTER)] = n_label
            self.prepared_data.sort_values([att_AFTER, continuous_att], inplace=True)

            

            value = 0
            upper_flag = True
            lower_flag = True
            value_flag = True
            value_sufx = ''

            while value < label_simulator:
                if value_flag:
                    upper_flag = True
                    lower_flag = True
                    value_sufx = ''
                indices = self.prepared_data[self.prepared_data[att_AFTER] == value].index.tolist()
                if not len(indices) == 0:
                    value_dict = '%s%s' % (value, value_sufx)
                    if value not in dependency_dict.keys():
                        dependency_dict.update({value_dict: {}})
                        if upper_flag:
                            dependency_dict[value_dict].update({
                                'upper': {True: 0, False: 0}
                            })

                        if lower_flag:
                            dependency_dict[value_dict].update({
                                'lower': {True: 0, False: 0}
                            })

                    for combination in self.all_combinations:
                        _combination = '|'.join(i for i in combination)
                        _combination_AFTER = list(list(combination) + [att_AFTER])

                        self.lower_data = self.prepared_data.copy()
                        self.upper_data = self.prepared_data.copy()
                        
                        self.rst = RST(
                            self.prepared_data,
                            continuous_columns=self.continuous_columns,
                            decision_column_name=self.decision_column_name,
                            include_continuous_col=True
                        )
                        _, dependency = self.rst.get_dependency(combination=list(_combination_AFTER))
                        dependency_original_dict.update({_combination: dependency})
                        
                        
                        self._set_lower_edge_value(continuous_att, indices, value, label_simulator)
                        self._set_upper_edge_value(continuous_att, indices, value, label_simulator)
                        
                        if value not in [0] and lower_flag:
                            self.lower_data.loc[self.lower_edge_value[value]['idx'], (att_AFTER)] = value - 1
                            self.rst = RST(
                                self.lower_data,
                                continuous_columns=self.continuous_columns,
                                decision_column_name=self.decision_column_name,
                                include_continuous_col=True
                            )
                            _, dependency = self.rst.get_dependency(
                                combination=list(_combination_AFTER)
                            )
                            dependency_dict[value_dict]['lower'][dependency >= dependency_original_dict[_combination]] += 1
                    
                        if value not in [label_simulator - 1] and upper_flag:
                            self.upper_data.loc[self.upper_edge_value[value]['idx'], (att_AFTER)] = value + 1
                            self.rst = RST(
                                self.upper_data,
                                continuous_columns=self.continuous_columns,
                                decision_column_name=self.decision_column_name,
                                include_continuous_col=True
                            )
                            lower, dependency = self.rst.get_dependency(
                                combination=list(_combination_AFTER)
                            )
                            dependency_dict[value_dict]['upper'][dependency >= dependency_original_dict[_combination]] += 1

                    if lower_flag:
                        is_lower = len(self.all_combinations)*self.ensamble_theshold <= dependency_dict[value_dict]['lower'][True]
                    if upper_flag:
                        is_upper = len(self.all_combinations)*self.ensamble_theshold <= dependency_dict[value_dict]['upper'][True]

                    if is_lower or is_upper:
                        lower_flag=is_lower
                        upper_flag=is_upper
                        value_flag=False
                        value_sufx += '-'
                        if is_lower:
                            self.prepared_data.loc[self.lower_edge_value[value]['idx'], (att_AFTER)] = value - 1
                            value_sufx += 'l'
                        if is_upper:
                            self.prepared_data.loc[self.upper_edge_value[value]['idx'], (att_AFTER)] = value + 1
                            value_sufx += 'u'
                    else:
                        value += 1
                        value_flag = True
                else:
                    value += 1
                    value_flag = True

        dependency_dict = collections.OrderedDict(sorted(dependency_dict.items()))
        if self.save_output:
            Path(self.output_loction + '\\_log').mkdir(parents=True, exist_ok=True)
            with open(self.output_loction + '\\_log\\logMap.json', 'w+') as fp:
                json.dump(dependency_dict, fp)
                
            self.prepared_data.to_csv(self.output_loction + '\\DataAfterDRST.csv')
            data.to_csv(self.output_loction + '\\DataBeforeDRST.csv')
        return [self.prepared_data, self.continuous_columns]

    def _dbscan_model(self):
        result = {}
        for att in self.continuous_columns:
            clustering = DBSCAN(eps=1, min_samples=5).fit(
                self._np_array_reshaped(self.data[att])
            )
            result[att] = clustering.labels_
        return pd.DataFrame(result)

    def _kmeans_model(self):
        result = {}
        for att in self.continuous_columns:
            clustering = KMeans(n_clusters=self.silhouette_scores[att], random_state=41).fit(
                self._np_array_reshaped(self.scaled_data[att])
            )
            result[att] = clustering.labels_
        return pd.DataFrame(result)

    def _np_array_reshaped(self, data, reshape=(-1, 1)):
        return np.array(data.tolist()).reshape(reshape[0], reshape[1])

    def _check_continuous(self, continous_columns: list = []):
        top_n = self.checker_top_n
        likely = []
        # check if attribute is continuous or discrete dataframe
        for var in self.data.columns:
            if self.data[var].dtype in [np.int64, np.float64]:
                if self.dis_checker == 'topN':
                    # Check if the top n unique values account for more than a certain proportion of all values
                    if 1.*self.data[var].value_counts(normalize=True).head(top_n).sum() < self.topN_thrshold:
                        likely.append(var)
                elif self.dis_checker == 'ratio':
                    # Find the ratio of number of unique values to the total number of unique values. Something like the following
                    if 1.*self.data[var].nunique()/self.data[var].count() > self.ratio_thrshold:
                        likely.append(var)

        common_names = ['id']

        return [x for x in likely if x.lower() not in common_names]

    def _labeling_discrete(self, data):

        data_endocded = data.copy()

        for col in data_endocded:
            if col not in self.continuous_columns:
                le = self.encoder.fit(data_endocded[col])
                data_endocded[col] = self.encoder.transform(data_endocded[col])
                self.encoder_dict[col] = self.encoder
        return data_endocded

    def _scaling_continuous(self):
        data_cluster = self.data[self.continuous_columns].copy()
        scaled_columns = self.scaler.fit_transform(data_cluster)
        self.scaled_data = self.data.copy()
        self.scaled_data[self.continuous_columns] = scaled_columns

        return self.scaled_data

    def _get_silhouette_score(self):
        scores = {}
        for att in self.continuous_columns:
            temp_score = []
            scores[att] = 3
            for n_clusters in range(3, 9):
                # Initialize the clusterer with n_clusters value and a random generator
                # seed of 10 for reproducibility.
                clusterer = KMeans(n_clusters=n_clusters, random_state=10)
                cluster_labels = clusterer.fit_predict(
                    self._np_array_reshaped(self.scaled_data[att]))

                # The silhouette_score gives the average value for all the samples.
                # This gives a perspective into the density and separation of the formed
                # clusters
                silhouette_avg = silhouette_score(
                    self._np_array_reshaped(self.scaled_data[att]), cluster_labels)
                temp_score.append(silhouette_avg)
            scores[att] = temp_score.index(max(temp_score)) + 3
        return scores

    def get_combinations(self, comb_max_depth: int = None):
        if not comb_max_depth:
            comb_max_depth = self.comb_max_depth
        combX = []
        for idx, _ in enumerate(self.discrete_columns):
            if comb_max_depth and comb_max_depth == idx:
                break
            combX.extend(list(combinations(self.discrete_columns, idx + 1)))

        return sorted(combX, key=len)
    
    def _set_lower_edge_value(self, att, indices, value, s_c):
        same_value = self.prepared_data.loc[indices][att].values.tolist()
        lower_same_value = self.prepared_data[att].isin([same_value[0]])
        lower_temp = self.prepared_data[lower_same_value]
        self.lower_edge_value.update({
            value: {
                'idx': lower_temp.index.tolist(),
                'value': lower_temp.values.tolist()[0][0],
            }
        })

        return self.lower_edge_value

    def _set_upper_edge_value(self, att, indices, value, s_c):
        same_value = self.prepared_data.loc[indices][att].values.tolist()
        upper_same_value = self.prepared_data[att].isin([same_value[-1]])
        upper_temp = self.prepared_data[upper_same_value]
        self.upper_edge_value.update({
            value: {
                'idx': upper_temp.index.tolist(),
                'value': upper_temp.values.tolist()[0][0],
            }
        })

        return self.upper_edge_value