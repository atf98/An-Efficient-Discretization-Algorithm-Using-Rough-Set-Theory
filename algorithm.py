import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score


class Error(Exception):
    """Base class for other exceptions"""
    pass


class NotMatchResult(Error):
    """Raised when the input value is too small"""
    pass


class RST:
    def __init__(
        self,
        data: pd.DataFrame = None,
        continuous_columns: list = [],
        decision_column_name: str = 'class',
    ):

        # Main variable of RST class
        self.data = data
        self.data_indiscernibility = {}
        self.continuous_columns = continuous_columns
        self.decision_column_name = decision_column_name

        # Main variable of RST process
        self.lower_approx = {}
        self.upper_approx = {}
        self.boundary_region = {}
        self.outside_region = {}
        self.U = self._list_to_set_conversion(self.data.index.values.tolist())

        # Config Variable
        self.target_sets = self.get_indiscernibility([decision_column_name])
        self.target_unique = self.data[decision_column_name].unique()
        self.target_sets_dict = {self.target_unique[idx]: ele for idx, ele in enumerate(
            self.target_sets[decision_column_name])}

    def get_indiscernibility(self, combination: list = []):

        def indices(lst, item):
            return [i for i, x in enumerate(lst) if x == item]

        selected_columns = self.data[combination].to_dict('list')
        data = list(zip(*selected_columns.values()))
        self.data_indiscernibility.update({
            self._create_names(combination): list(indices(data, x) for x in set(data) if data.count(x) > 1)
        })
        return self.data_indiscernibility

    def set_main_variable(self, combination: list = []):
        _c_n = self._create_names(combination)  # combination name
        combination_upper = {}
        combination_lower = {}

        for target, set in self.target_sets_dict.items():
            if _c_n not in self.data_indiscernibility:
                self.get_indiscernibility(combination)

            c = combination.copy()
            c.append(str(target))
            __c_n = self._create_names(c)

            self.lower_approx.update({__c_n: []})
            self.upper_approx.update({__c_n: []})
            self.boundary_region.update({__c_n: []})
            self.outside_region.update({__c_n: []})

            for IND in self.data_indiscernibility[_c_n]:
                if all(True if x in set else False for x in IND):
                    self.lower_approx[__c_n].extend(IND)

                if any(True if x in set else False for x in IND):
                    self.upper_approx[__c_n].extend(IND)

            combination_lower.update({__c_n: self.lower_approx[__c_n]})
            combination_upper.update({__c_n: self.upper_approx[__c_n]})

            upper = self._list_to_set_conversion(self.upper_approx[__c_n])
            lower = self._list_to_set_conversion(self.lower_approx[__c_n])

            self.boundary_region[__c_n].extend(
                upper - lower
            )

            self.outside_region[__c_n].extend(
                self.U - upper
            )

        return [
            combination_upper,
            combination_lower,
            self.boundary_region[__c_n],
            self.outside_region[__c_n]
        ]

    def get_dependency(self, combination: list = []):
        _, lower, _, _ = self.set_main_variable(combination)
        divider = sum([len(v) for v in lower.values()])
        dependency = divider / len(self.U)
        return [
            lower,
            dependency
        ]

    def _create_names(self, e) -> str:
        return '_'.join(e)

    def _list_to_set_conversion(self, l):
        se = set()
        for x in l:
            se.add(x)
        return se


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

    def __init__(self, checker_type: str = 'topN'):
        # Primary Data Information
        self.data = None
        self.columns = None
        self.continuous_columns = None
        self.discrete_columns = None
        self.scaled_data = None

        # Secondary Data infromation
        self.data_after_INI = None
        self.silhouette_scores = None

        # check_discrete
        self.dis_check_threshold = 0.5
        self.dis_checker = checker_type  # ratio
        self.checker_top_n = 10  # defult number of top N for topN checker

        # Config Tools
        self.scaler = StandardScaler()
        self.natural_interval_model = 'dbscan'
        self.NIM_message = 'Model for initiat the natural intervals cannot be blank'

    def fit(
            self,
            x: pd.DataFrame,
            continous_columns: list = [],
            natural_interval_model: str = 'kmeans'
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
        self.data = x
        self.columns = self.data.columns
        self.continuous_columns = continous_columns if continous_columns else self._check_continuous()
        self.discrete_columns = list(
            set(self.columns).difference(self.continuous_columns))
        self.scaled_data = self._scaling_continuous()
        self.silhouette_scores = self._get_silhouette_score()
        print(self.silhouette_scores)

        # Intiate classification
        self.data_after_INI = getattr(
            self, '_%s_model' % natural_interval_model, lambda: self.NIM_message)()
        print(self.data_after_INI.nunique())
        self.data_after_INI.sort_values(['Annual_Premium']).to_csv(
            '%s_model.csv' % natural_interval_model)
        return self.data_after_INI

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
            clustering = KMeans(n_clusters=self.silhouette_scores[att]).fit(
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
            if self.dis_checker == 'topN':
                # Check if the top n unique values account for more than a certain proportion of all values
                if 1.*self.data[var].value_counts(normalize=True).head(top_n).sum() < 0.5:
                    likely.append(var)
            elif self.dis_checker == 'ratio':
                # Find the ratio of number of unique values to the total number of unique values. Something like the following
                if 1.*self.data[var].nunique()/self.data[var].count() > 0.5:
                    likely.append(var)

        common_names = ['id']

        return [x for x in likely if x.lower() not in common_names]

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
