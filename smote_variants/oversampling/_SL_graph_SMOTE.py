import numpy as np

from scipy.stats import skew

from .._NearestNeighborsWithClassifierDissimilarity import NearestNeighborsWithClassifierDissimilarity, MetricTensor
from ._OverSampling import OverSampling
from ._SMOTE import SMOTE
from ._Safe_Level_SMOTE import Safe_Level_SMOTE
from ._Borderline_SMOTE import Borderline_SMOTE1

from .._logger import logger
_logger= logger

__all__= ['SL_graph_SMOTE']

class SL_graph_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @inproceedings{sl_graph_smote,
                    author = {Bunkhumpornpat,
                        Chumpol and Subpaiboonkit, Sitthichoke},
                    booktitle= {13th International Symposium on Communications
                                and Information Technologies},
                    year = {2013},
                    month = {09},
                    pages = {570-575},
                    title = {Safe level graph for synthetic minority
                                over-sampling techniques},
                    isbn = {978-1-4673-5578-0}
                    }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_borderline,
                  OverSampling.cat_classifier_distance]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                    to sample e.g. 1.0 means that after
                                    sampling the number of minority samples
                                    will be equal to the number of majority
                                    samples
            n_neighbors (int): number of neighbors in nearest neighbors
                                component
            nn_params (dict): additional parameters for nearest neighbor calculations
                                use {'metric': 'precomputed'} for random forest induced
                                metric {'classifier_params': {...}} to set the parameters
                                of the RandomForestClassifier
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # Fitting nearest neighbors model
        n_neighbors = min([len(X), self.n_neighbors])
        
        nn_params= self.nn_params.copy()
        if not 'metric_tensor' in self.nn_params:
            metric_tensor = MetricTensor(**self.nn_params).tensor(X, y)
            nn_params['metric_tensor']= metric_tensor
        
        nn= NearestNeighborsWithClassifierDissimilarity(n_neighbors=n_neighbors, 
                                                        n_jobs=self.n_jobs, 
                                                        **nn_params, 
                                                        X=X, 
                                                        y=y)
        nn.fit(X)
        indices = nn.kneighbors(X[y == self.min_label], return_distance=False)

        # Computing safe level values
        safe_level_values = np.array(
            [np.sum(y[i] == self.min_label) for i in indices])

        # Computing skewness
        skewness = skew(safe_level_values)

        if skewness < 0:
            # left skewed
            s = Safe_Level_SMOTE(proportion=self.proportion,
                                 n_neighbors=self.n_neighbors,
                                 nn_params=nn_params,
                                 n_jobs=self.n_jobs,
                                 random_state=self.random_state)
        else:
            # right skewed
            s = Borderline_SMOTE1(proportion=self.proportion,
                                  n_neighbors=self.n_neighbors,
                                  nn_params=nn_params,
                                  n_jobs=self.n_jobs,
                                  random_state=self.random_state)

        return s.sample(X, y)

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
