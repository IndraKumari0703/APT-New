import numpy as np

from .._NearestNeighborsWithClassifierDissimilarity import NearestNeighborsWithClassifierDissimilarity, MetricTensor
from ._OverSampling import OverSampling
from ..noise_removal import TomekLinkRemoval
from ._SMOTE import SMOTE

from .._logger import logger
_logger= logger

__all__= ['SMOTE_TomekLinks']

class SMOTE_TomekLinks(OverSampling):
    """
    References:
        * BibTex::

            @article{smote_tomeklinks_enn,
                     author = {Batista, Gustavo E. A. P. A. and Prati,
                                Ronaldo C. and Monard, Maria Carolina},
                     title = {A Study of the Behavior of Several Methods for
                                Balancing Machine Learning Training Data},
                     journal = {SIGKDD Explor. Newsl.},
                     issue_date = {June 2004},
                     volume = {6},
                     number = {1},
                     month = jun,
                     year = {2004},
                     issn = {1931-0145},
                     pages = {20--29},
                     numpages = {10},
                     url = {http://doi.acm.org/10.1145/1007730.1007735},
                     doi = {10.1145/1007730.1007735},
                     acmid = {1007735},
                     publisher = {ACM},
                     address = {New York, NY, USA},
                    }
    """

    categories = [OverSampling.cat_sample_ordinary,
                  OverSampling.cat_noise_removal,
                  OverSampling.cat_changes_majority,
                  OverSampling.cat_classifier_distance]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the SMOTE object

        Args:
            proportion (float): proportion of the difference of n_maj and
                                n_min to sample e.g. 1.0 means that after
                                sampling the number of minority samples
                                will be equal to the number of majority
                                samples
            n_neighbors (int): control parameter of the nearest neighbor
                                technique
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
        return SMOTE.parameter_combinations(raw)

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

        nn_params= self.nn_params.copy()
        if not 'metric_tensor' in self.nn_params:
            metric_tensor = MetricTensor(**self.nn_params).tensor(X, y)
            nn_params['metric_tensor']= metric_tensor

        smote = SMOTE(self.proportion,
                      self.n_neighbors,
                      nn_params=nn_params,
                      n_jobs=self.n_jobs,
                      random_state=self.random_state)
        X_new, y_new = smote.sample(X, y)

        t = TomekLinkRemoval(strategy='remove_both', 
                             n_jobs=self.n_jobs,
                             nn_params=nn_params)

        X_samp, y_samp = t.remove_noise(X_new, y_new)

        if len(X_samp) == 0:
            m = ("All samples have been removed, "
                 "returning the original dataset.")
            _logger.info(self.__class__.__name__ + ": " + m)
            return X.copy(), y.copy()

        return X_samp, y_samp

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
