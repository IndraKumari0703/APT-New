import numpy as np

from .._NearestNeighborsWithClassifierDissimilarity import NearestNeighborsWithClassifierDissimilarity, MetricTensor
from ._OverSampling import OverSampling
from .._base import mode
from ._SMOTE import SMOTE

from .._logger import logger
_logger= logger

__all__= ['Borderline_SMOTE1',
          'Borderline_SMOTE2']


class Borderline_SMOTE1(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{borderlineSMOTE,
                            author="Han, Hui
                            and Wang, Wen-Yuan
                            and Mao, Bing-Huan",
                            editor="Huang, De-Shuang
                            and Zhang, Xiao-Ping
                            and Huang, Guang-Bin",
                            title="Borderline-SMOTE: A New Over-Sampling Method
                                     in Imbalanced Data Sets Learning",
                            booktitle="Advances in Intelligent Computing",
                            year="2005",
                            publisher="Springer Berlin Heidelberg",
                            address="Berlin, Heidelberg",
                            pages="878--887",
                            isbn="978-3-540-31902-3"
                            }
    """

    categories = [OverSampling.cat_sample_ordinary,
                  OverSampling.cat_extensive,
                  OverSampling.cat_borderline,
                  OverSampling.cat_classifier_distance]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 k_neighbors=5,
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
            n_neighbors (int): control parameter of the nearest neighbor
                                    technique for determining the borderline
            k_neighbors (int): control parameter of the nearest neighbor
                                    technique for sampling
            nn_params (dict): additional parameters for nearest neighbor calculations
                                use {'metric': 'precomputed'} for random forest induced
                                metric {'classifier_params': {...}} to set the parameters
                                of the RandomForestClassifier
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_greater_or_equal(k_neighbors, 'k_neighbors', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.k_neighbors = k_neighbors
        self.nn_params= nn_params
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
                                  'n_neighbors': [3, 5, 7],
                                  'k_neighbors': [3, 5, 7]}

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

        # determining number of samples to be generated
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # fitting model
        X_min = X[y == self.min_label]

        metric_tensor = MetricTensor(**self.nn_params).tensor(X, y)
        nn_params = self.nn_params.copy()
        if 'metric_tensor' not in self.nn_params:
            nn_params['metric_tensor']= metric_tensor
        nn_params['metric']='minkowski'

        n_neighbors = min([len(X), self.n_neighbors + 1])
        
        nn= NearestNeighborsWithClassifierDissimilarity(n_neighbors=n_neighbors, 
                                                        n_jobs=self.n_jobs, 
                                                        **(nn_params),
                                                        X=X, 
                                                        y=y)
        nn.fit(X)
        indices= nn.kneighbors(X_min, return_distance=False)

        # determining minority samples in danger
        noise = []
        danger = []
        for i in range(len(indices)):
            if self.n_neighbors == sum(y[indices[i][1:]] == self.maj_label):
                noise.append(i)
            elif mode(y[indices[i][1:]]) == self.maj_label:
                danger.append(i)
        X_danger = X_min[danger]
        X_min = np.delete(X_min, np.array(noise).astype(int), axis=0)

        if len(X_danger) == 0 or len(X_min) < 2:
            _logger.info(self.__class__.__name__ +
                         ": " + "Not enough samples")
            return X.copy(), y.copy()

        # fitting nearest neighbors model to minority samples
        k_neigh = min([len(X_min), self.k_neighbors + 1])
        
        nn= NearestNeighborsWithClassifierDissimilarity(n_neighbors=k_neigh, 
                                                        n_jobs=self.n_jobs, 
                                                        **(nn_params), 
                                                        X=X, 
                                                        y=y)
        nn.fit(X_min)
        indices= nn.kneighbors(X_danger, return_distance=False)

        # generating samples near points in danger
        base_indices = self.random_state.choice(list(range(len(X_danger))),
                                                n_to_sample)
        neighbor_indices = self.random_state.choice(list(range(1, k_neigh)),
                                                    n_to_sample)

        X_base = X_danger[base_indices]
        X_neighbor = X_min[indices[base_indices, neighbor_indices]]
        
        
        mt= nn_params['metric_tensor']
        if mt is None:
            mt= np.ones(len(X[0]))
        else:
            mt= np.diag(mt)/np.max(np.diag(mt))
            mt= 1.0 - mt

        samples= []
        for i in range(n_to_sample):
            samples.append(X_base[i] + self.random_state.rand()*mt*(X_neighbor[i] - X_base[i]))
        samples= np.array(samples)
        """
        samples = X_base + \
            np.multiply(self.random_state.rand(
                n_to_sample, 1), X_neighbor - X_base)
        """
        return (np.vstack([X, samples]),
                np.hstack([y, np.hstack([self.min_label]*n_to_sample)]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'k_neighbors': self.k_neighbors,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class Borderline_SMOTE2(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{borderlineSMOTE,
                            author="Han, Hui
                            and Wang, Wen-Yuan
                            and Mao, Bing-Huan",
                            editor="Huang, De-Shuang
                            and Zhang, Xiao-Ping
                            and Huang, Guang-Bin",
                            title="Borderline-SMOTE: A New Over-Sampling
                                    Method in Imbalanced Data Sets Learning",
                            booktitle="Advances in Intelligent Computing",
                            year="2005",
                            publisher="Springer Berlin Heidelberg",
                            address="Berlin, Heidelberg",
                            pages="878--887",
                            isbn="978-3-540-31902-3"
                            }
    """

    categories = [OverSampling.cat_sample_ordinary,
                  OverSampling.cat_extensive,
                  OverSampling.cat_borderline,
                  OverSampling.cat_classifier_distance]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 k_neighbors=5,
                 *,
                 nn_params={},
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and
                                n_min to sample e.g. 1.0 means that after
                                sampling the number of minority samples
                                will be equal to the number of majority
                                samples
            n_neighbors (int): control parameter of the nearest neighbor
                                technique for determining the borderline
            k_neighbors (int): control parameter of the nearest neighbor
                                technique for sampling
            nn_params (dict): additional parameters for nearest neighbor calculations
                                use {'metric': 'precomputed'} for random forest induced
                                metric {'classifier_params': {...}} to set the parameters
                                of the RandomForestClassifier
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()

        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_greater_or_equal(k_neighbors, 'k_neighbors', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.k_neighbors = k_neighbors
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
                                  'n_neighbors': [3, 5, 7],
                                  'k_neighbors': [3, 5, 7]}
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

        # determining number of samples to be generated
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        metric_tensor = MetricTensor(**self.nn_params).tensor(X, y)
        nn_params = self.nn_params.copy()
        if 'metric_tensor' not in self.nn_params:
            nn_params['metric_tensor']= metric_tensor
        nn_params['metric']='minkowski'

        # fitting nearest neighbors model
        X_min = X[y == self.min_label]

        n_neighbors = min([self.n_neighbors+1, len(X)])
        
        nn= NearestNeighborsWithClassifierDissimilarity(n_neighbors=n_neighbors, 
                                                        n_jobs=self.n_jobs, 
                                                        **(nn_params), 
                                                        X=X, 
                                                        y=y)
        nn.fit(X)
        indices= nn.kneighbors(X_min, return_distance=False)

        # determining minority samples in danger
        noise = []
        danger = []
        for i in range(len(indices)):
            if self.n_neighbors == sum(y[indices[i][1:]] == self.maj_label):
                noise.append(i)
            elif mode(y[indices[i][1:]]) == self.maj_label:
                danger.append(i)
        X_danger = X_min[danger]
        X_min = np.delete(X_min, np.array(noise).astype(int), axis=0)

        if len(X_min) < 2:
            m = ("The number of minority samples after preprocessing (%d) is "
                 "not enough for sampling")
            m = m % (len(X_min))
            _logger.warning(self.__class__.__name__ + ": " + m)
            return X.copy(), y.copy()

        if len(X_danger) == 0:
            m = "No samples in danger"
            _logger.info(self.__class__.__name__ + ": " + m)
            return X.copy(), y.copy()

        # fitting nearest neighbors model to samples
        k_neigh = self.k_neighbors + 1
        k_neigh = min([k_neigh, len(X)])
        
        nn= NearestNeighborsWithClassifierDissimilarity(n_neighbors=k_neigh, 
                                                        n_jobs=self.n_jobs, 
                                                        **(nn_params), 
                                                        X=X, 
                                                        y=y)
        nn.fit(X)
        indices= nn.kneighbors(X_danger, return_distance=False)

        # generating the samples
        base_indices = self.random_state.choice(
            list(range(len(X_danger))), n_to_sample)
        neighbor_indices = self.random_state.choice(
            list(range(1, k_neigh)), n_to_sample)

        X_base = X_danger[base_indices]
        X_neighbor = X[indices[base_indices, neighbor_indices]]
        diff = X_neighbor - X_base
        r = self.random_state.rand(n_to_sample, 1)
        mask = y[neighbor_indices] == self.maj_label
        r[mask] = r[mask]*0.5

        
        mt= nn_params['metric_tensor']
        if mt is None:
            mt= np.ones(len(X[0]))
        else:
            mt= np.diag(mt)/np.max(np.diag(mt))
            mt= 1.0 - mt

        samples= []
        for i in range(n_to_sample):
            samples.append(X_base[i] + r[i]*mt*(X_neighbor[i] - X_base[i]))
        samples= np.array(samples)
        """

        samples = X_base + np.multiply(r, diff)
        """
        return (np.vstack([X, samples]),
                np.hstack([y, np.hstack([self.min_label]*n_to_sample)]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'k_neighbors': self.k_neighbors,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
