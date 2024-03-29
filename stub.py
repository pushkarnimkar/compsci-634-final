from datagen import double_mickey
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from modAL import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from scipy import spatial
import abc
import coresets
from sklearn.exceptions import NotFittedError
from tqdm.auto import tqdm
from algorithms import WeightedKMeans


def random_sampling(classifier, X_pool, n_instances=10):
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples), size=n_instances, replace=False)
    return query_idx


class SamplingMethod(abc.ABC):
    @abc.abstractmethod
    def sample(self, classifier, x_pool, n_instances=1):
        pass


class GreedyHittingSetSampling(SamplingMethod):
    def __init__(self, x_train, mask_sampled):
        self._dist_mat = spatial.distance_matrix(x_train, x_train)
        self._indices = np.arange(x_train.shape[0])
        self._mask_sampled = mask_sampled

    def sample(self, classifier, x_pool, n_instances=1):
        mask_sampled_old = self._mask_sampled.copy()

        indices = []
        for b in range(n_instances):
            if not self._mask_sampled.any():
                idx = np.random.choice(self._indices)
            else:
                min_dist = (self._dist_mat[self._mask_sampled]
                            [:, ~self._mask_sampled].min(axis=0))
                idx = self._indices[~self._mask_sampled][np.argmax(min_dist)]
            self._mask_sampled[idx] = True

            # We only need the following to maintain the query indices in sync
            # with the pool indices as modAL requires it so.
            # This does not contribute to the main logic as the data fed to the
            # learner is selected from self._mask_sampled, shared array
            indices.append(idx - mask_sampled_old[:idx].sum())
        return np.array(indices)


class KMeansCoresetSampling(SamplingMethod):
    @staticmethod
    def plot_kmeans(kmc: coresets.KMeansCoreset, n_instances: int,
                    x_pool: np.ndarray, classifier):

        cs, ws = kmc.generate_coreset(n_instances)
        km = WeightedKMeans()
        km.fit(cs, ws)
        x_min, x_max = x_pool[:, 0].min(), x_pool[:, 0].max()
        y_min, y_max = x_pool[:, 1].min(), x_pool[:, 1].max()
        xs = np.linspace(1.1 * x_min, 1.1 * x_max, 300)
        ys = np.linspace(1.1 * y_min, 1.1 * y_max, 300)
        xx, yy = np.meshgrid(xs, ys)
        Z, _ = km.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1, figsize=(8, 6))
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5)

        try:
            proba = classifier.predict_proba(x_pool)
            uncertainty = stats.entropy(proba, axis=1)
            plt.scatter(x_pool[:, 0], x_pool[:, 1], s=80 * (0.1 + uncertainty),
                        color='black', alpha=0.25, label='Uncertainty')
        except NotFittedError:
            pass

        plt.scatter(cs[:, 0], cs[:, 1],
                    marker='x', s=100, color='black', label='Selected samples')
        plt.legend()
        plt.title('k-Means Coreset samples')
        plt.show()

    def sample(self, classifier, x_pool, n_instances=1):
        try:
            proba = classifier.predict_proba(x_pool)
            uncertainty = stats.entropy(proba, axis=1)
        except NotFittedError:
            uncertainty = np.ones(x_pool.shape[0])
        adjusted = 0.1 + 0.9 * uncertainty
        kmc = coresets.KMeansCoreset(x_pool, n_clusters=8, w=adjusted)
        # self.plot_kmeans(kmc, n_instances, x_pool, classifier)
        return kmc.coreset_indices(n_instances)


def plot_sampled(xs, ys, mask_sampled, method):
    for y in np.unique(ys):
        dist = xs[ys == y]
        plt.scatter(dist[:, 0], dist[:, 1])

    x_known = xs[mask_sampled]
    plt.scatter(x_known[:, 0], x_known[:, 1], marker='x', color='black')
    plt.title(f'Points sampled with {method} method')
    plt.show()


def train_learner(method, x_train, y_train, x_test, y_test, estimator=None):
    mask_sampled = np.zeros_like(y_train, dtype=bool)
    indices = np.arange(mask_sampled.shape[0])

    if estimator is None:
        estimator = LogisticRegression(C=np.inf, max_iter=1000)

    if method == 'greedy_hitting_set':
        sampler = GreedyHittingSetSampling(x_train, mask_sampled)
        learner = ActiveLearner(
            estimator=estimator, query_strategy=sampler.sample
        )
    elif method == 'kmeans_coreset':
        sampler = KMeansCoresetSampling()
        learner = ActiveLearner(
            estimator=estimator, query_strategy=sampler.sample
        )
    elif method == 'random':
        learner = ActiveLearner(estimator=estimator, query_strategy=random_sampling)
    else:
        learner = ActiveLearner(
            estimator=estimator, query_strategy=uncertainty_sampling
        )

    batch_size, progress = 24, []
    for batch in range(6):
        x_pool, pool_idx = x_train[~mask_sampled], indices[~mask_sampled]
        query_idx, _ = learner.query(x_pool, n_instances=batch_size)
        mask_sampled[pool_idx[query_idx]] = True
        learner.teach(x_train[mask_sampled], y_train[mask_sampled])

        y_proba = learner.predict_proba(x_test)
        unique_y_sampled = np.unique(y_train[mask_sampled])
        unique_y_train = np.unique(y_train)

        if unique_y_train.shape[0] != unique_y_sampled.shape[0]:
            cols = []
            for i in unique_y_train:
                if i in unique_y_sampled:
                    loc = np.where(i == unique_y_sampled)[0]
                    col = y_proba[:, loc]
                else:
                    col = np.zeros(y_proba.shape[0]).reshape(-1, 1)
                cols.append(col)
            y_proba = np.hstack(cols)

        y_pred = np.argmax(y_proba, axis=1) + 1
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        accuracy = accuracy_score(y_test, y_pred)
        progress.append((accuracy, roc_auc, unique_y_sampled.shape[0]))

    # plot_sampled(x_train, y_train, mask_sampled, method)
    return learner, mask_sampled, np.array(progress)


def main():
    xs, ys = double_mickey(seed=1000, majority_var=0.16, minority_var=0.04)
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, train_size=0.8, stratify=ys)

    (progress_kmeans_coreset, progress_random,
     progress_greedy_hitting_set, progress_uncertainty) = [], [], [], []
    for _ in tqdm(range(100)):
        learner, mask_sampled, progress_kmeans_coreset_ = train_learner(
            'kmeans_coreset', x_train, y_train, x_test, y_test
        )
        progress_kmeans_coreset.append(progress_kmeans_coreset_)
        learner, mask_sampled, progress_greedy_hitting_set_ = train_learner(
            'greedy_hitting_set', x_train, y_train, x_test, y_test
        )
        progress_greedy_hitting_set.append(progress_greedy_hitting_set_)
        learner, mask_sampled, progress_uncertainty_ = train_learner(
            'uncertainty', x_train, y_train, x_test, y_test
        )
        progress_uncertainty.append(progress_uncertainty_)
        learner, mask_sampled, progress_random_ = train_learner(
            'random', x_train, y_train, x_test, y_test
        )
        progress_random.append(progress_random_)

    # accuracy, roc_auc, unique_y_sampled.shape[0]
    progress_kmeans_coreset = np.mean(progress_kmeans_coreset, axis=0)
    progress_greedy_hitting_set = np.mean(progress_greedy_hitting_set, axis=0)
    progress_uncertainty = np.mean(progress_uncertainty, axis=0)
    progress_random = np.mean(progress_random, axis=0)

    bs = np.arange(24, 168, 24)

    plt.plot(bs, progress_greedy_hitting_set[:, 0],
             label='Hitting Set', marker='o')
    plt.plot(bs, progress_kmeans_coreset[:, 0], label='k-Means Coreset', marker='o')
    plt.plot(bs, progress_uncertainty[:, 0], label='Uncertainty', marker='o')
    plt.plot(bs, progress_random[:, 0], label='Random', marker='o')
    plt.title('Comparison of active sampling methods')
    plt.xlabel('Number of labels queried')
    plt.ylabel('Accuracy score')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(bs, progress_greedy_hitting_set[:, 1],
             label='Hitting Set', marker='o')
    plt.plot(bs, progress_kmeans_coreset[:, 1], label='k-Means Coreset', marker='o')
    plt.plot(bs, progress_uncertainty[:, 1], label='Uncertainty', marker='o')
    plt.plot(bs, progress_random[:, 1], label='Random', marker='o')
    plt.title('Comparison of active sampling methods')
    plt.xlabel('Number of labels queried')
    plt.ylabel('ROC AUC score')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(bs, progress_greedy_hitting_set[:, 2],
             label='Hitting Set', marker='o')
    plt.plot(bs, progress_kmeans_coreset[:, 2], label='k-Means Coreset', marker='o')
    plt.plot(bs, progress_uncertainty[:, 2], label='Uncertainty', marker='o')
    plt.plot(bs, progress_random[:, 2], label='Random', marker='o')
    plt.title('Comparison of active sampling methods')
    plt.xlabel('Number of labels queried')
    plt.ylabel('Average classes discovered')
    plt.legend()
    plt.grid()
    plt.show()

    # for y in np.unique(ys):
    #     dist = xs[ys == y]
    #     proba = learner.predict_proba(dist)
    #     uncertainty = stats.entropy(proba, axis=1)
    #     plt.scatter(dist[:, 0], dist[:, 1], s=80 * (0.1 + uncertainty))
    #
    # y_proba = learner.predict_proba(xs)
    # y_pred = learner.predict(xs)
    # incorrect = xs[ys != y_pred]
    # plt.scatter(incorrect[:, 0], incorrect[:, 1], facecolor='none', edgecolors='black', s=89)
    #
    # x_known = x_train[mask_sampled]
    # plt.scatter(x_known[:, 0], x_known[:, 1], marker='x', color='black')
    # plt.show()
    #
    # print(roc_auc_score(ys, y_proba, multi_class='ovr'))


if __name__ == '__main__':
    main()
