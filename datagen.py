import numpy as np
from matplotlib import pyplot as plt


def double_mickey(minority_var=0.05, majority_var=0.2,
                  majority_size=1000, minority_size=30, seed=None):

    state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)

    dist1 = np.random.multivariate_normal((-1, 0), majority_var * np.eye(2), size=majority_size)
    dist2 = np.random.multivariate_normal((1, 0), majority_var * np.eye(2), size=majority_size)
    dist3 = np.random.multivariate_normal((-2, 1), minority_var * np.eye(2), size=minority_size)
    dist4 = np.random.multivariate_normal((-2, -1), minority_var * np.eye(2), size=minority_size)
    dist5 = np.random.multivariate_normal((2, 1), minority_var * np.eye(2), size=minority_size)
    dist6 = np.random.multivariate_normal((2, -1), minority_var * np.eye(2), size=minority_size)
    ys = np.concatenate([
        np.repeat(1, dist1.shape[0]),
        np.repeat(2, dist2.shape[0]),
        np.repeat(3, dist3.shape[0]),
        np.repeat(4, dist4.shape[0]),
        np.repeat(5, dist5.shape[0]),
        np.repeat(6, dist6.shape[0]),
    ])
    xs = np.vstack([dist1, dist2, dist3, dist4, dist5, dist6])
    np.random.set_state(state)
    return xs, ys


def main():
    xs, ys = double_mickey()
    for y in np.unique(ys):
        dist = xs[ys == y]
        plt.scatter(dist[:, 0], dist[:, 1])
    plt.show()


if __name__ == '__main__':
    main()
