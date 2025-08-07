import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

from visualization import plot_ellipsoid


def loading_data_example():
    import numpy as np

    arr = np.load("dti_data_subset.npz")
    X = arr["X"]
    Y = arr["Y"]

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)


def mallasto_example():
    data = scipy.io.loadmat("./DTI_data.mat")

    # Save independent variables as X
    X = data["t_orig"].T

    # Dependent as Y
    Y = data["pop_orig"].T

    X = X[::4, :]
    Y = Y[::4, :]
    n = len(X)

    # Scale independent variables for nicer plots
    t_plot = 0.05 * np.hstack([X, np.zeros((n, 1))])
    # t_plot = np.hstack([X, np.zeros((n, 1))])

    diagonal = t_plot[::4][:-3]
    Y_diag = Y[::4][:-3]

    new_x = diagonal[:, 0]
    new_y = Y_diag
    np.savez("dti_data_subset.npz", X=new_x, Y=new_y)

    fig = plt.figure()
    _ = LightSource(270, 45)
    ax = fig.add_subplot(111, projection="3d")
    for t, y in zip(t_plot, Y):
        plot_ellipsoid(ax, t, y, scale=0.025)

    ax.set_zlim(-0.2, 0.2)
    plt.show()


def geodesic_regression_on_subset_using_geomstats():
    from geomstats.geometry.spd_matrices import SPDMatrices, SPDLogEuclideanMetric
    from geomstats.learning.geodesic_regression import GeodesicRegression

    arr = np.load("dti_data_subset.npz")
    X = arr["X"]
    Y = arr["Y"]

    spd_matrices = SPDMatrices(3, equip=False)
    spd_matrices.equip_with_metric(SPDLogEuclideanMetric)
    geodesic_regression = GeodesicRegression(
        space=spd_matrices, initialization="data", method="riemannian"
    )
    # geodesic_regression.fit(torch.from_numpy(X), torch.from_numpy(Y))

    print("Geodesic regression coefficients:", geodesic_regression)


def geodesic_regression_on_subset_using_manifold():
    from manifold import PD

    arr = np.load("dti_data_subset.npz")
    X = arr["X"]
    Y = arr["Y"]

    pd_mani = PD(3, "LogEuclidean")
    geodesic_regression = pd_mani.tangent_space_regression(X, Y)

    unseen_x = np.linspace(-1.0, 1.0, 20)
    pred = geodesic_regression(unseen_x)

    t_plot = np.zeros((len(X), 3))
    t_plot[:, 0] = X

    t_pred = np.zeros((len(pred), 3))
    t_pred[:, 0] = unseen_x

    print("Geodesic regression coefficients:", geodesic_regression)

    fig = plt.figure()
    _ = LightSource(270, 45)
    ax_data = fig.add_subplot(121, projection="3d")
    ax_prediction = fig.add_subplot(122, projection="3d")
    for t, y in zip(t_plot, Y):
        plot_ellipsoid(ax_data, t, y, scale=0.025)

    ax_data.set_zlim(-0.1, 0.1)
    ax_data.set_xlim(-1.2, 1.2)
    ax_data.set_ylim(-0.1, 0.1)

    for t, y in zip(t_pred, pred):
        plot_ellipsoid(ax_prediction, t, y, scale=0.025)

    ax_prediction.set_zlim(-0.1, 0.1)
    ax_prediction.set_xlim(-1.2, 1.2)
    ax_prediction.set_ylim(-0.1, 0.1)
    plt.show()


if __name__ == "__main__":
    # mallasto_example()
    # loading_data_example()
    # geodesic_regression_on_subset_using_geomstats()
    geodesic_regression_on_subset_using_manifold()
