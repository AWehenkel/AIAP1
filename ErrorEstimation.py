import numpy as np
from sklearn import linear_model
"""
computeErrors:
Def: This function computes empirically the 3 types of error of a supervised
        learning algorithm on a given distribution function.

IN:
    nb_estimation: int, the number of model build to estimate error.
    sample_size: int, The size of each sample used to build  each model.
    sample_range: np.matrix of shape (n_feature, 2), The range of values of each
                    feature
    estimator: An estimator with function fit and predict(with same behavior as sklearn estimator)
    x: A np.array of shape (n_point, n_feature), the feature point where the errors will be estimated
    y_function: A function which takes a np.array of shape (n_point, n_feature) and possibly
                supplementary arguments, this function defines the distribution and which returns a np.array
                of shape (n_point, n_y).
    y_arguments: *, supplementary argument of the y_function
"""
def computeErrors(nb_estimation, sample_size, sample_range, estimator, x, nb_y, y_function, *y_arguments):
    print("ok")
    nb_point = x.shape[0]
    print(nb_point)
    true_y = np.zeros((nb_point, nb_y, nb_estimation))
    y_estimated = np.zeros((nb_point,nb_y, nb_estimation))

    #Generates a sample, fits the estimator and registers the prediction in y_estimated
    for i in range(0, nb_estimation):
        samples_x = np.multiply(np.random.rand(sample_size, sample_range.shape[1]),
                              np.repeat(sample_range[1, :] - sample_range[0, :], sample_size, axis=0)) + \
                  np.repeat(sample_range[0, :], sample_size, axis = 0)
        true_y[:, :, i] = y_function(x, y_arguments)
        samples_values = y_function(samples_x, y_arguments)
        estimator.fit(samples_x, samples_values)
        y_estimated[:, :, i] = estimator.predict(x)

    bias = np.mean(true_y - y_estimated, 2)
    variance_y = np.std(true_y, 2, ddof=1)
    variance_est = np.var(y_estimated, 2, ddof=1)

    print(np.mean(np.square(true_y - y_estimated), 2))
    print(np.square(bias) + np.square(variance_y) + np.square(variance_est))
    return (bias, variance_y, variance_est)


def yFunction(x, param):
    sigma_e = param[0]
    mu_e = param[1]
    x = np.asarray(x)
    e = np.random.normal(mu_e, sigma_e, x.shape)
    return (np.multiply(x, np.square(np.cos(x) + np.sin(x))) + e)

# Create linear regression object
regr = linear_model.LinearRegression()
a = np.array([[1], [2]])
b = np.array([[1], [2]])
print(computeErrors(100000, 1000, np.matrix('-10; 10'), regr, np.array([[1],[2]]), 1, yFunction, 0.5, 0))