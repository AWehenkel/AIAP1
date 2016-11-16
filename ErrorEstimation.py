import numpy as np
from sklearn import linear_model
"""


"""
def computeErrors(nb_estimation, sample_size, sample_range, estimator, x, y_function, *y_arguments):
    nb_point = x.size
    true_y = np.zeros((nb_point, nb_estimation))
    y_estimated = np.zeros((nb_point, nb_estimation))

    #Generates a sample, fits the estimator and registers the prediction in y_estimated
    for i in range(1, nb_estimation):
        samples_x = np.multiply(np.random.rand(sample_size, sample_range.shape[1]),
                              np.repeat(sample_range[1, :] - sample_range[0, :], sample_size, axis=0)) + \
                  np.repeat(sample_range[0, :], sample_size, axis = 0)
        true_y[:, i] = y_function(x, y_arguments)
        samples_values = y_function(samples_x, y_arguments)
        estimator.fit(samples_x, samples_values)
        y_estimated[:, i] = estimator.predict(x)

    bias = np.mean(true_y - y_estimated, 1)
    variance_y = np.std(true_y, 1)
    variance_est = np.std(y_estimated, 1)

    print(np.sum(np.square(true_y - y_estimated))/nb_estimation)
    print(np.square(bias) + np.square(variance_y) + np.square(variance_est))
    return (bias, variance_y, variance_est)


def yFunction(x, param):
    sigma_e = param[0]
    mu_e = param[1]
    x = np.asarray(x)
    e = np.random.normal(mu_e, sigma_e, x.shape)
    return (np.multiply(x, np.square(np.cos(x) + np.sin(x))) + e)

#print(yFunction(np.matrix('1'), 1, 1))
# Create linear regression object
regr = linear_model.LinearRegression()
print(computeErrors(100000, 10, np.matrix('-10; 10'), regr, np.array([[1]]), yFunction, 0.5, 0))