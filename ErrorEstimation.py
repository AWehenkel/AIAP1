import numpy as np
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot as plt
from matplotlib import patches
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

    nb_point = x.shape[0]
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

    sq_bias = np.square(np.mean(true_y - y_estimated, 2))
    noise = np.var(true_y, 2, ddof=1)
    variance = np.var(y_estimated, 2, ddof=1)

    #Compute the total square error to compare it to the sum of the 3 estimators
    tot_error = np.mean(np.square(true_y - y_estimated), 2)

    return (noise, sq_bias, variance, tot_error)


def yFunction(x, param):
    sigma_e = param[0]
    mu_e = param[1]
    x = np.asarray(x)
    e = np.random.normal(mu_e, sigma_e, x.shape)
    return (np.multiply(x, np.square(np.cos(x) + np.sin(x))) + e)

def plotErrors(title_model, errors, x_range, x_label):
    '''
    This function plots the four errors relatively to a certain feature for
    a certain model

    Parameters
    ----------
    title_model, string
        the name of the model
    errors, np.array of shape (4, range)
        the four errors for different values of the feature
    x_range
        the different values of the the feature
    x_label
        the name of the feature
    '''

    plt.figure(1)
    plt.plot(x_range, errors[0], 'r')
    plt.plot(x_range, errors[1], 'b')
    plt.plot(x_range, errors[2], 'g')
    plt.plot(x_range, errors[3], 'k')
    r_patch = patches.Patch(color='r', label='Noise')
    b_patch = patches.Patch(color='b', label='Squared Biais')
    g_patch = patches.Patch(color='g', label='Variance')
    k_patch = patches.Patch(color='k', label='Squared error')
    plt.legend(handles=[r_patch, b_patch, g_patch, k_patch])
    plt.title(title_model)
    plt.xlabel(x_label)
    plt.show()

    plt.figure(1)
    plt.subplot(221)
    plt.plot(x_range, errors[0], 'r')
    plt.title(title_model + " - Noise")
    plt.subplot(222)
    plt.plot(x_range, errors[1], 'b')
    plt.title(title_model + " - Squared Biais")
    plt.subplot(223)
    plt.plot(x_range, errors[2], 'g')
    plt.title(title_model + " - Variance")
    plt.xlabel(x_label)
    plt.subplot(224)
    plt.plot(x_range, errors[3], 'k')
    plt.title(title_model + " - Squared error")
    plt.xlabel(x_label)
    plt.show()


def LSSizeInfluence(title_model, model):
    '''
    This function plots the evolution of mean values the squared error, residual
    error, bias and variance for a given model as a function of the size of the
    learning sample

    Parameters
    ----------
    title_model, string
        name of the model

    model
        An estimator with function fit and predict(with same behavior as
        sklearn estimator)
    '''
    nb_estimations = 10000
    nb_points = 1000
    start = 10
    stop = 100
    step = 1
    sizes_range = range(start, stop, step)
    means = np.zeros((4, (stop-start)/step))
    x_array = np.zeros((nb_points,1))

    i = 0
    for value in np.linspace(-10,10,nb_points):
        x_array[i] = value
        i += 1

    i = 0
    for nb_samples in sizes_range:
        errors = computeErrors(nb_estimations, nb_samples, np.matrix('-10; 10'), model, x_array, 1, yFunction, 0.5, 0)
        means[:,i] = np.ravel(np.mean(errors[:],1))
        i += 1

    plotErrors(title_model, means, np.ravel(sizes_range), 'LS size')

def kNNComplexityInfluence():
    '''
    This function plots the evolution of mean values the squared error, residual
    error, bias and variance for kNN algorithm as a function of its complexity
    '''
    nb_estimations = 1000
    nb_points = 1000
    nb_samples = 10
    start = 1
    stop = 10
    step = 1
    k_range = range(start, stop, step)
    means = np.zeros((4,(stop-start)/step))
    x_array = np.zeros((nb_points,1))

    i = 0
    for value in np.linspace(-10,10,nb_points):
        x_array[i] = value
        i += 1

    i = 0
    for k in k_range:
        model = KNeighborsRegressor(n_neighbors=k)
        errors = computeErrors(nb_estimations, nb_samples, np.matrix('-10; 10'), model, x_array, 1, yFunction, 0.5, 0)
        means[:, i] = np.ravel(np.mean(errors[:],1))
        i += 1

    plotErrors("kNN", means, k_range, 'k')

def linearRegrComplexityInfluence():
    '''
    This function plots the evolution of mean values the squared error, residual
    error, bias and variance for linear regression as a function of its complexity
    '''
    nb_estimations = 10000
    nb_points = 1000
    nb_samples = 1000
    start = 1
    stop = 100
    nb_steps = 100
    alpha_range = np.linspace(start, stop, nb_steps)
    means = np.zeros((4,nb_steps))
    x_array = np.zeros((nb_points,1))

    i = 0
    for value in np.linspace(-10,10,nb_points):
        x_array[i] = value
        i += 1

    i = 0
    for alpha in alpha_range:
        model = linear_model.Ridge(alpha)
        errors = computeErrors(nb_estimations, nb_samples, np.matrix('-10; 10'), model, x_array, 1, yFunction, 0.5, 0)
        means[:, i] = np.ravel(np.mean(errors[:],1))
        i += 1

    plotErrors("Linear model", means, alpha_range, 'alpha')


def stdNoiseInfluence(title_model, model):
    '''
    This function plots the evolution of mean values the squared error, residual
    error, bias and variance for a given model as a function of the std of the
    noise epsilon

    Parameters
    ----------
    title_model, string
        name of the model

    model
        An estimator with function fit and predict(with same behavior as
        sklearn estimator)
    '''
    nb_estimations = 1000
    nb_points = 1000
    nb_samples = 1000
    start = 0.01
    stop = 10
    nb_steps = 1000
    epsilon_range = np.linspace(start, stop, nb_steps)
    means = np.zeros((4,nb_steps))
    x_array = np.zeros((nb_points,1))

    i = 0
    for value in np.linspace(-10,10,nb_points):
        x_array[i] = value
        i += 1

    i = 0
    for epsilon in epsilon_range:
        errors = computeErrors(nb_estimations, nb_samples, np.matrix('-10; 10'), model, x_array, 1, yFunction, epsilon, 0)
        means[:,i] = np.ravel(np.mean(errors[:],1))
        i += 1

    plotErrors(title_model, means, epsilon_range, 'epsilon')


if __name__ == "__main__":

    # Question 2.b
    nb_estimations = 1000
    nb_points = 1000
    nb_samples = 1000
    x_array = np.zeros((nb_points,1))

    i = 0
    for value in np.linspace(-10,10,nb_points):
        x_array[i] = value
        i += 1

    # Linear regression method
    '''
    regr = linear_model.LinearRegression()
    lin_errors = computeErrors(nb_estimations, nb_samples, np.matrix('-10; 10'), regr, x_array, 1, yFunction, 0.5, 0)
    plotErrors("Linear Model", lin_errors, np.ravel(x_array), 'x')

    # Non-linear regression method: kNN
    n_neighbours = 10
    neigh = KNeighborsRegressor(n_neighbors=n_neighbours)
    kNN_errors = computeErrors(nb_estimations, nb_samples, np.matrix('-10; 10'), neigh, x_array, 1, yFunction, 0.5, 0)
    plotErrors("kNN - " + str(n_neighbours), kNN_errors, np.ravel(x_array), 'x')
    '''

    # Question 2.d
    #Linear regression method
    regr = linear_model.LinearRegression()
    LSSizeInfluence("Linear Model", regr)
    #stdNoiseInfluence("Linear Model", regr)
    #linearRegrComplexityInfluence()

    '''
    # Non-linear regression method: kNN
    n_neighbours = 10
    neigh = KNeighborsRegressor(n_neighbors=n_neighbours)
    #LSSizeInfluence("kNN - " + str(n_neighbours), neigh)
    #stdNoiseInfluence("kNN - " + str(n_neighbours), neigh)
    kNNComplexityInfluence()
    '''