from class_obs_tica import ObservableTicaObject
import numpy as np
from sklearn import decomposition


def epsilon_close(a, b, epsilon = .00001):
    return True if abs(a-b) < epsilon else False


def epsilon_close_mat(a,b, epsilon = .00001):  # easy function that just checks an epsilon closeness of two corresponding elements
    if np.array(a).shape == np.array(b).shape:
        n = abs(a-b)
        for element in n.flatten():
            if element > epsilon:
                return False
        return True
    return 'the shapes are not the same'


def test_fit_kxx_shapes():  # the fit() func tests will be done after I garuntee that all the parts of fit() work first
    obj = ObservableTicaLObject()
    X = np.random.randint(2, 100, (4, 100, 40))
    Y = np.random.randint(2, 100, (4, 100, 30))
    obj.fit(X,Y)
    shape = np.array(K_xx).shape
    shape.all() == shape[0]


def test_fit_svd_shapes():
    return


def test_whiten_correctness_mean():
    X = np.random.randint(2, 100, (4, 100, 40))  # these two lines creates the random vectors to test on
    Y = np.random.randint(2, 100, (4, 100, 30))

    tica_obj = ObservableTicaObject()  # This chunk sets up the Obs_tICA obj so that we van run the whiten func on it
    tica_obj.x = X
    tica_obj.y = Y
    tica_obj.x_obj = decomposition.PCA(whiten=True)
    tica_obj.y_obj = decomposition.PCA(whiten=True)
    tica_obj.x_obj.fit(np.vstack(X))
    tica_obj.y_obj.fit(np.vstack(Y))

    tica_obj.whiten()  # Running the function

    x = np.array(np.vstack(tica_obj.x_whitened))  # Flattened versions of the whitened data
    y = np.array(np.vstack(tica_obj.y_whitened))  # for ease of cov mat calc

    x_avg = x.mean(0)  # finding the averages for x and y, post whitening
    y_avg = y.mean(0)

    return 'Dimensional means of whitened x data are all 0?: '+ str(epsilon_close_mat(x_avg, np.zeros(x_avg.shape))) + '\nDimensional means of whitened y data are all 0?: '+ str(epsilon_close_mat(y_avg, np.zeros(y_avg.shape)))


def test_whiten_correctness_cov():
    X = np.random.randint(2,100, (4,100,40))  # these two lines creates the random vectors to test on
    Y = np.random.randint(2,100, (4,100,30))

    tica_obj = ObservableTicaObject()  # This chunk sets up the Obs_tICA obj so that we van run the whiten func on it
    tica_obj.x = X
    tica_obj.y = Y
    tica_obj.x_obj = decomposition.PCA(whiten = True)
    tica_obj.y_obj = decomposition.PCA(whiten = True)
    tica_obj.x_obj.fit(np.vstack(X))
    tica_obj.y_obj.fit(np.vstack(Y))

    tica_obj.whiten()  # Running the function

    x = np.array(np.vstack(tica_obj.x_whitened))  # Flattened versions of the whitened data
    y = np.array(np.vstack(tica_obj.y_whitened))  # for ease of cov mat calc

    n_x = 1  # counting the number of data points to normalize the cov mats
    n_y = 1
    for i in range(len(X.shape)-1):
        n_x *= X.shape[i]
        n_y *= X.shape[i]

    cov_x = x.T.dot(x)/(n_x-1)  # '-1' for bessel correction
    cov_y = y.T.dot(y)/(n_y-1)

    return 'Cov matrices of X and Y data are identity? ' + str(epsilon_close_mat(cov_x, np.identity(X.shape[-1]))) + ' ' + str(epsilon_close_mat(cov_y, np.identity(Y.shape[-1])))
    # cov mat should be Identity now


def test_Kxx_vals():
    X = np.random.randint(2, 100, (4, 100, 40))  # these two lines creates the random vectors to test on
    Y = np.random.randint(2, 100, (4, 100, 30))

    tica_obj = ObservableTicaObject()  # This chunk sets up the Obs_tICA obj so that we van run the whiten func on it

    tica_obj.x = X
    tica_obj.y = Y
    tica_obj.x_obj = decomposition.PCA(whiten=True)
    tica_obj.y_obj = decomposition.PCA(whiten=True)
    tica_obj.x_obj.fit(np.vstack(X))
    tica_obj.y_obj.fit(np.vstack(Y))

    tica_obj.whiten()  # Running the function
    K_xx, K_evals = tica_obj.estimate_koop_xx(tica_obj.x_0, tica_obj.x_tau)

    print ('K_xx square and has the right dimensions? ', K_xx.shape == (X.shape[-1], X.shape[-1]))

    M = np.amax(K_evals)
    m = np.amin(K_evals)

    # not sure what i should be returning here, the range that the eigenvalues are in? a bool of them being in (-1, 1]?

    def
print (test_Kxx_vals())
