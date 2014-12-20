import matplotlib.pyplot as plt #for simplicity
import numpy
import csv
import copy
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression # basic linear regression
from sklearn.linear_model import Lasso # L1 regularizer
from sklearn.linear_model import Ridge # L2 regularizer
from sklearn.linear_model import ElasticNet # Linear regression with combined L1 and L2 priors as regularizer
from sklearn.linear_model import LogisticRegression # nonlinear regression
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR  # SVM regression
from sklearn.feature_selection import SelectKBest, chi2, RFE

# Read data from csv
# Parameter: csv filename
# Returns: feature matrix (X in class notation) and quality vector (y in class notation)
def read_data(filename):
    csvfile = open(filename, 'r')
    spamreader = csv.reader(csvfile, delimiter=';')
    D = []
    for row in spamreader:
        D.append(row)
    csvfile.close()
    return ((numpy.array(D)[1:,:-1]).astype(float), (numpy.array(D)[1:, -1]).astype(float))
    # 1: means 1st row to last row, :-1 means 0th col to second last col, which are features, -1 means last col, which is quality

# K-fold cross validation
# Paramter: X: feature matrix, y: quatity vector, k: number of fold, let default value = 3
# Returns: training set and test set
def get_kfold_train_test(X, y, k = 3):
    # len(X) is number of samples
    kf = KFold(len(X), k)
    for train, test in kf:
        # training sets and test sets
        yield X[train], y[train], X[test], y[test]

# mean absolute deviation
# Parameter: pred: prediction value(y_hat in class notation), actual: actual value(y_test in class notation)
# Returns: mean absolute deviation(a float value)
def mean_absolute_dev(pred, actual):
    return mean_absolute_error(actual, pred)

# mean squared error
# Parameter: pred: prediction value(y_hat in class notation), actual: actual value(y_test in class notation)
# Returns: mean squared error(a float value)
def mean_squared_err(pred, actual):
    return mean_squared_error(actual, pred)

# print confusion matrix
# Parameter: pred: prediction vector(y_hat in class notation), actual: actual vector(y_test in class notation), these vector should be int type
def print_confusion_matrix(actual, pred):
    # get confusion matrix (we dont have sample whose quality is <=3 or >=9, but we still keep 3 and 9 but neglect 0, 1, 2 and 10 to save memory)
    lb = range(3, 10)
    cm = confusion_matrix(actual, numpy.round(pred), lb)
    print('    3   4   5   6   7   8   9 and the rows are also 3 4 5 6 7 8 9')
    print(cm)

# plot result
# actual_list: actual value lists, preds_list: predicted value lists, name_list: names of the lists
def plot_result(actual_list, preds_list, name_list):
    # number of plots in a window(typically 1, sometimes 2 or 3 when we want to compare)
    pn = len(actual_list)
    # number of values in a list
    x = range(len(actual_list[0]))
    fig = plt.figure()
    for i in range(pn):        
        plt.subplot(pn, 1, i+1)
        plt.title(name_list[i], fontsize = 10)
        plt.plot(x[:len(actual_list[0])], actual_list[i][:len(actual_list[0])], 'k.')
        plt.plot(x[:len(actual_list[0])], preds_list[i][:len(actual_list[0])], 'r.')
        plt.axis([0, len(actual_list[0]), 0, 10])
    plt.tight_layout()
    plt.show()

# select K best features(PCA)
# Returns: new X and the best features index
def select_KBest(X, y, k = 6):
    selector = SelectKBest(chi2, k) # use chi squared statistic for each class/feature combination to select 6 best features
    X_new = selector.fit_transform(X, y)
    index = selector.get_support(True) # we want to show which features are left
    return X_new, index

# K-fold cross validation regression prototype
# Parameters: estimators: a vector of tuple(s) whose struction is/are [('name_of_predictor1', predictor1),('name_of_predictor2', predictor2),...], X: feature matrix, y: quatity vector
# Returns: a dictionary with keys as the names of estimators, values as (prediction_vector_of_all_samples, corresponding_MAD)
def make_KfoldCV_regression(estimators, X, y):
    # return value, a dictionary
    ets = {}
    for name, estimator in estimators:
        # select best model with minimum error (et1, er1 for MAD, et2, er2 for MSE)
        et1 = None
        et2 = None
        er1 = None
        er2 = None
        finaly_test = None #used to calculate E_out
        finalX_test = None
        for X_train, y_train, X_test, y_test in get_kfold_train_test(X, y):
            # do training
            estimator.fit(X_train, y_train)
            # get prediction vector
            preds = estimator.predict(X_test)
            # calculate MAD, MSE (out of sample)
            error1 = mean_absolute_dev(preds, y_test)
            error2 = mean_squared_err(preds, y_test)
            # select the best training set and test set, i.e select the best estimator, and the corresponding MAD and MSE (out of sample)
            if er1 is None:
                et1 = copy.deepcopy(estimator)
                er1 = error1
                finaly_test = copy.deepcopy(y_test)
                finalX_test = copy.deepcopy(X_test)
            else:
                if error1 < er1:
                    et1 = copy.deepcopy(estimator)
                    er1 = error1
                    finaly_test = copy.deepcopy(y_test)
                    finalX_test = copy.deepcopy(X_test)
            if er2 is None:
                et2 = copy.deepcopy(estimator)
                er2 = error2
            else:
                if error2 < er2:
                    et2 = copy.deepcopy(estimator)
                    er2 = error2
        print(name, ':\n', 'MAD (out of sample):', '%.4f' % er1, '; MSE (out of sample):', '%.4f' % er2)
        # use the best estimator(respectively based on MAD and MSE) to predict all 
        y_preds1 = et1.predict(X)
        y_preds2 = et2.predict(X)
        # calculate E_out
        finaly_preds = et1.predict(finalX_test)
        count = 0
        tol = 0.8
        for i in range(len(finaly_test)):
            if abs(finaly_preds[i]-finaly_test[i]) <= tol:
                count += 1
        print('With tolerance ', '%.4f' % tol, ', E_out is ', '%.4f' %(1 - count/len(finaly_test)))
        # put corresponding vectors of prediction of all samples and MAD, MSE into a dictionary with keys as names of the estimators
        ets[name] = (y_preds1, mean_absolute_dev(y_preds1, y), y_preds2, mean_squared_err(y_preds2, y))
    return ets            
# basic linear regression
# Parameters: X: feature matrix, y: quatity vector
# Returns: a dictionary created by make_KfoldCV_regression() method
def basic_linear_regression(X, y):
    # define estimator
    estimator = LinearRegression()
    # create the vector of tuples required as parameter of make_KfoldCV_regression(), here we have 1 estimator
    estimators = [('linear regression', estimator)]
    return make_KfoldCV_regression(estimators, X, y)
    
# Linear regression with L1, L2, ElasticNet regularizer
# Parameters: X: feature matrix, y: quatity vector
# Returns: a dictionary created by make_KfoldCV_regression() method
def regularizer_linear_regression(X, y):
    # define estimators
    estimator1 = Lasso(alpha = 0.2)
    estimator2 = Ridge(alpha = 0.8)
    estimator3 = ElasticNet(alpha = 0.9)
    # create the vector of tuples required as parameter of make_KfoldCV_regression(), here we have 3 estimators
    estimators = [('L1 Regularizer', estimator1), ('L2 Regularizer', estimator2), ('Elastic Net Regularizer', estimator3)]
    return make_KfoldCV_regression(estimators, X, y)

# Linear regression with RFE method, using L1, L2 and Elastic Net Regularizer
# Parameters: X: feature matrix, y: quatity vector
# Returns: a dictionary created by make_KfoldCV_regression() method
def RFE_linear_regression(X, y, n_features = 6):
    # define estimators
    estimator1 = Lasso(alpha = 0.2)
    estimator2 = Ridge(alpha = 0.8)
    estimator3 = ElasticNet(alpha = 0.9)
    # create the vector of tuples required as parameter of make_KfoldCV_regression(), here we have 3 estimators
    estimators = [('RFE L1 Regularizer', estimator1), ('RFE L2 Regularizer', estimator2), ('RFE Elastic Net Regularizer', estimator3)]
    ets = {} 
    for name, estimator in estimators:  
        # select best model with minimum error (et1, er1 for MAD, et2, er2 for MSE)      
        et1 = None
        et2 = None
        er1 = None
        er2 = None
        finaly_test = None #used to calculate E_out
        finalX_test = None
        for X_train, y_train, X_test, y_test in get_kfold_train_test(X, y):
            # do feature selection
            selector = RFE(estimator, n_features, step=1)
            # do training
            selector.fit(X_train, y_train)
            # get prediction vector
            preds = selector.predict(X_test)
            # calculate MAD, MSE (out of sample)
            error1 = mean_absolute_dev(preds, y_test)
            error2 = mean_squared_err(preds, y_test)
            # select the best training set and test set, i.e select the best estimator, and the corresponding MAD and MSE (out of sample)
            if er1 is None:
                et1 = copy.deepcopy(selector)
                er1 = error1
                finaly_test = copy.deepcopy(y_test)
                finalX_test = copy.deepcopy(X_test)
            else:
                if error1 < er1:
                    et1 = copy.deepcopy(selector)
                    er1 = error1
                    finaly_test = copy.deepcopy(y_test)
                    finalX_test = copy.deepcopy(X_test)
            if er2 is None:
                et2 = copy.deepcopy(selector)
                er2 = error2
            else:
                if error2 < er2:
                    et2 = copy.deepcopy(selector)
                    er2 = error2
        print(name, ':\n', 'MAD (out of sample):', '%.4f' % er1, '; MSE (out of sample):', '%.4f' % er2)
        # use the best estimator(respectively based on MAD and MSE) to predict all 
        y_preds1 = et1.predict(X)
        y_preds2 = et2.predict(X)
        # calculate E_out
        finaly_preds = et1.predict(finalX_test)
        count = 0
        tol = 0.8
        for i in range(len(finaly_test)):
            if abs(finaly_preds[i]-finaly_test[i]) <= tol:
                count += 1
        print('With tolerance ', '%.4f' % tol, ', E_out is ', '%.4f' %(1 - count/len(finaly_test)))
        # put corresponding vectors of prediction of all samples and MAD, MSE into a dictionary with keys as names of the estimators
        # put corresponding vectors of prediction of all samples and MAD, MSE into a dictionary with keys as names of the estimators
        ets[name] = (y_preds1, mean_absolute_dev(y_preds1, y), y_preds2, mean_squared_err(y_preds2, y))
    return ets

# Nonlinear regression with Logistic L1, L2 regulatizer and BayesianRidge regularizer
# Parameters: X: feature matrix, y: quatity vector
# Returns: a dictionary created by make_KfoldCV_regression() method
def regularizer_nonlinear_regression(X, y):
    # define estimators
    estimator1 = LogisticRegression(penalty='l1', C = 0.8)
    estimator2 = LogisticRegression(penalty='l2', C = 0.8)
    estimator3 = BayesianRidge(n_iter=300)
    # create the vector of tuples required as parameter of make_KfoldCV_regression(), here we have 3 estimators
    estimators = [('Logistic L1 Regularizer', estimator1), ('Logistic L2 Regularizer', estimator2), ('Bayesian Ridge Regularizer', estimator3)]
    return make_KfoldCV_regression(estimators, X, y)       

# SVM regression
# Parameters: X: feature matrix, y: quatity vector
# Returns: a dictionary created by make_KfoldCV_regression() method
def SVM_regression(X, y):
    # define estimators
    estimator1 = SVR(kernel='linear')
    estimator2 = SVR(kernel='rbf')
    # create the vector of tuples required as parameter of make_KfoldCV_regression(), here we have 2 estimators
    estimators = [('SVM with linear kernel', estimator1), ('SVM with RBF kernel', estimator2)]
    return make_KfoldCV_regression(estimators, X, y)    

# launch a selected regression and show MAD, confusion matrix and plot
def lauch_regression(regression_handler, X, y, n_features = None):
    # call regression
    if n_features is None:
        dic = regression_handler(X, y)
    else:
        dic = regression_handler(X, y, n_features)
    # handle results
    actual_list = []
    preds_list = []
    name_list = []
    for key in dic:
        # print confusion matrix
        print('confusion matrix for %s:' % key)
        print_confusion_matrix(y, dic[key][0])
        # create parameters for plot_result()
        actual_list.append(y)
        preds_list.append(dic[key][0])
        name_list.append(key + ', MAD (in sample and out of sample error): ' + str(numpy.round(dic[key][1],4)) + '\n MSE (in sample and out of sample error): ' + str(numpy.round(dic[key][3],4)))               
    # plot result
    plot_result(actual_list, preds_list, name_list)

# run the program
# red wine
input('Press any key to load red wine data...')
X, y = read_data('winequality-red.csv')
print('%d rows, %d features loaded!' % (len(X), len(X[0])) )
print('=============================================')
input('Press any key to start linear regression...')
lauch_regression(basic_linear_regression, X, y)    
print('=============================================')
input('Press any key to start regularizer linear regression...')
lauch_regression(regularizer_linear_regression, X, y)
print('=============================================')

input('Press any key to select best K features by feature selection...')
X_new, indices = select_KBest(X, y)    
print(X_new.shape, indices)
print('=============================================')
input('Press any key to start linear regression with above K features...')
lauch_regression(basic_linear_regression, X_new, y)    
print('=============================================')
input('Press any key to start regularizer linear regression with above K features...')
lauch_regression(regularizer_linear_regression, X_new, y)

print('=============================================')
input('Press any key to start feature selection by RFE...')
lauch_regression(RFE_linear_regression, X, y, n_features=6)
print('=============================================')

input('Press any key to start regularizer nonlinear regression...')
lauch_regression(regularizer_nonlinear_regression, X, y)
print('=============================================')
input('Press any key to start SVM regression...')
lauch_regression(SVM_regression, X, y)

# white wine
input('Red winecx Finished!\nPress any key to load white wine data...')
X, y = read_data('winequality-white.csv')
print('%d rows, %d features loaded!' % (len(X), len(X[0])) )
print('=============================================')
input('Press any key to start linear regression...')
lauch_regression(basic_linear_regression, X, y)    
print('=============================================')
input('Press any key to start regularizer linear regression...')
lauch_regression(regularizer_linear_regression, X, y)
print('=============================================')

input('Press any key to select best K features by feature selection...')
X_new, indices = select_KBest(X, y)    
print(X_new.shape, indices)
print('=============================================')
input('Press any key to start linear regression with above K features...')
lauch_regression(basic_linear_regression, X_new, y)    
print('=============================================')
input('Press any key to start regularizer linear regression with above K features...')
lauch_regression(regularizer_linear_regression, X_new, y)

print('=============================================')
input('Press any key to start feature selection by RFE...')
lauch_regression(RFE_linear_regression, X, y, n_features=6)
print('=============================================')

input('Press any key to start regularizer nonlinear regression...')
lauch_regression(regularizer_nonlinear_regression, X, y)
print('=============================================')
input('Press any key to start SVM regression...')
lauch_regression(SVM_regression, X, y)
    

