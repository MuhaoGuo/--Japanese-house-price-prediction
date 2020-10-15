from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd



def SVC_Algorithm(Train_X, Train_Y, Test_X, Test_Y):

    # general model
    # clf_svc = SVC(decision_function_shape='ovr', kernel='rbf', probability=True)

    # validation to select the best parameters
    param_grid = {'gamma': [0.1, 0.01, 0.001, 0.0001], 'C': [0.1, 1, 10, 100]}
    clf = GridSearchCV(SVC(decision_function_shape='ovr', kernel='rbf', probability=True,), cv=10, param_grid=param_grid)

    clf.fit(Train_X, Train_Y)

    best_params = clf.best_params_
    best_score = clf.best_score_
    best_estimator = clf.best_estimator_    # our selected model

    print("the parameters for the best model are " + str(best_params))
    print("the cross validation score for the best model is " + str(best_score))

    score_test = best_estimator.score(Test_X, Test_Y)   # score of the model

    predict_class_SVC = clf.predict(Test_X)    # predict value

    d = {
        "predict_score": score_test,
        "predict_class": predict_class_SVC,
    }

    return d