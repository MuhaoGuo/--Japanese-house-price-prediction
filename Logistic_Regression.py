from sklearn.linear_model import LogisticRegressionCV

def Logistic_Regression_Algorithm_l2penalty(Train_X, Train_Y, Test_X, Test_Y):
    # validation
    val_list = [
        10**-5,
        10**-4,
        10**-3,
        10**-2,
        10**-1,
        1,
        10,
        100,
        10**3,
        10**4,
        10**5
    ]

    clf = LogisticRegressionCV(
        Cs=val_list,
        cv=10,
        penalty="l2",
        multi_class="multinomial",
        max_iter=10000,
        solver="lbfgs"
    )

    clf.fit(Train_X, Train_Y)
    score_test = clf.score(Test_X, Test_Y)       # SCORE

    predict_class_LR_l2 = clf.predict(Test_X)    # predict class

    d = {
        "predict_score": score_test,
        "predict_class": predict_class_LR_l2,
    }

    return d


def Logistic_Regression_Algorithm_l1penalty(Train_X,Train_Y,Test_X,Test_Y):
    # validation
    val_list = [
        10**-5,
        10**-4,
        10**-3,
        10**-2,
        10**-1,
        1,
        10,
        100,
        10**3,
        10**4,
        10**5
    ]

    # solver="liblinear" is used for l1 penalty
    clf = LogisticRegressionCV(
        Cs=val_list,
        cv=10,
        penalty="l1",
        multi_class="ovr",
        max_iter=10000,
        solver="liblinear",
    )

    clf.fit(Train_X, Train_Y)
    score_test = clf.score(Test_X, Test_Y)

    predict_class_LR_l1 = clf.predict(Test_X)

    d = {
        "predict_score": score_test,
        "predict_class": predict_class_LR_l1,
    }

    return d
