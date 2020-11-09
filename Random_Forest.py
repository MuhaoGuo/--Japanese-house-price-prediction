# from sklearn.model_selection import train_test_split
#
# def Randomforest_Algorithm(Train_X,Train_Y,Test_X,Test_Y):
#
#     train_score_list = []
#     test_score_list_val = []
#     # 选择一个好的森林： bag作为训练集， unuse作为验证集
#     x_train_bag, unuse1, y_train_bag, unuse2 = train_test_split(Train_X, Train_Y, test_size=0.67)
#     # y_train_bag = np.ravel(y_train_bag)
#
#     # validation, select the best model and get the parameters
#     for maxfeature in range(5, 10):
#         for treenumber in range(100, 1500, 50):
#             clf = RandomForestClassifier(n_estimators=treenumber,
#                                          bootstrap=True,     # 部分数据样本构成森林
#                                          max_features=maxfeature,
#                                          )
#             clf.fit(x_train_bag, y_train_bag)
#
#             score_train_each_B = clf.score(Train_X, Train_Y)
#             score_test_each_B_val = clf.score(unuse1, unuse2)
#
#             ##calculate the score
#             train_score_list.append(score_train_each_B)
#             test_score_list_val.append(score_test_each_B_val)
#
#
#     #get the selected model's parameters
#     selected_model_index = test_score_list_val.index(max(test_score_list_val))
#     print("selected_model_index is " + str(selected_model_index))
#     best_maxfeature = (selected_model_index//28)+5
#     best_treenumber = ((selected_model_index % 28) - 1) * 50 + 100
#     print("the best maxfeature for the best model is "+str(best_maxfeature))
#     print("the best treenumber for the best model is "+str(best_treenumber))
#
#
#     # our predict model： we choose this random forest model
#     clf = RandomForestClassifier(
#         n_estimators=best_treenumber,
#         bootstrap=True,
#         max_features=best_maxfeature
#     )
#
#     # training model
#     clf.fit(Train_X, Train_Y)
#
#     # test score
#     score_test = clf.score(Test_X, Test_Y)
#
#     # predict class
#     predict_class_RF = clf.predict(Test_X)
#
#     d = {
#         "predict_score": score_test,
#         "predict_class": predict_class_RF,
#     }
#
#     return d




from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import GridSearchCV

'''仅仅在Train 的阶段需要， 训练完会将模型存入 model_name'''
def Randomforest_Algorithm(Train_X, Train_Y, Test_X, Test_Y, model_name):

    # 设置参数
    parameters = {'n_estimators': [i for i in range(100, 1500, 50)],
                  'max_features': [i for i in range(5, 10)],
                  # 'max_depth': [None, 10, 20]
                  }
    # ------- 训练 --------
    clf_RF = RandomForestClassifier()
    clf = GridSearchCV(clf_RF,
                       param_grid=parameters,
                       cv=5,      # default is 5
                       )
    clf.fit(Train_X, Train_Y)


    best_estimator = clf.best_estimator_  #最优模型
    best_parameter = clf.best_params_    # 最优参数
    print("the parameters for the best model are " + str(best_parameter))

    best_score = best_estimator.score(Train_X, Train_Y)  #最优模型下的训练集得分
    print("the training score for the best model is " + str(best_score) + '\n')

    # 最优模型存入 sav 文件
    filename = model_name
    pickle.dump(best_estimator, open(filename, 'wb'))

    # --------- 测试 ---------
    # predict class
    predict_class_RF = best_estimator.predict(Test_X)
    # test score
    score_test = best_estimator.score(Test_X, Test_Y)


    d = {
        "predict_score": score_test,
        "predict_class": predict_class_RF,
    }

    return d
