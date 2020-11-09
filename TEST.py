import time
import os
import pandas as pd
import numpy as np
import pickle


# # 精准预测函数
# def predict_precise(dc, predict_class):
#     Precise_Predict.Linear_Regression_Algorithm(
#                                             dc.X_train_0A, dc.Y_train_0A,
#                                             dc.X_train_AB, dc.Y_train_AB,
#                                             dc.X_train_BC, dc.Y_train_BC,
#                                             dc.X_train_CD, dc.Y_train_CD,
#                                             dc.X_test,     dc.Y_test_price,
#                                             predict_class
#                                             )


# 进入 机器学习算法 的格式 为 array ， 一般数据传递为df

''' ========   WE HAVE SAVE THE MODEL INTO A .asv FILE! =======  '''
def TEST(model_name):
    X_test_PCA = pd.read_csv('./data/X_test_PCA.csv', low_memory=False, index_col=0).values  #转化为array，要index_col=0
    Y_test = pd.read_csv('./data/Y_test.csv', low_memory=False, index_col=0).values.ravel()
    X_test = pd.read_csv('./data/X_test.csv', low_memory=False, index_col=0).values

    # if we has the model, we do not need to train:
    if os.path.exists(model_name):
        print("There is a .sav file, that's the trained model, we don't need train again.")
        print("------------Class Predict-------------------")
        time_start = time.time()

        # 1. load the model
        loaded_model = pickle.load(open(model_name, 'rb'))  # load the trained model
        print("Random Forest algorithm is running ...")

        # 2. start predict
        # print(X_test_PCA)
        predict_class_RF = loaded_model.predict(
            # X_test_PCA
            X_test
        )

        score_test = loaded_model.score(
            # X_test_PCA,
            X_test,
            Y_test)
        print("Random Forest algorithm TEST score is " + str(score_test))
        print("Random Forest algorithm has finished. \n")

        # 3.record the predict class of test data
        print("result is loading to a file ...")
        predict_class_RF_array = np.array(predict_class_RF)                 # list 转为 array
        predict_class_RF_df = pd.DataFrame(predict_class_RF_array)          # array 转为 df
        predict_class_RF_df.columns = ['Predict class']                     # df 加列标签
        predict_class_RF_df.to_csv('./data/predict_test.csv', index='False')  # df 写入文件
        print("result has writen into a file.\n")
        time_end = time.time()
        print('time cost', time_end - time_start, 's')

        return predict_class_RF

    else:
        print("There is no such a model_name file")

























