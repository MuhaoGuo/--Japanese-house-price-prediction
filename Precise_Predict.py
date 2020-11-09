from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import time
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA




def Linear_Regression(X_train_0A,
                    Y_train_0A,
                    X_train_AB,
                    Y_train_AB,
                    X_train_BC,
                    Y_train_BC,
                    X_train_CD,
                    Y_train_CD,
                    X_test,
                    Y_test_price,
                    predict_class,
                    draw_num,
                    ):

    lr_0A = LinearRegression().fit(X_train_0A, Y_train_0A)
    lr_AB = LinearRegression().fit(X_train_AB, Y_train_AB)
    lr_BC = LinearRegression().fit(X_train_BC, Y_train_BC)
    lr_CD = LinearRegression().fit(X_train_CD, Y_train_CD)

    result = []

    for i in range(len(X_test)):

        pre_c = predict_class[i]
        x_t = X_test[i].reshape(1, -1)

        if pre_c == "0 ~ A":
            result.append(lr_0A.predict(x_t)[0])
        elif pre_c == "A ~ B":
            result.append(lr_AB.predict(x_t)[0])
        elif pre_c == "B ~ C":
            result.append(lr_BC.predict(x_t)[0])
        elif pre_c == "C ~ +":
            result.append(lr_CD.predict(x_t)[0])

    # print("linear regression predict result is" + str(result))

    # print(Y_test_price)
    # print(result)

    score = r2_score(Y_test_price, result)
    print("linear regression score is " + str(score))


    # 画图
    # plt.figure()
    # plt.figure()
    plt.subplot(221)
    X = [i for i in range(draw_num)]
    # plt.scatter(X, result[:draw_num], color='blue', s=2,  label="predict price")
    # plt.scatter(X, Y_test_price[:draw_num], color='red', s=2, label="true price")
    plt.plot(X, result[:draw_num], color='blue', label="predict price")
    plt.plot(X, Y_test_price[:draw_num], color='red',label="true price")
    plt.title('Linear Regression')
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('Price')
    # plt.show()



def Polynomial_Regression(
        X_train_0A,
        Y_train_0A,
        X_train_AB,
        Y_train_AB,
        X_train_BC,
        Y_train_BC,
        X_train_CD,
        Y_train_CD,
        X_test,
        Y_test_price,
        predict_class,# RF 预测的类别
        degree,       # 阶数
        draw_num,     # 画点数
):

    Input = [('polynomial', PolynomialFeatures(degree = degree)),
             ('modal', LinearRegression())
             ]
    lr_0A = Pipeline(Input).fit(X_train_0A, Y_train_0A)
    lr_AB = Pipeline(Input).fit(X_train_AB, Y_train_AB)
    lr_BC = Pipeline(Input).fit(X_train_BC, Y_train_BC)
    lr_CD = Pipeline(Input).fit(X_train_CD, Y_train_CD)

    result = []
    for i in range(len(X_test)):
        pre_c = predict_class[i]
        x_t = X_test[i].reshape(1, -1)   # 每次预测一个样本  把 list 转化为一维 array

        if pre_c == "0 ~ A":
            result.append(lr_0A.predict(x_t)[0])
        elif pre_c == "A ~ B":
            result.append(lr_AB.predict(x_t)[0])
        elif pre_c == "B ~ C":
            result.append(lr_BC.predict(x_t)[0])
        elif pre_c == "C ~ +":
            result.append(lr_CD.predict(x_t)[0])


    score = r2_score(Y_test_price, result)
    print("Polynomial Regression score is " + str(score))

    # 画图 只取 前 draw_num 个数据
    if(degree == 2):
        plt.subplot(222)
    elif(degree == 3):
        plt.subplot(223)
    else:
        plt.subplot(224)
    X = [i for i in range(draw_num)]
    # plt.scatter(X, result[:draw_num], color='blue', s=2,  label="predict price")
    # plt.scatter(X, Y_test_price[:draw_num], color='red', s=2, label="true price")
    plt.plot(X, result[:draw_num], color='blue', label="predict price")
    plt.plot(X, Y_test_price[:draw_num], color='red', label="true price")
    plt.title('Polynomial Regression degree='+str(degree))
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('Price')



''' ===================Precise predict================='''

def Precise_predict_Algorithm(predict_class_RF, PCA_num , draw_num):
    print("------------Precise Predict-------------------")
    # get data
    Y_train_0A = pd.read_csv('./data/Y_train_0A.csv', low_memory=False, index_col=0).values.ravel()
    X_train_0A = pd.read_csv('./data/X_train_0A.csv', low_memory=False, index_col=0).values
    Y_train_AB = pd.read_csv('./data/Y_train_AB.csv', low_memory=False, index_col=0).values.ravel()
    X_train_AB = pd.read_csv('./data/X_train_AB.csv', low_memory=False, index_col=0).values
    Y_train_BC = pd.read_csv('./data/Y_train_BC.csv', low_memory=False, index_col=0).values.ravel()
    X_train_BC = pd.read_csv('./data/X_train_BC.csv', low_memory=False, index_col=0).values
    Y_train_CD = pd.read_csv('./data/Y_train_CD.csv', low_memory=False, index_col=0).values.ravel()
    X_train_CD = pd.read_csv('./data/X_train_CD.csv', low_memory=False, index_col=0).values

    X_test = pd.read_csv('./data/X_test.csv', low_memory=False, index_col=0).values
    Y_test_price = pd.read_csv('./data/Y_test_price.csv', low_memory=False, index_col=0).values.ravel()



    if(PCA_num == 'None'):    # 不用standardize and PCA
        X_test_PCA = X_test
    else:
        ''' 
        对4 个小训练集 进行  standardize 和 PCA 
        '''
        scaler1 = StandardScaler()
        pca1 = PCA(n_components=10)
        X_train_0A = pca1.fit_transform(scaler1.fit_transform(X_train_0A))

        scaler2 = StandardScaler()
        pca2 = PCA(n_components=10)
        X_train_AB = pca2.fit_transform(scaler2.fit_transform(X_train_AB))

        scaler3 = StandardScaler()
        pca3 = PCA(n_components=10)
        X_train_BC = pca3.fit_transform(scaler3.fit_transform(X_train_BC))

        scaler4 = StandardScaler()
        pca4 = PCA(n_components=10)
        X_train_CD = pca4.fit_transform(scaler4.fit_transform(X_train_CD))


        '''
        对测试集 进行 标准化 和 PCA ， 其函数直接加载训练集的对应 函数
        X_test -> X_test_PCA
        '''
        X_test_PCA =[]
        # print(X_test)
        for i in range(len(X_test)):
            pre_c = predict_class_RF[i]
            x_t = X_test[i].reshape(1, -1)   # 每次 转化 一个样本  把 list 转化为一维 array
            if pre_c == "0 ~ A":
                X_test_PCA_i = pca1.transform(scaler1.transform(x_t))
                X_test_PCA.append(X_test_PCA_i[0])
            elif pre_c == "A ~ B":
                X_test_PCA_i = pca2.transform(scaler2.transform(x_t))
                X_test_PCA.append(X_test_PCA_i[0])
            elif pre_c == "B ~ C":
                X_test_PCA_i = pca3.transform(scaler3.transform(x_t))
                X_test_PCA.append(X_test_PCA_i[0])
            elif pre_c == "C ~ +":
                X_test_PCA_i = pca4.transform(scaler4.transform(x_t))
                X_test_PCA.append(X_test_PCA_i[0])


    # 输出
    plt.figure('PCA: '+ PCA_num)
    # plt.title()

    '''run  Linear_Regression algorithm'''
    print("### Linear_Regression algorithm is running ... ###")
    time_start = time.time()
    Linear_Regression( X_train_0A,  Y_train_0A, X_train_AB, Y_train_AB, X_train_BC, Y_train_BC,
                       X_train_CD, Y_train_CD, X_test_PCA, Y_test_price, predict_class_RF, draw_num)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
    print("Linear_Regression algorithm has finished.\n")


    '''run Polynomial_Regression_Degree2 algorithms'''
    print("### Polynomial_Regression_Degree2 algorithm is running ... ###")
    time_start = time.time()
    Polynomial_Regression( X_train_0A, Y_train_0A, X_train_AB,  Y_train_AB, X_train_BC, Y_train_BC,
                           X_train_CD,  Y_train_CD, X_test_PCA, Y_test_price, predict_class_RF, 2, draw_num)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
    print("Polynomial_Regression_Degree2 algorithm has finished.\n")

    #
    '''run  Polynomial_Regression_Degree3 algorithms'''
    print("### Polynomial_Regression_Degree3 algorithm is running ... ###")
    time_start = time.time()
    Polynomial_Regression( X_train_0A, Y_train_0A, X_train_AB,  Y_train_AB, X_train_BC, Y_train_BC,
                           X_train_CD,  Y_train_CD, X_test_PCA, Y_test_price, predict_class_RF, 3, draw_num)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
    print("Polynomial_Regression_Degree3 algorithm has finished.\n")


    ''' run Polynomial_Regression_Degree4 algorithms'''
    print("### Polynomial_Regression_Degree4 algorithm is running ... ###")
    time_start = time.time()
    Polynomial_Regression( X_train_0A, Y_train_0A, X_train_AB,  Y_train_AB, X_train_BC, Y_train_BC,
                           X_train_CD,  Y_train_CD, X_test_PCA, Y_test_price, predict_class_RF, 4, draw_num)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
    print("Polynomial_Regression_Degree4 algorithm has finished.\n")

    plt.show()

