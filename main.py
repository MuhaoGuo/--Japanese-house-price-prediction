import pandas as pd
from sklearn.model_selection import train_test_split
import Linear_Regression

import numpy as np
import sklearn.preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import Logistic_Regression
import Random_Forest
import SVC
import Linear_Regression


class DATA_PREPROCESS():
    '''特征 '''

    def __init__(self, path):
        self.columns_name = ['Type', 'Region', 'MunicipalityCode', 'Prefecture', 'Municipality', 'DistrictName', 'NearestStation',
                   'TimeToNearestStation', 'MinTimeToNearestStation', 'MaxTimeToNearestStation', 'TradePrice', 'FloorPlan',
                   'Area', 'AreaIsGreaterFlag', 'UnitPrice', 'PricePerTsubo', 'LandShape', 'Frontage', 'FrontageIsGreaterFlag',
                   'TotalFloorArea', 'TotalFloorAreaIsGreaterFlag', 'BuildingYear', 'PrewarBuilding', 'Structure', 'Use',
                   'Purpose', 'Direction', 'Classification', 'Breadth', 'CityPlanning', 'CoverageRatio', 'FloorAreaRatio',
                   'Period', 'Year', 'Quarter', 'Renovation', 'Remarks']

        self.columns_name_use = ['Type', 'Region', 'MunicipalityCode', 'Prefecture', 'DistrictName', 'NearestStation',
                   'TimeToNearestStation', 'MinTimeToNearestStation', 'MaxTimeToNearestStation', 'TradePrice', 'FloorPlan',
                   'Area', 'AreaIsGreaterFlag', 'LandShape', 'Frontage', 'FrontageIsGreaterFlag','TotalFloorArea',
                   'TotalFloorAreaIsGreaterFlag', 'BuildingYear', 'PrewarBuilding', 'Structure', 'Use',
                   'Purpose', 'Direction', 'Classification', 'Breadth', 'CityPlanning', 'CoverageRatio', 'FloorAreaRatio',
                   'Year', 'Quarter']


        self.str_features = ['Type', 'Region', 'MunicipalityCode', 'Prefecture', 'DistrictName', 'NearestStation',
                       'FloorPlan', 'AreaIsGreaterFlag', 'LandShape', 'FrontageIsGreaterFlag','TotalFloorAreaIsGreaterFlag','BuildingYear',
                        'PrewarBuilding', 'Structure', 'Use', 'Purpose', 'Direction', 'Classification',
                       'CityPlanning', 'Year', 'Quarter', ]

        self.float_features =[ 'MinTimeToNearestStation', 'MaxTimeToNearestStation', 'TradePrice', 'Area', 'Frontage',
                        'TotalFloorArea','Breadth','CoverageRatio', 'FloorAreaRatio', 'TimeToNearestStation',]

        self.path = path


    def read_data(self):
        ''' 输入数据 '''
        # path = "archive/trade_prices/02.csv"
        self.data = pd.read_csv(
                                self.path,
                                usecols=self.columns_name_use,
                                low_memory=False,
                                # dtype={'AreaIsGreaterFlag': pd.np.bool, 'Area': pd.np.int, "LandShape": pd.np.str}
                                )

    # print(data.iloc[:, [13]])    # 获取某列
    # columns_name = data.columns.tolist() # 获取列名字
    # print(columns_name)
    def analysis_price(self):
        '''分析  price'''
        price = self.data.loc[:, ['TradePrice']].values # 获取某列，转为array
        price = [p[0] for p in price]   # 转为list
        s = pd.Series(price)   # 分析 price

        self.A = s.describe()['25%']
        self.B = s.describe()['50%']
        self.C = s.describe()['75%']


        ''' 对标签分类，其实变为分类问题'''

        Y = []
        for i in range(len(self.data)):
            pri = self.data.loc[i, 'TradePrice']
            if pri < self.A:
                Y.append("0 ~ A")
            elif self.A <= pri < self.B:
                Y.append("A ~ B")
            elif self.B <= pri < self.C:
                Y.append("B ~ C")
            else:
                Y.append("C ~ +")
        self.data["TradePrice_class"] = Y


    def encoding(self):
        '''
        对 str 特征 编码:  encoding for features: encode data : transfrom the str features to int features

        '''
        le = sklearn.preprocessing.LabelEncoder()
        for feature in self.str_features:
            self.data.loc[:, feature] = le.fit_transform(self.data[feature].astype(str))
        # print(self.data)

        '''处理int 特征中的空值 : drop the NaN data
        此时， str 特征已经编码了，不存在空值了
        '''
        self.data.dropna(axis=0, how='any', inplace=True)

        # 重新标序
        self.data.index = [i for i in range(self.data.shape[0])]


        '''特殊 int 特征 之  TimeToNearestStation ， 从而 将str 特征转化为 float 特征
            取中间值作为数值
            30-60minutes ：取45 min
            1H-1H30 ：     取值 75 min
            1H30-2H ：     取值105 min
        
        '''
        for i in range(len(self.data)):
            if self.data.loc[i, "TimeToNearestStation"] == "30-60minutes":
                self.data.loc[i, "TimeToNearestStation"] = 45

            elif self.data.loc[i, "TimeToNearestStation"] == "1H-1H30":
                self.data.loc[i, "TimeToNearestStation"] = 75

            elif self.data.loc[i, "TimeToNearestStation"] == "1H30-2H":
                self.data.loc[i, "TimeToNearestStation"] = 105


        # # 删除 ？ 或者用中间值代替 ？
        # float_features_mean =[]


        # 先取部分
        self.data = self.data[:20000]


    def create_train_and_test(self):
        '''分开 train data 和 test data: '''
        self.train, self.test = train_test_split(self.data, test_size=0.3)

        self.X_train = self.train.drop(['TradePrice', 'TradePrice_class'], axis=1)
        self.Y_train = self.train.loc[:, ['TradePrice_class']].values.ravel()

        self.X_test = self.test.drop(['TradePrice', 'TradePrice_class'], axis=1)
        self.Y_test = self.test.loc[:, ['TradePrice_class']].values.ravel()


        # 精确预测 - 回归 用到此标签
        self.Y_test_price = self.test.loc[:, ['TradePrice']].values.ravel()


        ''' train 数据分为4部分 ，4个小 train data， 后期 精准 训练使用'''
        # 0 ~ A 的 训练数据:
        df_0_A = self.train[self.train["TradePrice"] < self.A]
        self.Y_train_0A = df_0_A.loc[:, ['TradePrice']].values.ravel()
        self.X_train_0A = df_0_A.drop(['TradePrice', 'TradePrice_class'], axis=1)

        # A ~ B 的 训练数据:
        df_A_B = self.train[(self.A <= self.train["TradePrice"]) & (self.train["TradePrice"] < self.B)]
        self.Y_train_AB = df_A_B.loc[:, ['TradePrice']].values.ravel()
        self.X_train_AB = df_A_B.drop(['TradePrice', 'TradePrice_class'], axis=1)

        # B ~ C 的 训练数据:
        df_B_C = self.train[(self.B <= self.train["TradePrice"]) & (self.train["TradePrice"] < self.C)]
        self.Y_train_BC = df_B_C.loc[:, ['TradePrice']].values.ravel()
        self.X_train_BC = df_B_C.drop(['TradePrice', 'TradePrice_class'], axis=1)

        # C ~ + 的 训练数据:
        df_C_D = self.train[self.C <= self.train["TradePrice"]]
        self.Y_train_CD = df_C_D.loc[:, ['TradePrice']].values.ravel()
        self.X_train_CD = df_C_D.drop(['TradePrice', 'TradePrice_class'], axis=1)


    def PCA_preprocessing(self, components = 10):
        ''' PCA  主成分分析 降维 -- 训练集 train
            "X_train_PCA": X_train_PCA,
            "X_test_PCA": X_test_PCA,
        '''

        pca = PCA(n_components=components)
        pca.fit(self.X_train)

        # train data:
        self.X_train_PCA = pca.transform(self.X_train)
        # test data:
        self.X_test_PCA = pca.transform(self.X_test)

        # d = {
        #     "X_train_PCA": self.X_train_PCA,
        #     "X_test_PCA": self.X_test_PCA,
        # }

        # return d


    def LDA_preprocessing(self):
        ''' LDA  线性判别分析:
            "X_train_LDA": X_train_LDA,
            "X_test_LDA": X_test_LDA,
            "predict class": predict_class_lda,
            "model": lda
         '''

        lda = LinearDiscriminantAnalysis(
            # n_components=3,
            solver='svd'
        )
        lda.fit(self.X_train, self.Y_train)

        # train data transform:
        self.X_train_LDA = lda.transform(self.X_train)
        # test data transform:
        self.X_test_LDA = lda.transform(self.X_test)
        # predict class of lda:  # 用训练好的lda，预测时，输入是原来的X_test 而不是X_test_LDA, 因为lda模型会自动降维。
        self.predict_class_lda = lda.predict(self.X_test)

        # print("predict_class is" + str(predict_class_lda))
        #
        # d = {
        #     "X_train_LDA": self.X_train_LDA,
        #     "X_test_LDA": self.X_test_LDA,
        #     "predict_class": self.predict_class_lda,
        #     "model": lda
        #     }

        # return the model of lda
        return lda

        # print(X_train)
        # print(X_train_LDA)
        # print(len(X_train_LDA[0]))



# 实例化：
path = "archive/trade_prices/02.csv"
dc = DATA_PREPROCESS(path)      # dc:  data of this city

dc.read_data()
dc.analysis_price()
dc.encoding()
dc.create_train_and_test()


#  LDA 处理：
dc.LDA_preprocessing() #在运行时产生 LDA 数据

#  PCA 处理：
dc.PCA_preprocessing(10)  #在运行时产生 PCA 数据



# ----------------------------------  分类预测: --------------------------------------

'''
 #1   LDA
model_lda = dc.LDA_preprocessing()                 #获取模型
score_lda = model_lda.score(dc.X_test, dc.Y_test)  # test score lda
predict_class_lda = model_lda.predict(dc.X_test)   # predict class

print("LDA : the predict Score is " + str(score_lda))
print("LDA : the predict Class is " + str(predict_class_lda))

'''


'''
#2  LR_L2
d_LR_l2 = Logistic_Regression.Logistic_Regression_Algorithm_l2penalty(dc.X_train_PCA, dc.Y_train, dc.X_test_PCA, dc.Y_test)  #

score_LR_l2 = d_LR_l2["predict_score"]           # test score
predict_class_LR_l2 = d_LR_l2["predict_class"]   # predict class
print("Logistic Regression_L2 : the predict Score is " + str(score_LR_l2))
print("Logistic Regression_L2 : the predict Class is " + str(predict_class_LR_l2))
'''


'''
#3 LR_L1
d_LR_l1 = Logistic_Regression.Logistic_Regression_Algorithm_l1penalty(dc.X_train_PCA, dc.Y_train, dc.X_test_PCA, dc.Y_test)

score_LR_l1 = d_LR_l1["predict_score"]           # test score
predict_class_LR_l1 = d_LR_l1["predict_class"]   # predict class
print("Logistic Regression_L1 : the predict Score is " + str(score_LR_l1))
print("Logistic Regression_L1 : the predict Class is " + str(predict_class_LR_l1))
'''





# 4 Random Forest
d_RF = Random_Forest.Randomforest_Algorithm(dc.X_train_PCA, dc.Y_train, dc.X_test_PCA, dc.Y_test)

score_RF = d_RF["predict_score"]           # test score
predict_class_RF = d_RF["predict_class"]   # predict class

print("Random Forest : the predict Score is " + str(score_RF))
print("Random Forest : the predict Class is " + str(predict_class_RF))




'''
# 5 SVC 
d_SVC = SVC.SVC_Algorithm(dc.X_train_PCA, dc.Y_train, dc.X_test_PCA, dc.Y_test)

score_SVC = d_SVC["predict_score"]           # test score
predict_class_SVC = d_SVC["predict_class"]   # predict class

print("SVC : the predict Score is " + str(score_SVC))
print("SVC : the predict Class is " + str(predict_class_SVC))
'''





''' 精准预测 '''

Linear_Regression.Linear_Regression_Algorithm(
                                        dc.X_train_0A, dc.Y_train_0A,
                                        dc.X_train_AB, dc.Y_train_AB,
                                        dc.X_train_BC, dc.Y_train_BC,
                                        dc.X_train_CD, dc.Y_train_CD,
                                        dc.X_test, dc.Y_test_price,

                                        predict_class_RF

                                        )





















