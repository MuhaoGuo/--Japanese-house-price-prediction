import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import Logistic_Regression
import Random_Forest
import SVC
import Precise_Predict
import os
import time


class DATA_PREPROCESS:
    # '''特征 '''

    def __init__(self, path):
        self.columns_name = ['Type', 'Region', 'MunicipalityCode', 'Prefecture', 'Municipality', 'DistrictName',
                             'NearestStation',
                             'TimeToNearestStation', 'MinTimeToNearestStation', 'MaxTimeToNearestStation', 'TradePrice',
                             'FloorPlan',
                             'Area', 'AreaIsGreaterFlag', 'UnitPrice', 'PricePerTsubo', 'LandShape', 'Frontage',
                             'FrontageIsGreaterFlag',
                             'TotalFloorArea', 'TotalFloorAreaIsGreaterFlag', 'BuildingYear', 'PrewarBuilding',
                             'Structure', 'Use',
                             'Purpose', 'Direction', 'Classification', 'Breadth', 'CityPlanning', 'CoverageRatio',
                             'FloorAreaRatio',
                             'Period', 'Year', 'Quarter', 'Renovation', 'Remarks']

        self.columns_name_use = ['Type', 'Region', 'MunicipalityCode', 'Prefecture', 'DistrictName', 'NearestStation',
                                 'TimeToNearestStation', 'MinTimeToNearestStation', 'MaxTimeToNearestStation',
                                 'TradePrice', 'FloorPlan',
                                 'Area', 'AreaIsGreaterFlag', 'LandShape', 'Frontage', 'FrontageIsGreaterFlag',
                                 'TotalFloorArea',
                                 'TotalFloorAreaIsGreaterFlag', 'BuildingYear', 'PrewarBuilding', 'Structure', 'Use',
                                 'Purpose', 'Direction', 'Classification', 'Breadth', 'CityPlanning', 'CoverageRatio',
                                 'FloorAreaRatio',
                                 'Year', 'Quarter']

        self.str_features = ['Type', 'Region', 'MunicipalityCode', 'Prefecture', 'DistrictName', 'NearestStation',
                             'FloorPlan', 'AreaIsGreaterFlag', 'LandShape', 'FrontageIsGreaterFlag',
                             'TotalFloorAreaIsGreaterFlag', 'BuildingYear',
                             'PrewarBuilding', 'Structure', 'Use', 'Purpose', 'Direction', 'Classification',
                             'CityPlanning', 'Year', 'Quarter', ]

        self.float_features = ['MinTimeToNearestStation', 'MaxTimeToNearestStation', 'TradePrice', 'Area', 'Frontage',
                               'TotalFloorArea', 'Breadth', 'CoverageRatio', 'FloorAreaRatio', 'TimeToNearestStation', ]

        self.path = path
        self.data = None
        self.A = None
        self.B = None
        self.C = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.Y_test_price = None

    def read_data(self):
        ''' 输入数据 '''
        # path = "archive/trade_prices/02.csv"
        self.data = pd.read_csv(
            self.path,
            usecols=self.columns_name_use,
            low_memory=False,
            # dtype={'AreaIsGreaterFlag': pd.np.bool, 'Area': pd.np.int, "LandShape": pd.np.str}
        )

    def analysis_price(self):
        '''分析  price  ,25% ，50% , 75% 分位 '''
        price = self.data.loc[:, ['TradePrice']].values  # 获取某列，转为array
        price = [p[0] for p in price]  # 转为list
        s = pd.Series(price)  # 分析 price

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
        对 str 特征 编码:  encoding for features: encode data : transfrom the str features to int feature
        '''
        le = sklearn.preprocessing.LabelEncoder()
        for feature in self.str_features:
            self.data.loc[:, feature] = le.fit_transform(self.data[feature].astype(str))
        # print(self.data)

        '''处理 float 特征中的空值 : drop the NaN data , 所有空值直接舍弃：
        原因：  1. 数据量大， 完整的样本够用
               2. 对于float 型数据，如房面积，离车站距离等，主观判断会直接影响房屋价格，故此处不宜填充 众数/均值等。
        '''
        self.data.dropna(axis=0, how='any', inplace=True)

        # # 删除 ？ 或者用中间值代替 ？
        # float_features_mean =[]

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

        # 先取部分
        # print(len(self.data))
        self.data = self.data[:20000]

    def create_train_and_test(self):
        '''分开 train data 和 test data: '''

        # 分离 训练集 测试集 X,Y
        self.train, self.test = train_test_split(self.data, test_size=0.3)

        self.X_train = self.train.drop(['TradePrice', 'TradePrice_class'], axis=1)
        self.Y_train = self.train.loc[:, ['TradePrice_class']]

        self.X_test = self.test.drop(['TradePrice', 'TradePrice_class'], axis=1)
        self.Y_test = self.test.loc[:, ['TradePrice_class']]

        # 重新标序
        self.X_train.index = [i for i in range(self.X_train.shape[0])]
        self.Y_train.index = [i for i in range(self.Y_train.shape[0])]
        self.X_test.index = [i for i in range(self.X_test.shape[0])]
        self.Y_test.index = [i for i in range(self.Y_test.shape[0])]

        # 训练集 写入文件，测试集写入文件
        self.X_train.to_csv('./data/X_train.csv', index='False')
        self.Y_train.to_csv('./data/Y_train.csv', index='False')
        self.X_test.to_csv('./data/X_test.csv', index='False')
        self.Y_test.to_csv('./data/Y_test.csv', index='False')

        # Y 标签转化为 一维 list
        self.Y_train = self.Y_train.values.ravel()
        self.Y_test = self.Y_test.values.ravel()

        # 精确预测 - 回归 用到此标签
        self.Y_test_price = self.test.loc[:, ['TradePrice']]  # .values.ravel()
        self.Y_test_price.to_csv('./data/Y_test_price.csv', index='False')

        ''' train 数据分为4部分 ，4个小 train data， 后期 精准 训练使用'''
        # 0 ~ A 的 训练数据:
        df_0_A = self.train[self.train["TradePrice"] < self.A]
        self.Y_train_0A = df_0_A.loc[:, ['TradePrice']]  # .values.ravel()
        self.X_train_0A = df_0_A.drop(['TradePrice', 'TradePrice_class'], axis=1)

        # A ~ B 的 训练数据:
        df_A_B = self.train[(self.A <= self.train["TradePrice"]) & (self.train["TradePrice"] < self.B)]
        self.Y_train_AB = df_A_B.loc[:, ['TradePrice']]  # .values.ravel()
        self.X_train_AB = df_A_B.drop(['TradePrice', 'TradePrice_class'], axis=1)

        # B ~ C 的 训练数据:
        df_B_C = self.train[(self.B <= self.train["TradePrice"]) & (self.train["TradePrice"] < self.C)]
        self.Y_train_BC = df_B_C.loc[:, ['TradePrice']]  # .values.ravel()
        self.X_train_BC = df_B_C.drop(['TradePrice', 'TradePrice_class'], axis=1)

        # C ~ + 的 训练数据:
        df_C_D = self.train[self.C <= self.train["TradePrice"]]
        self.Y_train_CD = df_C_D.loc[:, ['TradePrice']]  # .values.ravel()
        self.X_train_CD = df_C_D.drop(['TradePrice', 'TradePrice_class'], axis=1)

        # 小训练集 写入文件
        self.Y_train_0A.to_csv('./data/Y_train_0A.csv', index='False')
        self.X_train_0A.to_csv('./data/X_train_0A.csv', index='False')
        self.Y_train_AB.to_csv('./data/Y_train_AB.csv', index='False')
        self.X_train_AB.to_csv('./data/X_train_AB.csv', index='False')
        self.Y_train_BC.to_csv('./data/Y_train_BC.csv', index='False')
        self.X_train_BC.to_csv('./data/X_train_BC.csv', index='False')
        self.Y_train_CD.to_csv('./data/Y_train_CD.csv', index='False')
        self.X_train_CD.to_csv('./data/X_train_CD.csv', index='False')

    def PCA_preprocessing(self, components=10):
        ''' PCA  主成分分析 降维 -- 训练集 train
            "X_train_PCA": X_train_PCA,
            "X_test_PCA": X_test_PCA,
        '''

        pca = PCA(n_components=components)
        pca.fit(self.X_train)

        # train data:
        self.X_train_PCA = pca.transform(self.X_train)  # PCA
        self.X_train_PCA = pd.DataFrame(self.X_train_PCA)  # array to df
        self.X_train_PCA.to_csv('./data/X_train_PCA.csv', index='False')

        # test data:
        self.X_test_PCA = pca.transform(self.X_test)  # PCA
        self.X_test_PCA = pd.DataFrame(self.X_test_PCA)  # array to df
        self.X_test_PCA.to_csv('./data/X_test_PCA.csv', index='False')

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

        return lda


'''
# ----------------------------------  分类预测 函数: --------------------------------------
 #1   LDA
def predict_LDA(dc):
    model_lda = dc.LDA_preprocessing()                 #获取模型
    score_lda = model_lda.score(dc.X_test, dc.Y_test)  # test score lda
    predict_class_lda = model_lda.predict(dc.X_test)   # predict class

    print("LDA : the predict Score is " + str(score_lda))
    print("LDA : the predict Class is " + str(predict_class_lda))
    return predict_class_lda

#2  LR_L2  with PCA
def predict_LR_L2(dc):
    d_LR_l2 = Logistic_Regression.Logistic_Regression_Algorithm_l2penalty(dc.X_train_PCA, dc.Y_train, dc.X_test_PCA, dc.Y_test)  #
    score_LR_l2 = d_LR_l2["predict_score"]           # test score
    predict_class_LR_l2 = d_LR_l2["predict_class"]   # predict class
    print("Logistic Regression_L2 : the predict Score is " + str(score_LR_l2))
    print("Logistic Regression_L2 : the predict Class is " + str(predict_class_LR_l2))
    return predict_class_LR_l2

#3 LR_L1 with PCA
def predict_LR_L1(dc):
    d_LR_l1 = Logistic_Regression.Logistic_Regression_Algorithm_l1penalty(dc.X_train_PCA, dc.Y_train, dc.X_test_PCA, dc.Y_test)

    score_LR_l1 = d_LR_l1["predict_score"]           # test score
    predict_class_LR_l1 = d_LR_l1["predict_class"]   # predict class
    print("Logistic Regression_L1 : the predict Score is " + str(score_LR_l1))
    print("Logistic Regression_L1 : the predict Class is " + str(predict_class_LR_l1))
    return predict_class_LR_l1



# 4 Random Forest with PCA
def predict_random_forest(dc):

    d_RF = Random_Forest.Randomforest_Algorithm(dc.X_train_PCA, dc.Y_train, dc.X_test_PCA, dc.Y_test,)

    # score_RF = d_RF["predict_score"]           # test score
    # predict_class_RF = d_RF["predict_class"]   # predict class
    #
    # print("Random Forest : the predict Score is " + str(score_RF))
    # print("Random Forest : the predict Class is " + str(predict_class_RF))

    # return predict_class_RF

# 5 SVC with PCA
def predict_svc(dc):
    d_SVC = SVC.SVC_Algorithm(dc.X_train_PCA, dc.Y_train, dc.X_test_PCA, dc.Y_test)

    score_SVC = d_SVC["predict_score"]           # test score
    predict_class_SVC = d_SVC["predict_class"]   # predict class

    print("SVC : the predict Score is " + str(score_SVC))
    print("SVC : the predict Class is " + str(predict_class_SVC))
    return predict_class_SVC
'''


def TRAIN(model_name):
    outdir = './data'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # 预处理
    path = "./archive/trade_prices/02.csv"
    dc = DATA_PREPROCESS(path)  # dc:  data of this city
    dc.read_data()
    dc.analysis_price()
    dc.encoding()
    dc.create_train_and_test()

    # ----------------------------------  降维函数 : --------------------------------------
    # #  LDA 处理： 之后没有用到
    # dc.LDA_preprocessing() #在运行时产生 LDA 数据
    # print("LDA Processing has Done ! ")

    # '''
    # PCA 处理：
    print("PCA Processing is running ...")
    # dc.PCA_preprocessing(10)   #在运行过程中产生 PCA 数据，在dc 属性中
    print("PCA Processing has Done ! \n")
    # '''

    '''
    # ----------------------------------  分类预测 函数: --------------------------------------

    # 这些函数 不需要
    # predict_class_lda = predict_LDA(dc)

    # predict_class_LR_l2 = predict_LR_L2(dc)

    # predict_class_LR_l1 = predict_LR_L1(dc)

    # predict_class_SVC = predict_svc(dc)
    '''

    # print("we need train the model first")
    print("-------------TRAIN------------------")
    time_start = time.time()
    print("Random Forest algorithm is training ...")

    # predict_random_forest(dc)                              #random forest with PCA
    Random_Forest.Randomforest_Algorithm(dc.X_train, dc.Y_train, dc.X_test, dc.Y_test, model_name)

    print("Random Forest algorithm training has finished")
    print("NOW, RUN THE TRAIN.py AGAIN")
    time_end = time.time()
    print('time cost', time_end - time_start, 's')


'''
如果不把数据保存下，那么每次进行test的时候其实是 又 预处理了一遍。
'''






















