# --Japanese-house-price-prediction  

准备：首先在TRAIN.py文件的309行 添加 文件地址，给path变量。  
如：path = "/Users/muhaoguo/Documents/study/神州数码/日本房价/archive/trade_prices/02.csv"  
把地址替换成某个.csv 文件的实际地址。  


运行： 直接运行main函数。 第一次使用需要运行2遍main.py  
第一遍是训练分类模型模型：  
（1）运行过程中会自动调用pickle.dump函数，将训练好的模型自动存入同文件夹下对应的.sav文件  
（2）运行过程中会自动生成data文件夹，存储之后用到的训练和测试数据。  
第一遍main函数运行完之后，运行main函数第二遍：  
（1）会生成分类预测结果  
（2）会生成精准分类结果  

再次使用该模型时只需运行1遍main.py,因为省略了训练的过程。  


文件说明：  
MAIN.py: 入口函数  
TRAIN.py: 数据预处理，生成训练集，测试集，已经训练分类预测模型  
TSET.py: 测试分类预测函数（即Random Forest模型）并给出预测正确率  
Data文件夹： 在TRAIN过程中产生，存放数据  
Precise_Predict.py: 第二阶段的精确预测，里面包括训练和测试过程。  
Random_Forest.py: Random Forest模型，在TRAIN和TEST 过程中都会用到。  
Logistic_Regression.py: 2个Logistic_Regression 模型，在TRAN中已经注释掉，不会用到。  
SVC.py: SVC模型，在TRAN中已经注释掉，不会用到。  


