# --Japanese-house-price-prediction

1.运行： 
直接运行main函数。



2.数据位置说明：
请确保原始数据在main函数相同的文件夹下。
如果原始数据位置有变，只需在main函数中，DATA_PREPROCESS类 实例化之前可以手动输入文件地址，每次输入的文件地址是一个具体的csv文件的地址（对应某一个县的房屋价格数据），而不是文件夹地址。



3.函数说明：

(1） 分类预测：
运行main函数，会得到以下模型的分类预测结果。因为已经选择出 Random Forest 为最佳模型，所以除Random Forest 外，其他模型已经注释掉，若想运行其他模型，首先取消注释。
LDA （已注释掉）
LR_L2（已注释掉）
LR_L1（已注释掉）
Random Forest（正常运行）
SVC （已注释掉）

（2）精准预测：
main函数最下面是精准预测函数，默认是使用 Random Forest的预测结果作为 输入参数。

