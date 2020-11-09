from TEST import TEST
from TRAIN import TRAIN
from Precise_Predict import Precise_predict_Algorithm
import os


'''
1. if the model has not train, run the main.py first to train and run main.py again to test
2. if there exist a trained model, just run the main.py directly 
'''

model_name = "best_RF_model_no_PCA.sav"
if os.path.exists(model_name):
    predict_class_RF = TEST(model_name)
    PCA_num = "5"     #  None, or int 0-30 in Precise Predict
    Precise_predict_Algorithm(predict_class_RF, PCA_num, 100) # predict_class_RF, PCA_num and draw_num
else:
    TRAIN(model_name)     #Default: No PCA.  if you want to train use PCA, use the part PCA in TRAIN




# PCA: trian 注释
# rf :改名 best_RF_model_no_PCA
# train 函数
# test 的 输入 test pca