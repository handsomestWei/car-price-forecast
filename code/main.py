import pandas as pd
import matplotlib.pyplot as plt
import warnings
# 参数搜索和评价
from sklearn.model_selection import train_test_split
from model.price_model import *
from feature.price_feature import *

warnings.filterwarnings('ignore')
# Ipython编译器的魔法函数，功能是可以内嵌绘图，省略plt.show()。在Pycharm不支持，需要显式编码
# %matplotlib inline

if __name__ == "__main__":
    # 2、数据读取
    train_data = pd.read_csv('./data/used_car_train_20200313.csv', sep=' ')
    testA_data = pd.read_csv('./data/used_car_testB_20200421.csv', sep=' ')
    print_data_info("Train", train_data)
    print_data_info("TestA", testA_data)

    # 3、特征与标签构建
    x_data, y_data, x_test = get_feature_cols(train_data, testA_data)
    print_sta_info(y_data)
    # 绘制标签的统计图，查看标签分布
    plt.hist(y_data)
    plt.show()
    plt.close()
    # 缺省值用-1填补
    x_data = x_data.fillna(-1)
    x_test = x_test.fillna(-1)

    # 4、模型训练与预测
    print_mae(x_data, y_data)
    # 切分数据集，进行模型训练，评价和预测
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.3)
    mae_lgb, val_lgb, subA_lgb = train_predict_lgb(x_train, x_val, y_train, y_val, x_data, y_data, x_test)
    mae_xgb, val_xgb, subA_xgb = train_predict_xgb(x_train, x_val, y_train, y_val, x_data, y_data, x_test)
    # 两模型的结果加权融合
    sub_weighted = simple_combine_weight(mae_lgb, mae_xgb, val_lgb, val_xgb, subA_lgb, subA_xgb, y_val)
    # 输出结果
    sub = pd.DataFrame()
    sub['SaleID'] = testA_data.SaleID
    sub['price'] = sub_weighted
    sub.to_csv('./prediction_result/predictions.csv', index=False)
    sub.head()
