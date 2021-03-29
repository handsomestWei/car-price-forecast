import numpy as np


# 打印数据集信息
def print_data_info(data_name, data):
    # 输出数据的大小信息
    print(data_name + ' data shape:', data.shape)
    # 数据简要浏览：读取数据的形式
    data.head()
    # 数据信息查看：对应一些数据列名，以及NAN缺失信息
    data.info()
    # 数据统计信息浏览：数值特征列的一些统计信息
    data.describe()


# 打印标签统计信息
def print_sta_info(data):
    # 统计标签的基本分布信息
    print('sta of label:')
    print('_min', np.min(data))
    print('_max:', np.max(data))
    print('_mean', np.mean(data))
    print('_ptp', np.ptp(data))
    print('_std', np.std(data))
    print('_var', np.var(data))
