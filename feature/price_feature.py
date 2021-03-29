# 提取特征列
def get_feature_cols(train_data, test_data):
    # 提取数值类型特征列名
    numerical_cols = train_data.select_dtypes(exclude='object').columns
    print(numerical_cols)
    categorical_cols = train_data.select_dtypes(include='object').columns
    print(categorical_cols)
    # 选择特征列
    feature_cols = [col for col in numerical_cols if
                    col not in ['SaleID', 'name', 'regDate', 'creatDate', 'price', 'model', 'brand', 'regionCode',
                                'seller']]
    feature_cols = [col for col in feature_cols if 'Type' not in col]
    # 提取特征列，标签列构造训练样本和测试样本
    x_data = train_data[feature_cols]
    y_data = train_data['price']
    x_test = test_data[feature_cols]
    print('X train shape:', x_data.shape)
    print('X test shape:', x_test.shape)
    return x_data, y_data, x_test
