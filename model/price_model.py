import lightgbm as lgb
import xgboost as xgb
from code.show_data import *

# 参数搜索和评价
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import mean_absolute_error


# xgb模型函数
def build_model_xgb(x_train, y_train):
    model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.1, gamma=0, subsample=0.8, \
                             colsample_bytree=0.9, max_depth=7)  # , objective ='reg:squarederror'
    model.fit(x_train, y_train)
    return model


# lgb模型函数
def build_model_lgb(x_train, y_train):
    estimator = lgb.LGBMRegressor(num_leaves=127, n_estimators=150)
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
    }
    gbm = GridSearchCV(estimator, param_grid)
    gbm.fit(x_train, y_train)
    return gbm


# 查看模型的参数效果：使用绝对平均误差
def print_mae(x_data, y_data):
    scores_train = []
    scores = []
    # xgb-Model
    xgr = xgb.XGBRegressor(n_estimators=120, learning_rate=0.1, gamma=0, subsample=0.8, \
                           colsample_bytree=0.9, max_depth=7)  # ,objective ='reg:squarederror'
    # 5折交叉验证方式
    sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_ind, val_ind in sk.split(x_data, y_data):
        train_x = x_data.iloc[train_ind].values
        train_y = y_data.iloc[train_ind]
        val_x = x_data.iloc[val_ind].values
        val_y = y_data.iloc[val_ind]

        xgr.fit(train_x, train_y)
        pred_train_xgb = xgr.predict(train_x)
        pred_xgb = xgr.predict(val_x)

        score_train = mean_absolute_error(train_y, pred_train_xgb)
        scores_train.append(score_train)
        score = mean_absolute_error(val_y, pred_xgb)
        scores.append(score)

    print('Train mae:', np.mean(score_train))
    print('Val mae', np.mean(scores))


def train_predict_lgb(x_train, x_val, y_train, y_val, x_data, y_data, x_test):
    print('Train lgb...')
    model_lgb = build_model_lgb(x_train, y_train)
    val_lgb = model_lgb.predict(x_val)
    mae_lgb = mean_absolute_error(y_val, val_lgb)
    print('MAE of val with lgb:', mae_lgb)

    print('Predict lgb...')
    model_lgb_pre = build_model_lgb(x_data, y_data)
    subA_lgb = model_lgb_pre.predict(x_test)
    print('Sta of Predict lgb:')
    print_sta_info(subA_lgb)
    return mae_lgb, val_lgb, subA_lgb


def train_predict_xgb(x_train, x_val, y_train, y_val, x_data, y_data, x_test):
    print('Train xgb...')
    model_xgb = build_model_xgb(x_train, y_train)
    val_xgb = model_xgb.predict(x_val)
    mae_xgb = mean_absolute_error(y_val, val_xgb)
    print('MAE of val with xgb:', mae_xgb)

    print('Predict xgb...')
    model_xgb_pre = build_model_xgb(x_data, y_data)
    subA_xgb = model_xgb_pre.predict(x_test)
    print('Sta of Predict xgb:')
    print_sta_info(subA_xgb)
    return mae_xgb, val_xgb, subA_xgb


# 简单的加权融合
def simple_combine_weight(mae_lgb, mae_xgb, val_lgb, val_xgb, subA_xgb, subA_lgb, y_val):
    val_weighted = (1 - mae_lgb / (mae_xgb + mae_lgb)) * val_lgb + (1 - mae_xgb / (mae_xgb + mae_lgb)) * val_xgb
    # 预测的最小值有负数，而真实情况下，price为负是不存在的，由此进行对应的后修正
    val_weighted[val_weighted < 0] = 10
    print('MAE of val with Weighted ensemble:', mean_absolute_error(y_val, val_weighted))
    sub_weighted = (1 - mae_lgb / (mae_xgb + mae_lgb)) * subA_lgb + (1 - mae_xgb / (mae_xgb + mae_lgb)) * subA_xgb
    return sub_weighted
