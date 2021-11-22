from math import floor
import scipy.io
import numpy as np
import random
import itertools
from sklearn import svm
from sklearn.model_selection import train_test_split
from copy import deepcopy

file_name = "indiaP.mat"
no_classes = 16

def train_FDA(x1,x2):
    n= x1.shape[1]
    m1 = np.mean(x1,axis = 0).reshape(1,n)
    m2 = np.mean(x2,axis = 0).reshape(1,n)
    S1 = np.dot((x1 - m1).T,(x1 - m1))
    S2 = np.dot((x2 - m2).T,(x2 - m2))
    SW = S1 + S2
    W = np.dot(np.linalg.pinv(SW),(m1-m2).T)
    
    w0 = (np.dot(m1, W) + np.dot(m2, W)) / 2
    # w0 = -1/2*np.dot((m1+m2),W) #阈值
    return W,w0

def train_one2one_FDA_Model():
    '''
        FDA one2one 模型训练函数
        Input:
            None
        return:
            W_numbers: 两两标签二分类的W
            W0_numbers: 两两标签二分类的W0阈值
    '''
    W_numbers = []
    W0_numbers = []
    for i in range(1, no_classes + 1):
        for j in range(i + 1, no_classes + 1):
            W, w0 = train_FDA(train_data_dict[str(i)], train_data_dict[str(j)])
            W_numbers.append(W)
            W0_numbers.append(w0)
    return W_numbers, W0_numbers

def train_one2rest_FDA_Model():
    '''
        FDA one2rest 模型训练函数
        Input:
            None
        return:
            W_numbers: one2rest的W
            W0_numbers: one2rest的W0阈值
    '''
    W_numbers = []
    W0_numbers = []
    m_numbers = []
    for i in range(1, no_classes + 1):
        X1 = train_data_dict[str(i)]
        X2 = None
        for j in range(1, no_classes + 1):
            if j == i:
                continue
            if X2 is None:
                X2 = train_data_dict[str(j)]
            else:
                X2 = np.append(X2, train_data_dict[str(j)], axis = 0)    
        W, w0 = train_FDA(X1, X2)
        W_numbers.append(W)
        W0_numbers.append(w0)
        n = X1.shape[1]
        m_numbers.append(np.dot(np.mean(X1,axis = 0).reshape(1,n), W))
    return W_numbers, W0_numbers, m_numbers

def test_one2one_FDA_Model(x, W_numbers, W0_numbers):
    '''
        FDA one2one 模型预测函数
        Input:
            x: 测试的数据
            W_numbers: one2one的W数组
            W0_numbers: one2one的W0数组
        Return:
            y: 预测x的标签y
    '''
    cnt_numbers = [0 for i in range(no_classes)]
    ind = -1
    for i in range(1, no_classes + 1):
        for j in range(i + 1, no_classes + 1):
            ind = ind + 1
            y = np.dot(x, W_numbers[ind])
            if y >= W0_numbers[ind]:
                cnt_numbers[i - 1] += 1
            else:
                cnt_numbers[j - 1] += 1
    max_index = list.index(cnt_numbers, max(cnt_numbers))
    y = max_index + 1
    return y

def test_one2rest_FDA_Model(x, W_numbers, W0_numbers, m_numbers):
    '''
        FDA one2rest 模型预测函数
        Input:
            x: 测试的数据
            W_numbers: one2rest的W数组
            W0_numbers: one2rest的W0数组
        Return:
            y: 预测x的标签y
    '''
    predict_y = -1
    thre = 0x3f3f3f3f
    for i in range(no_classes):
        y = np.dot(x, W_numbers[i])
        if y >= W0_numbers[i] and abs(y - m_numbers[i]) < thre:
            predict_y = i + 1
            thre = abs(y - m_numbers[i])
    return predict_y

def test_between_two_classes(flag1, flag2):
    '''
        测试数据集中二分类的准确率
        Input:
            flag1: 类别1的数字
            flag2: 类别2的数字
        Return:
            QA: 准确率
    '''
    W, w0 = train_FDA(train_data_dict[str(flag1)], train_data_dict[str(flag2)])
    total_cnt = 0
    right_cnt = 0
    for i in range(X_test.shape[0]):
        if y_test[i] != flag1 and y_test[i] != flag2:
            continue
        total_cnt = total_cnt + 1
        y = np.dot(X_test[i], W)
        if y >= w0 and y_test[i] == flag1:
            right_cnt = right_cnt + 1
        if y < w0 and y_test[i] == flag2:
            right_cnt = right_cnt + 1
    return (right_cnt * 1.0) / total_cnt

def test_between_one2other_classes(flag):
    '''
        测试数据集中二分类的准确率
        Input:
            flag: 类别的数字
        Return:
            QA: flag classes vs other classes 准确率
    '''
    X1 = deepcopy(train_data_dict[str(flag)])
    X2 = None
    for i in range(1, no_classes + 1):
        if i == flag:
            continue
        t = deepcopy(train_data_dict[str(i)])
        if X2 is None:
            X2 = t
        else:
            X2 = np.append(X2, t, axis = 0)   
    W, w0 = train_FDA(X1, X2)
    total_cnt = 0
    right_cnt = 0
    for i in range(X_test.shape[0]):
        total_cnt += 1
        y = np.dot(X_test[i], W)
        if y >= w0 and y_test[i] == flag:
            right_cnt += 1
        if y < w0 and y_test[i] != flag:
            right_cnt += 1
    return (right_cnt * 1.0 / total_cnt)

def normalization(data):
    _range = np.max(abs(data))
    return data / _range

def my_train_test_split(useful_data, useful_data_label, test_size=0.9, random_state=None):
    if random_state != None:
        random.seed(random_state)
    total_data_index_dict = {'1':[], '2':[], '3':[], '4':[], '5':[], '6':[], '7':[], '8':[], '9':[], '10':[],
                 '11':[], '12':[], '13':[], '14':[], '15':[], '16':[]}
    for i in range(useful_data_label.shape[0]):
        total_data_index_dict[str(useful_data_label[i])].append(i)
    train_size_n = int(useful_data_label.shape[0] * (1 - test_size))
    max_n = train_size_n // no_classes
    X_train_indexes = []
    for i in range(1, no_classes + 1):
        random.shuffle(total_data_index_dict[str(i)])
        if len(total_data_index_dict[str(i)]) > max_n:
            X_train_indexes.append([total_data_index_dict[str(i)][ind] for ind in range(max_n)])
        else:
            X_train_indexes.append([total_data_index_dict[str(i)][ind] for ind in range(len(total_data_index_dict[str(i)]) // 2)])
    X_train_indexes = sum(X_train_indexes, [])
    if len(X_train_indexes) > train_size_n:
        X_train_indexes = random.sample(X_train_indexes, train_size_n)
    elif len(X_train_indexes) < train_size_n:
        add_indexes = random.sample([i for i in range(useful_data_label.shape[0])], train_size_n - len(X_train_indexes))
        X_train_indexes = [X_train_indexes]
        X_train_indexes.append(add_indexes)
        X_train_indexes = sum(X_train_indexes, [])
    random.shuffle(X_train_indexes)
    X_train, X_test, y_train, y_test = [], [], [], []
    for i in range(len(X_train_indexes)):
        X_train.append(useful_data[X_train_indexes[i]])
        y_train.append(useful_data_label[X_train_indexes[i]])
    X_train_indexes.sort(reverse=False)
    Y_test_indexes = []
    for ind in range(len(X_train_indexes)):
        i = X_train_indexes[ind]
        if ind == len(X_train_indexes) - 1:
            j = useful_data_label.shape[0]
        else:
            j = X_train_indexes[ind + 1]
        if i != j:
            Y_test_indexes.append([k for k in range(i+1, j)])
    Y_test_indexes = sum(Y_test_indexes, [])
    random.shuffle(Y_test_indexes)
    for i in range(len(Y_test_indexes)):
        X_test.append(useful_data[Y_test_indexes[i]])
        y_test.append(useful_data_label[Y_test_indexes[i]])
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

def confusion(true_label, estim_label):
    '''
        检验函数
        Input:
            true_label: 真实标签
            estim_label: 预测标签
        Return:
            oa, aa, K, ua
    '''
    n = max(true_label)
    confu = np.zeros((n, n))
    for i in range(true_label.shape[0]):
        confu[estim_label[i] - 1, true_label[i] - 1] += 1
    confu_diag = np.diagonal(confu)
    oa = np.sum(confu_diag) / np.sum(confu)
    ua = np.zeros(n)
    for i in range(ua.shape[0]):
        if np.sum(confu, axis=1)[i] != 0:
            ua[i] = confu_diag[i] / np.sum(confu, axis=1)[i]
    aa = np.sum(ua) / n
    
    Po = oa
    Pe = (np.sum(confu, axis=0) @ np.sum(confu, axis=1)) / (np.power(np.sum(confu), 2))

    K = (Po - Pe) / (1 - Pe)

    return oa, aa, K, ua

def loadData():
    '''
        加载IndiaP数据集
    '''
    # data = scipy.io.loadmat(file_name)
    # img = data['img']
    # GroundT_data = data['GroundT']
    # img_h, img_w, img_l = img.shape
    # img_data = img.reshape((img_h * img_w, img_l))
    data_files_name = "IndiaPData.txt"
    data_Label_name = "IndiaPLabel.txt"

    label_list = []
    with open(data_Label_name,'r') as f:
        for line in f:
            label_list.append(list(line.strip('\n').split('   ')[1:3]))

    for i in range(len(label_list)):
        for j in range(2):
            label_list[i][j] = int(eval(label_list[i][j]))

    data_list = []
    with open(data_files_name,'r') as f:
        for line in f:
            data_list.append(list(line.strip('\n').split('  ')[1:201]))
    
    for i in range(len(data_list)):
        for j in range(200):
            data_list[i][j] = int(eval(data_list[i][j]))
    return np.array(data_list), np.array(label_list)

print("正在加载数据...")
img_data, GroundT_data = loadData()
useful_data = []
useful_data_label = []

for i in range(GroundT_data.shape[0]):
    useful_data.append(img_data[ GroundT_data[i][0] - 1 ])
    useful_data_label.append(GroundT_data[i][1])

useful_data = np.array(useful_data)
# useful_data = normalization(np.array(useful_data))# 归一化数据
useful_data_label = np.array(useful_data_label)
print("加载数据完毕！")

X_train, X_test, y_train, y_test = my_train_test_split(useful_data, useful_data_label, test_size=0.9, random_state=1234)
# X_train, X_test, y_train, y_test = train_test_split(useful_data, useful_data_label, test_size=0.9, random_state=1)

clf = svm.SVC(kernel='linear', decision_function_shape='ovo', max_iter=1e6)
clf.fit(X_train, y_train.ravel())
print("SVM训练集准确率：{}".format(clf.score(X_train, y_train)))
print("SVM测试集准确率: {}".format(clf.score(X_test, y_test)))

train_data_dict = {'1':[], '2':[], '3':[], '4':[], '5':[], '6':[], '7':[], '8':[], '9':[], '10':[],
                 '11':[], '12':[], '13':[], '14':[], '15':[], '16':[]}
for i in range(X_train.shape[0]):
    train_data_dict[str(y_train[i])].append(X_train[i])
for i in range(no_classes):
    train_data_dict[str(i + 1)] = np.array(train_data_dict[str(i + 1)])
    print("{} label number is :{}".format(i+1, train_data_dict[str(i + 1)].shape))

# ***************************************************************测试各分类器的QA

# total_acc = 0.0
# for i in range(1, no_classes + 1):
#     total_acc += test_between_one2other_classes(i)
#     print("[{} , others] -> {}".format(i, test_between_one2other_classes(i)))
# print(total_acc / no_classes)

# ****************************************************************一对多 OVR

# one2rest_W_numbers, one2rest_W0_numbers, one2rest_m_numbers = train_one2rest_FDA_Model()


# right_numbers = 0
# for col in range(X_test.shape[0]):
#     predict_y = test_one2rest_FDA_Model(X_test[col], one2rest_W_numbers, one2rest_W0_numbers, one2rest_m_numbers)
#     if predict_y == y_test[i]:
#         right_numbers += 1
# print("y_test shape : {}".format(y_test.shape))
# print("predict right numbers : {}".format(right_numbers))
# print("QA: {}".format(right_numbers / y_test.shape[0]))
# print("total_cnt_sum: {}".format(total_cnt_sum))

# ****************************************************************一对一 OVO

one2one_W_numbers, one2one_W0_numbers = train_one2one_FDA_Model()

estim_labels = []
for col in range(X_test.shape[0]):
    predict_y = test_one2one_FDA_Model(X_test[col], one2one_W_numbers, one2one_W0_numbers)
    estim_labels.append(predict_y)
estim_labels = np.array(estim_labels)
oa , aa, K, ua = confusion(y_test, estim_labels)
print("Fisher模型测试集检验结果：\noa : {}\naa : {}\nK:{}\nua:{}".format(oa, aa, K, ua.reshape(no_classes, 1)))

# ***************************************************************测试各分类器的QA

# t = 0.0
# for i in range(1, no_classes + 1):
#     for j in range(i + 1, no_classes + 1):
#         t += test_between_two_classes(i, j)
#         print("[{} , {}] -> {}".format(i, j, test_between_two_classes(i, j)))
# print(t / (no_classes * (no_classes - 1) / 2))