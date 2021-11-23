from FDA_main import run_one2one_model, run_SVM_model, run_DecisionTree_model
from pylab import *
from PIL import Image

from RGB_Color_Generate import color
mpl.rcParams['font.sans-serif'] = ['SimHei']


#给每个柱子上面添加标注
def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
              xy=(rect.get_x() + rect.get_width() / 2, height),
              xytext=(0, 3),  # 3 points vertical offset
              textcoords="offset points",
              ha='center', va='bottom')
    
if __name__ == "__main__":
    test_size_numbers = [i*0.1 for i in range(9, 4, -1)]
    one2one_model_result = []
    SVM_model_result = []
    DecisionTree_model_result = []

    one2one_model_result_up = []
    SVM_model_result_up = []
    DecisionTree_model_result_up = []
    for i in range(5):
        print("test_size:{}".format(test_size_numbers[i]))
        oa, aa, K, uu, result_image = run_one2one_model(test_size=test_size_numbers[i])
        one2one_model_result.append([oa, aa, K, uu])
        result_image = Image.fromarray(np.uint8(result_image))
        result_image.save("./result_image/Fisher_test_size_{:.1f}_oa_{:.2f}_image.jpeg".format(test_size_numbers[i], oa), 'JPEG', quality=95)
        
        oa, aa, K, uu = run_SVM_model(test_size=test_size_numbers[i])
        SVM_model_result.append([oa, aa, K, uu])
        oa, aa, K, uu = run_DecisionTree_model(test_size=test_size_numbers[i])
        DecisionTree_model_result.append([oa, aa, K, uu])

        oa, aa, K, uu, result_image = run_one2one_model(test_size=test_size_numbers[i], is_dim_reduction=True)
        one2one_model_result_up.append([oa, aa, K, uu])
        result_image = Image.fromarray(np.uint8(result_image))
        result_image.save("./result_image/Fisher_Up_test_size_{:.1f}_oa_{:.2f}_image.jpeg".format(test_size_numbers[i], oa), 'JPEG', quality=95)
        
        oa, aa, K, uu = run_SVM_model(test_size=test_size_numbers[i], is_dim_reduction=True)
        SVM_model_result_up.append([oa, aa, K, uu])
        oa, aa, K, uu = run_DecisionTree_model(test_size=test_size_numbers[i], is_dim_reduction=True)
        DecisionTree_model_result_up.append([oa, aa, K, uu])
    
    one2one_model_result = np.array(one2one_model_result)
    SVM_model_result = np.array(SVM_model_result)
    DecisionTree_model_result = np.array(DecisionTree_model_result)

    one2one_model_result_up = np.array(one2one_model_result_up)
    SVM_model_result_up = np.array(SVM_model_result_up)
    DecisionTree_model_result_up = np.array(DecisionTree_model_result_up)

    # plt.plot([(1 - i * 0.1) for i in range(5, 10)], one2one_model_result[:, 0], 'ro-', color='#4169E1', alpha=0.8, label='Fisher')
    # plt.plot([(1 - i * 0.1) for i in range(5, 10)], SVM_model_result[:, 0], 'ro--', color='#90EE90', alpha=0.8, label='SVM')
    # plt.plot([(1 - i * 0.1) for i in range(5, 10)], DecisionTree_model_result[:, 0], 'ro-.', color='#FF1493', alpha=0.8, label='DecisionTree')

    # plt.plot([(1 - i * 0.1) for i in range(5, 10)], SVM_model_result_up[:, 0], 'ro--', color='#FFA500', alpha=0.8, label='SVM_Up')
    # plt.plot([(1 - i * 0.1) for i in range(5, 10)], DecisionTree_model_result_up[:, 0], 'ro-.', color='#FFFF00', alpha=0.8, label='DecisionTree_Up')
    # plt.legend(loc='best')
    # plt.xlabel('训练集比例')
    # plt.ylabel('IndiaP数据集QA')

    plt.figure(1)
    fig, ax = plt.subplots(figsize=(10, 8))
    x = np.arange(len(test_size_numbers))
    width = 0.35
    bars1 = plt.bar(x - width/2, one2one_model_result[:, 0], color='#4169E1', width=width, alpha=0.8, align='center', label='Fisher')
    bars2 = plt.bar(x + width/2, one2one_model_result_up[:, 0], color='#90EE90', width=width, alpha=0.8, align='center', label='Fisher_New')
    ax.set_xticks(x)
    ax.set_xticklabels(["{:.1f}".format((1 - i)) for i in test_size_numbers])
    autolabel(ax, bars1)
    autolabel(ax, bars2)
    plt.legend(loc='best')
    plt.xlabel("训练集比例")
    plt.ylabel('IndiaP数据集OA')

    plt.figure(2)
    fig, ax = plt.subplots(figsize=(10, 8))
    x = np.arange(len(test_size_numbers))
    width = 0.35
    bars1 = plt.bar(x - width/2, SVM_model_result[:, 0], color='#FF1493', width=width, alpha=0.8, align='center', label='SVM')
    bars2 = plt.bar(x + width/2, SVM_model_result_up[:, 0], color='#FFA500', width=width, alpha=0.8, align='center', label='SVM_New')
    ax.set_xticks(x)
    ax.set_xticklabels(["{:.1f}".format((1 - i)) for i in test_size_numbers])
    autolabel(ax, bars1)
    autolabel(ax, bars2)
    plt.legend(loc='best')
    plt.xlabel("训练集比例")
    plt.ylabel('IndiaP数据集OA')

    plt.figure(3)
    fig, ax = plt.subplots(figsize=(10, 8))
    x = np.arange(len(test_size_numbers))
    width = 0.35
    bars1 = plt.bar(x - width/2, DecisionTree_model_result[:, 0], color='#FFFF00', width=width, alpha=0.8, align='center', label='DecisionTree')
    bars2 = plt.bar(x + width/2, DecisionTree_model_result_up[:, 0], color='#008080', width=width, alpha=0.8, align='center', label='DecisionTree_New')
    ax.set_xticks(x)
    ax.set_xticklabels(["{:.1f}".format((1 - i)) for i in test_size_numbers])
    autolabel(ax, bars1)
    autolabel(ax, bars2)
    plt.legend(loc='best')
    plt.xlabel("训练集比例")
    plt.ylabel('IndiaP数据集OA')

    plt.show()