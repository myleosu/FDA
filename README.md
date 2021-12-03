# FDA
FDA_main.py是FDA线性判别的主程序入口。

- run_one2one_model函数代表FDA识别函数
- run_SVM_model函数代表SVM识别函数
- run_DecisionTree_model函数代表决策树识别函数

FDA_show.py是FDA线性判别结果展示的入口。

如需运行程序，请安装以下环境：
- python >= 3.6
- numpy
- scipy
- sklearn
- Pillow
- matplotlib

安装环境后请在中断输入以下命令：
```
python FDA_show.py
```

运行完成后将输出柱状图和折线图，FDA线性判别改进前后的效果图将存放至./result_image文件夹下。