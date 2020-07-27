import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as knn


"""
函数说明:将32x32的二进制图像转换为1x1024向量。

Parameters:
    filename - 文件名
Returns:
    returnVect - 返回的二进制图像的1x1024向量

Modify:
    2017-07-15
"""
def img2vector(filename):
    returnVect=np.zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect


"""
函数说明:手写数字分类测试

Parameters:
    无
Returns:
    无

Modify:
    2017-07-15
"""
def handwritingClassTest():
    hwlabels=[]
    trainingFileList  = listdir("../../data/mnist/trainingDigits")
    # 返回文件夹个数
    m=len(trainingFileList)
    # 初始化训练集
    trainMat=np.zeros((m,1024))
    # 从文件名解析训练集类别
    for i in range(m):
        fileNameStr=trainingFileList[i]
        classNumber=int(fileNameStr.split('_')[0])
        hwlabels.append(classNumber)
        trainMat[i,:]=img2vector("../../data/mnist/trainingDigits/"+fileNameStr)
    # 构建knn分类器
    neign=knn(n_neighbors=3,algorithm='auto')
    # 拟合模型
    neign.fit(trainMat,hwlabels)
    # 返回testDigits目录下的文件列表
    testFileList=listdir("../../data/mnist/testDigits")
    errorCount=0.0
    mTest=len(testFileList)
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        fileNameStr=testFileList[i]
        classNumber=int(fileNameStr.split('_')[0])
        vectorUnderTest=img2vector("../../data/mnist/testDigits/"+fileNameStr)
        # 获取预测结果
        classifierResult=neign.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if(classifierResult!=classNumber):
            errorCount+=1
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest * 100))

if __name__ == '__main__':
    handwritingClassTest()