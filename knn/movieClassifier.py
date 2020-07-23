import numpy as np
import operator


def createDataSet():
    group=np.array([[1,101],[5,89],[108,5],[115,8]])
    labels=['爱情片','爱情片','动作片','动作片']
    return group,labels


"""
函数说明:kNN算法,分类器
Parameters:
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labes - 分类标签
    k - kNN算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果
"""
def classfy0(inx,dataSet,labels,k):
    # 行数
    dataSetSize=dataSet.shape[0]
    # 在列向量方向上重复inX共1次(横向)，
    # 行向量方向上重复inX共dataSetSize次(纵向)
    diffMat=np.tile(inx,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    print(sqDiffMat)
    # sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqlDistance=sqDiffMat.sum(axis=1)
    # 开方求距离
    distance=sqlDistance**0.5
    #返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distance.argsort()
    #定一个记录类别次数的字典
    classCount={}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel]=classCount.get(voteLabel,0)+1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    group,labels = createDataSet()
    test=[101,20]
    test_class = classfy0(test, group, labels, 3)
    print(test_class)

