#encoding=utf8
import sys
import math
import random
import collections
from collections import Counter
from numpy import inf
from cart_regression import  *

class GBDT_CLASSIFICATION(object):
    #target必须是用数值表示类别
    def __init__(self, dataset, target, iter_number):
	self.dataset=dataset
	self.target=target
	self.iter_number=iter_number
	class_number=len(set(target)) #类别数目
	self.weak_leaners=[[ 0 for i in range(class_number) ] for j in range(iter_number)] #存储所有的树,每次迭代都会建立k个树

    def train(self):
	sample_number=len(self.target)
	class_number=len(set(self.target)) #类别数目
	p_matrix=[ [0 for i in range(class_number)] for j in range(sample_number) ] #每个样本属于每个类的概率

	f_matrix=[ [0 for i in range(class_number)] for j in range(sample_number) ] #每个样本关于每个类的函数预测值 , 用来计算p_matri的
	for i in range(self.iter_number): #每次迭代
	    #首先根据逻辑斯蒂变换把函数值变换为每个样本属于每个类的概率
	    for j in range(sample_number): #每个样本
		normalization=0.0
		#计算样本属于每个类的概率，这是逻辑斯蒂变换
		for k in range(class_number): #每个类别
		    p_matrix[j][k]=math.exp( f_matrix[j][k] )
		    normalization+=p_matrix[j][k]

		for k in range(class_number):
		    p_matrix[j][k]/=float(normalization)
    
	    class_label=list(set(self.target))
	    print "class_label", class_label
	    #为每个类建立决策树
	    for k in range(class_number):
		target_update= [ 0.0 for m in range(sample_number) ] #初始化每个样本属于类k的概率
		for j in range(sample_number):
		    target_j=-1
		    if self.target[j] == class_label[k]: #k这里有问题
			target_j = 1
		    else:
			target_j = 0

		    target_update[j]=target_j - p_matrix[j][k] #这是梯度，也是跟残差版本的联系

		print "p_matrix", p_matrix
		print "f_matrix", f_matrix
		print "target_update", target_update
		self.weak_leaners[i][k]=DecisionTree(self.dataset, target_update) #对第k类建立一个回归树
		self.weak_leaners[i][k].build_tree( class_number )
		gain=self.weak_leaners[i][k].get_gain(self.dataset ) #其实这个可以在树中加入记录训练集样本预测值的功能
		for m in range(sample_number):
		    f_matrix[m][k] += gain[m]
    
    def classify(self, testset ):
	sample_number=len(testset)
	class_number=len(set(self.target)) #类别数目
	target=[] #[ -1 for j in range(sample_number) ] #先初始化每个样本的类别
	for sample in testset:
	    sample_target=self.classify_a_sample(sample)
	    target.append(sample_target)
	return target

    #对样本进行分类时，把每棵树的预测值相加，得到每个类的预测值，再经过逻辑斯蒂变换，取取概率最大的类作为预测值
    def classify_a_sample(self, sample):
	class_number=len(set(self.target)) #类别数目
	p_list=[0.0 for i in range(class_number) ] #样本属于每个类的概率
	for i in range(class_number):#每个类
	    for j in range(self.iter_number): #每次迭代中的回归树
		p_list[i]+=self.weak_leaners[j][i].get_gain_a_sample(sample)
	
	normalization=0.0
	#计算样本属于每个类的概率，这是逻辑斯蒂变换
	for k in range(class_number): #每个类别
	    p_list[k]=math.exp( p_list[k] )
	    normalization+=p_list[k]

	for k in range(class_number):
	    p_list[k]/=float(normalization)
	
	maximum=max(p_list)
	index= [i for i, j in enumerate(p_list) if j == maximum ]

	class_label=list(set(self.target))
	print "class_index", index[0]
	return class_label[index[0]]

def read_iris_dataset(dataset_filename):
	dataset_file=file(dataset_filename,'r')
	dataset_label=[]
	dataset=[]
	for line in dataset_file:
		splitted=line.strip().split("\t")
		dataset_label.append(splitted[-1])
		sample=[]
		for i in range(0,len(splitted)-1):
			#feature=splitted[i].split(":")
			sample.append(splitted[i])
		dataset.append(sample)
	return dataset, dataset_label
if __name__=='__main__':
	
	dataset, target =read_iris_dataset('car')
	trainset_size=105 #iris
	#trainset_size=125 #wine
	trainset_size=1000 #car
#	print "dataset", dataset
#	print "label", target
	index=range(len(dataset))
#	print "index", index
	random.shuffle(index) #把数据集打乱
	trainset=[ dataset[index[i]] for i in range(trainset_size) ]
	trainset_target=[ target[index[i]] for i in range(trainset_size) ]
	testset=[ dataset[index[i]] for i in range(trainset_size, len(index)) ]
	testset_target=[ target[index[i]] for i in range(trainset_size, len(index)) ]

	print "len", len(set(trainset_target))
	gbdt=GBDT_CLASSIFICATION(trainset, trainset_target, 8)
	gbdt.train()
	print "here"
	pred_target=gbdt.classify(testset)

	print len(pred_target)
	for i in range(len(pred_target)):
		print str(pred_target[i])+"\t"+str(testset_target[i])

	correct=0
	for i in range(len(pred_target)):
		if pred_target[i] == testset_target[i]:
			correct+=1
	print "precision"
	print correct, len(pred_target), len(testset_target), float(correct)/len(pred_target)
