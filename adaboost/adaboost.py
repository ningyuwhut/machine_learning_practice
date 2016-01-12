#encoding=utf8
import sys
import math
import random
import collections
from collections import Counter
from numpy import inf
from decision_stump import *

class Adaboost(object):
    def __init__(self, dataset, target, iter_number):
	self.dataset=dataset
	self.target=target
	self.iter_number=iter_number
	self.weak_leaners=[]
	self.alphas=[]
	self.weights=[1/float(len(dataset)) for i in range(len(dataset)) ]

    def train(self):
	for i in range(self.iter_number):
	    print "i", i, "self.weights", self.weights
	    weak_leaner=DecisionTree(self.dataset, self.target, self.weights)
	    weak_leaner.build_tree()
	    self.weak_leaners.append(weak_leaner)
	    alpha=float(0.5*math.log( (1-weak_leaner.error_rate)/max(weak_leaner.error_rate,1e-16)) )
	    print "alpha", alpha
	    self.alphas.append(alpha)
	    sample_number=len(self.dataset)
	    for j in range(sample_number):
		self.weights[j]*=math.exp(-1 * alpha * float(self.target[j]) * float(weak_leaner.pred_target[j]))
	    normalization=float(sum(self.weights))
	    for j in range(sample_number):
		self.weights[j]/=float(normalization)
#	    print "iter", i, "weights", self.weights
	for i in range(self.iter_number):
	    self.weak_leaners[i].print_tree()
	    
    def classify(self, testset, testset_labels):
	pred_targets={}
	for i in range(self.iter_number): #每个弱分类器
	    weak_leaner_pred_targets=self.weak_leaners[i].classify(testset)
	    for j in range(len(testset)):
		if j not in pred_targets:
		#    print "i", i, "j", j, "pred_target", len(pred_targets), "weak", len(weak_leaner_pred_targets)
		    pred_targets[j] = self.alphas[i]*float(weak_leaner_pred_targets[j])
		else:
		    pred_targets[j] += self.alphas[i]*float(weak_leaner_pred_targets[j])
	correct=0
	for i in range(len(testset)):
	    if pred_targets[i] >= 0:
		pred_targets[i] = 1
	    else:
		pred_targets[i] = -1

	    if int(testset_labels[i]) == pred_targets[i]:
		correct+=1
	    
	print correct, len(testset), correct/float(len(testset))

def read_libsvm_format_file(dataset_filename):
	dataset_file=file(dataset_filename,'r')
	dataset_label=[]
	dataset=[]
	for line in dataset_file:
		splitted=line.strip().split()
		dataset_label.append(splitted[0])
		sample=[]
		for i in range(1,len(splitted)):
			index_value=splitted[i].split(":")
			sample.append(index_value[1])
		dataset.append(sample)
	return dataset, dataset_label

if __name__ == "__main__":
	dataset, target =read_libsvm_format_file('diabetes')

	for i in dataset:
	    print "\t".join(i)
	    
#	print "dataset", dataset
#	print "target", target
	sys.exit(0)
	trainset_size=500 #wine
	index=range(len(dataset))
	random.shuffle(index) #把数据集打乱
	trainset=[ dataset[index[i]] for i in range(trainset_size) ]
	trainset_target=[ target[index[i]] for i in range(trainset_size) ]

	testset=[ dataset[index[i]] for i in range(trainset_size, len(index)) ]
	testset_target=[ target[index[i]] for i in range(trainset_size, len(index)) ]

	adaboost=Adaboost(trainset, trainset_target, 10)
	adaboost.train()
	#adaboost.classify(testset, testset_target)
