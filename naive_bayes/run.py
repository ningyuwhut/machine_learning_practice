#encoding=utf8
from __future__ import division
import random

class NaiveBayes(object):
    def __init__(self, dataset, labels ):
	self.dataset=dataset
	self.labels=labels
	self.p_label={} #每个label出现的次数
    def train(self):
	sample_number=len(self.labels)
	feature_number=len(self.dataset[0])
	uniq_labels=set(self.labels)#所有的类	
	label_sample={} #该label下所有的样本
	for i in range(sample_number):
	    label=self.labels[i]
	    if label in self.p_label:
		self.p_label[label]+=1
		label_sample[label].append(i)
	    else:
		self.p_label[label]=1
		label_sample[label]=[]
		label_sample[label].append(i)

	#拉普拉斯平滑
	class_number=len(self.p_label) #类的个数
	for label in self.p_label: #先验概率
	    self.p_label[label]+=1#/=sample_number
	    self.p_label[label]/=(sample_number+class_number)

	feature_value={} #统计每个特征的所有取值
	for i in range(sample_number):
	    sample=self.dataset[i]
	    for j in range(feature_number):
		if j in feature_value:
		    feature_value[j].add(sample[j])
		else:
		    feature_value[j]=set()
		    feature_value[j].add(sample[j])
	#统计所有条件概率
	self.p_feature_label={}
	for label in self.p_label:
	    for i in feature_value: #特征下标
		all_feature_value=feature_value[i]
		for value in all_feature_value:
		    label_feature=(label, i, value) #类别label下第i个特征取值为value
		    if label_feature not in self.p_feature_label:
			self.p_feature_label[label_feature]=1
	    
	for label in label_sample:
	    sample_in_label=label_sample[label]#该label下的所有样本
	    for i in sample_in_label: #该类下每个样本的下标
		sample=self.dataset[i]
		for j in range(len(sample)):
		    key=(label, j,sample[j])
		    self.p_feature_label[key]+=1 #拉普拉斯
	for key in self.p_feature_label:
	    label=key[0]
	    feature_index=key[1]
	    label_count=len(label_sample[label]) #该类的样本数
	    print "label_count", "label", label_count, self.p_feature_label[key]
	    self.p_feature_label[key]/=(label_count + len(feature_value[feature_index])) #拉普拉斯
	    print "key", key, "p_feature_label", self.p_feature_label[key]
    def predict(self, testset):
	pred_label=[]
	for sample in testset:
	    p_y_value={} #记录属于每个类的概率
	    y_labels=set(self.p_label.keys())
	    for label in y_labels:
		p=self.p_label[label]
		for i in range(len(sample)):
		    p*=self.p_feature_label[(label, i, sample[i])] 
		p_y_value[label]=p
	    p_y_value=sorted(p_y_value.items(), key=lambda e:e[1], reverse=True) #p_value是list
	    print "p_y_value sorted", p_y_value
	    pred_label.append(p_y_value[0][0])
	return pred_label

    def predict_proba(self, testset):
	pred_proba=[]
	for sample in testset:
	    p_y_value={} #记录属于每个类的概率
	    y_labels=set(self.p_label.keys())
	    for label in y_labels:
		p=self.p_label[label]
		for i in range(len(sample)):
		    p*=self.p_feature_label[(label, i, sample[i])] 
		p_y_value[label]=p
	    p_y_value=sorted(p_y_value.items(), key=lambda e:e[1], reverse=True) #p_value是list
	    print "p_y_value sorted", p_y_value
	    print "p_0_1", p_y_value[0][1]
	    pred_proba.append(p_y_value[0][1])
	return pred_proba

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

def auc(labels):

if __name__ == '__main__':
    dataset, labels =read_iris_dataset('car')
    index=range(len(dataset))
    #print "index", index
    random.shuffle(index)
    #print "index", index
    trainset=[ dataset[index[i]] for i in range(1000) ]
    trainset_labels=[ labels[index[i]] for i in range(1000) ]
    testset=[ dataset[index[i]] for i in range(1000, len(index)) ]
    testset_labels=[ labels[index[i]] for i in range(1000, len(index)) ]
    nb=NaiveBayes(trainset, trainset_labels)
    nb.train()
    pred_label=nb.predict(testset)
    correct=0
    for i in range(len(pred_label)):
	if pred_label[i] == testset_labels[i]:
	    correct+=1
    print "correct", correct, "testset", len(pred_label), "precision", correct/len(pred_label)

    pred_proba=nb.predict_proba(testset) #每个样本属于正类的概率
    print "pred_proba", pred_proba
    pred_proba_hash={}
    for i in range(len(pred_proba)):
	if i not in pred_proba_hash:
	    pred_proba_hash[i]=pred_proba[i]
    
    pred_proba_hash_sorted=sorted(pred_proba_hash.items(), key=lambda e:e[1], reverse=True) #p_value是list
    print "pred_proba_hash_sorted", pred_proba_hash_sorted
    corr_testset_label=[]

    for i in pred_proba_hah_sorted:
	corr_testset_label.append(i)
    print "corr_testset_label", corr_testset_label
    auc=auc(testset_labels)
