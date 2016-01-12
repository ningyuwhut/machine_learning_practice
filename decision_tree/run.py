#encoding=utf8
import sys
import math
import random

class DecisionTreeNode(object):
	def __init__(self,split_feature=-1, feature_value=-1):
		self.split_feature=split_feature
		self.feature_value=feature_value
		self.childs=None
		self.label=None

	def get_split_feature(self):
		return self.split_feature
	def set_split_feature(self,split_feature):
		self.split_feature=split_feature
	def get_feature_value(self):
		return self.feature_value
	def set_feature_value(self,feature_value):
		self.feature_value=feature_value
	def get_childs(self):
		return self.childs
	def set_childs(self,childs):
		self.childs=childs
	def get_label(self):
		return self.label
	def set_label(self, label):
		self.label=label

class DecisionTree(object):
	def __init__(self, dataset , labels):
		self.dataset=dataset
		self.labels=labels
		self.root=None

	def build_tree(self):
		feature_list=[ 1 for i in range(len(self.dataset[0])) ] 
		self.root=self.build_tree_sub(self.dataset, self.labels, feature_list)	

	def build_tree_sub(self, dataset, labels, feature_list):
	#print "dataset", dataset
		for i in range(len(dataset)):
			print i, dataset[i]

		print "labels", labels
		print "feature_list", feature_list
		if len(dataset) == 1: #只有一个样本，此时不需要再选择特征，直接将该样本的target作为该叶子节点的目标值
			leaf=DecisionTreeNode()
			leaf.set_label(labels[0])
			return leaf
		elif len(set(labels))==1: #剩余样本都属于同一类
			leaf=DecisionTreeNode()
			leaf.set_label(labels[0])
			return leaf
		elif feature_list.count(1) == 0:#只有一个特征可用
			leaf=DecisionTreeNode()
			leaf.set_label( self.majority( labels ) )
			return leaf
		else:
			split_feature=self.choose_feature(dataset,labels, feature_list)
			#根据这个特征把数据集分成几个部分
			print "split_feature", split_feature
			tmp_hash=self.split_dataset_by_feature( dataset, split_feature)
			child_nodes=[]
			parent_feature_list=feature_list[:]
			parent_feature_list[split_feature]=-1 #该特征不再用来分裂
			for i in tmp_hash:
				subset=[ dataset[j] for j in tmp_hash[i] ]
				corr_labels=[ labels[j] for j in tmp_hash[i]]
				child=self.build_tree_sub(subset, corr_labels, parent_feature_list)
				child.set_feature_value(subset[0][split_feature])
				child_nodes.append(child)
			parent_node=DecisionTreeNode(split_feature)
			parent_node.set_childs(child_nodes)
			return parent_node
	def classify(self, testset):
		target=[]
		for sample in testset:
			print "sample", sample
			target.append(self.classify_a_sample(sample))
		return target
	def classify_a_sample(self, sample):
		node=self.root
		print "root", node.get_split_feature(), len(node.get_childs())
		while True:
			if node.get_label() != None:
				return node.get_label()
			else:
				split_feature=node.get_split_feature() #当前节点用来分裂的特征
				split_feature_value=sample[split_feature] #该样本该分裂特征的值
				print "node", split_feature, split_feature_value
				a=True
				for child in node.get_childs():
					print "child", child.get_feature_value(), split_feature_value
					if child.get_feature_value() == split_feature_value:
						node=child
						a=False
						break
				if a==True:
					print "here"
					return "failure"
	
	def print_tree(self):
		self.print_tree_sub(self.root, 1)
	def print_tree_sub(self, node, depth):
		if node.label != None:
			print "depth" +"\t"+str(depth) + "\t"+ "label"+"\t"+str(node.get_label()) +"\t" +"feature_value" +"\t"+ node.get_feature_value()
		else:
			print "depth" +"\t"+str(depth) + "\t"+"split_feature"+"\t"+ str(node.get_split_feature())+"\t"+"feature_value", str(node.get_feature_value())
			for  child in node.get_childs():
				self.print_tree_sub(child, depth+1)
	
	def majority(labels):
	#首先统计每个label的出现次数
	#然后求次数最多的label作为叶子节点的类标
		count_hash={}
		for i in labels:
			if i in count_hash:
				count_hash[i]+=1
			else:
				count_hash[i]=1
		maximum_count=0
		majority_label=None
		for i in count_hash:
			if count_hash[i] > maximum_count:
				maximum_count=count_hash[i]
				majority_label=i
		
		return majority_label

	

#根据某个特征的取值将数据集分成几个部分，每个部分该特征的取值相同
	def split_dataset_by_feature(self, dataset, feature_index):
		sample_number=len(dataset)
		feature_list=[ dataset[j][feature_index] for j in range( sample_number) ]

		tmp_hash={}
		for j in range(sample_number):
			if feature_list[j] in tmp_hash:
				tmp_hash[feature_list[j]].append(j)
			else:
				tmp_hash[feature_list[j]]=[j]
		return tmp_hash

#feature_index:用来分割的特征下标
	def information_gain(self, dataset, labels, feature_list, feature_index):
		sample_number=len(self.labels)
		dataset_entropy=self.entropy(labels)
		tmp_hash=self.split_dataset_by_feature(dataset, feature_index)

		#首先，计算该特征的所有取值，以及每个取值对应的下标集合
		tmp_entropy=0.0
		for k in tmp_hash:
			tmp_label=[ labels[i] for i in tmp_hash[k]]
			tmp_entropy+=len(tmp_hash[k])/float(sample_number)*self.entropy(tmp_label)

		information_gain=dataset_entropy-tmp_entropy

		return information_gain

		
	#index是要计算熵的元素集合的下标构成的列表 
	def entropy(self, labels):
	#compute the occurance of each label in labels
		count={}
		sample_number=len(labels)
		for i in range(sample_number):
			if labels[i] in count:
				count[labels[i]]+=1
			else:
				count[labels[i]]=1
		entropy=0.0
		for i in count.keys():
			p_i=float(count[i])/sample_number#attention how to get the result of division
			entropy -= p_i * math.log(p_i,2)
		return entropy

	#统计每个特征的信息增益
	#dataset 为数据集
	#labels为对应的目标变量
	#feature_list为可以用来分割的特征，若值为1表示该特征可以用于分裂
	def choose_feature(self, dataset, labels, feature_list):
		print "in choose feature"
		#print "dataset", dataset
		for i in range(len(dataset)):
			print i, dataset[i]

		print "labels", labels
		print "feature_list", feature_list
		target_feature_index=-1
		max_information_gain=-1
		for i in range(len(feature_list)):
			if feature_list[i] == 1:
				ig=self.information_gain(dataset, labels, feature_list, i)
				print "feature", i, "gain", ig
				if ig > max_information_gain:
					max_information_gain=ig
					target_feature_index=i
		return target_feature_index		

def read_dataset( dataset_filename ):
	dataset_file=file(dataset_filename,'r')
	dataset_label=[]
	dataset=[]
	for line in dataset_file:
		splitted=line.strip().split()
	
		dataset_label.append(int(splitted[0]))
		sample=[]
		for i in range(1,len(splitted)):
			feature=splitted[i].split(":")
			sample.append(int(feature[1]))
		dataset.append(sample)
	return dataset, dataset_label

def read_weather_dataset(dataset_filename):
	dataset_file=file(dataset_filename,'r')
	dataset_label=[]
	dataset=[]
	for line in dataset_file:
		splitted=line.strip().split("\t")
		dataset_label.append(splitted[4])
		sample=[]
		for i in range(0,len(splitted)-1):
			#feature=splitted[i].split(":")
			sample.append(splitted[i])
		dataset.append(sample)
	return dataset[1:], dataset_label[1:]

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

if __name__ == '__main__':
	dataset=[
		[0,0,0,0],
		[0,0,0,1],
		[0,1,0,1],
		[0,1,1,0],
		[0,0,0,0],
		[1,0,0,0],
		[1,0,0,1],
		[1,0,1,1],
		[1,1,1,2],
		[1,0,1,2],
		[2,0,1,2],
		[2,0,1,1],
		[2,1,0,1],
		[2,1,0,2],
		[2,0,0,0]
		]
	#print len(dataset)
	test_sample=[2,1,1,1]
	labels=[0,0,1,1,0,0,0,1,1,1,1,1,1,1,0]
	dataset, labels=read_dataset('./545/train')

#	print dataset
#	print len(dataset), len(dataset[0])
#	print labels
#	testset, testset_labels=read_dataset('./545/test')
#	print testset
#	print len(testset), len(testset[0])
#	print testset_labels
#
#	dataset, labels=read_weather_dataset('weather_dataset')
#	testset=[dataset[-1]]

	dataset, labels =read_iris_dataset('car')
	print "dataset", dataset
	print "label", labels
	index=range(len(dataset))
	print "index", index
	random.shuffle(index)
	print "index", index
	trainset=[ dataset[index[i]] for i in range(1000) ]
	trainset_file=file('trainset', 'w')
	trainset_labels=[ labels[index[i]] for i in range(1000) ]

	k=0
	for i in trainset:
		trainset_file.write("\t".join(i)+"\t"+str(trainset_labels[k])+"\n")
		k+=1

	testset=[ dataset[index[i]] for i in range(1000, len(index)) ]
	testset_labels=[ labels[index[i]] for i in range(1000, len(index)) ]
	k=0
	testset_file=file('testset', 'w')
	for i in testset:
		testset_file.write("\t".join(i)+"\t"+str(testset_labels[k])+"\n")
		k+=1

	print "trainset", trainset
	print "trainset_label", trainset_labels
	print "testset", testset
	print "testset_label", testset_labels
	print "trainset", len(trainset)
	print "trainset_label", len(trainset_labels)
	print "testset", len(testset)
	print "testset_label", len(testset_labels)

#	trainset, trainset_labels=read_iris_dataset('lenses.txt')

	test_tree=DecisionTree(trainset, trainset_labels)
	#print test_tree.entropy(labels)
	print "start to build tree"
	test_tree.build_tree()
	print "print tree"
	test_tree.print_tree()
#	sys.exit(0)
   # print test_tree.classify_a_sample(test_sample)
	pred_label=test_tree.classify(testset)
	#print "pred_label", str(pred_label)+"\t"+str(testset_labels)
	print len(pred_label)
	for i in range(len(pred_label)):
		print str(pred_label[i])+"\t"+str(testset_labels[i])

	correct=0
	for i in range(len(pred_label)):
		if pred_label[i] == testset_labels[i]:
			correct+=1
	print correct, len(pred_label), len(testset_labels), float(correct)/len(pred_label)
