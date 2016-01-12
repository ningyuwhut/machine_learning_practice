#encoding=gbk
import sys
import math
import random
import collections

class DecisionTreeNode(object):
	def __init__(self,split_feature=-1, feature_value=-1):
		self.split_feature=split_feature
		self.feature_value=feature_value
		self.childs=None
		self.label=None
		self.feature_type="discrete" #Ĭ������ɢ������
		self.greater_or_less="equal"

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
	#Ϊ���������������õ�����
	def get_feature_type(self):
	    return self.feature_type
	def set_feature_type(self, feature_type):
	    self.feature_type=feature_type
	def set_greater_or_less(self, greater_or_less):
	    self.greater_or_less=greater_or_less

	def get_greater_or_less(self):
	    return self.greater_or_less

class DecisionTree(object):
	def __init__(self, dataset , labels):
		self.dataset=dataset
		self.labels=labels
		self.root=None
		self.feature_value={} #��¼ÿ������������Щȡֵ
		for i in range(len(self.dataset)):#ÿ������
			for j in range(len(self.dataset[i])):#ÿ������
				if j not in self.feature_value:
					self.feature_value[j]=set()
				self.feature_value[j].add(self.dataset[i][j])

		self.feature_type={} #��¼��������ɢ�ͻ�������������
		for i in self.feature_value:#�����±�
			continuous=False
			for value in self.feature_value[i]:#����������ȡֵ
				if type(value) == float:#����ȡֵΪ������
					self.feature_type[i]='continuous'
					continuous=True
					break
			if continuous== False : 
				if len(self.feature_value[i]) >= 10: #����ȡֵ��Ŀ����10������Ϊ����������
					self.feature_type[i]='continuous'
				else:
					self.feature_type[i]='discrete'

	def build_tree(self):
		feature_list=[ 1 for i in range(len(self.dataset[0])) ] 
		self.root=self.build_tree_sub(self.dataset, self.labels, feature_list)	

	def build_tree_sub(self, dataset, labels, feature_list):
		#������Ըĳɱ����������ĳ�������ٽ��з���,���Կ�����һ��Ԥ��֦�ķ���
		if len(dataset) == 1: #ֻ��һ����������ʱ����Ҫ��ѡ��������ֱ�ӽ���������target��Ϊ��Ҷ�ӽڵ��Ŀ��ֵ
			leaf=DecisionTreeNode()
			leaf.set_label(labels[0])
			return leaf
		elif len(set(labels))==1: #ʣ������������ͬһ��
			leaf=DecisionTreeNode()
			leaf.set_label(labels[0])
			return leaf
		elif feature_list.count(1) == 0:#û����������ʹ��
			leaf=DecisionTreeNode()
			leaf.set_label( self.majority( labels ) )
			return leaf
		else:
			split_feature, split_feature_value, subset_hash=self.choose_feature(dataset,labels, feature_list)
			#����������������ݼ��ֳɼ�������
			print "split_feature", split_feature
			print "split_feature_value", split_feature_value
			child_nodes=[]
			parent_feature_list=feature_list[:]
			#������ɢ������֧�������ݼ�ʱ�������ݼ��в��ٰ�������������Ϊÿ����֧�µ������ݼ���������ȡֵ�ͻ���һ���ģ���Ϣ�������Gini Gain�����ٱ仯��������������������֧ʱ������֧�µ������ݼ��������ɰ�������������Ȼ�����ҷ�֧�������ķֱ���ȡֵС�ڡ����ڵ��ڷ���ֵ�������ݼ�������Ϊ�����������ٽ�����������֧�����п����������ž��������á�
			if self.feature_type[split_feature] =="discrete": #�����������ɢ�ģ��Ǻ��������������������,��������������������Ȼ
			#���������ָ�
			    parent_feature_list[split_feature]=-1 
			#������������ʱ����Ҫȷ��С�ڷ���ֵ�������ߵ����֧�����ڵ��ߵ��ҷ�֧������һ��Լ��
			#���ڵ�ʵ���������������ĵ�һ�����ӽڵ������֧,�ڶ������ҷ�֧
			print "subset_hash_1", subset_hash, type(subset_hash)
			for i in subset_hash:
				print "i_1", i	
				subset=[ dataset[j] for j in subset_hash[i] ]
				corr_labels=[ labels[j] for j in subset_hash[i]]
				child=self.build_tree_sub(subset, corr_labels, parent_feature_list)
				if self.feature_type[split_feature] == "discrete":
				    child.set_feature_value(subset[0][split_feature])
				else:
				    child.set_feature_value(split_feature_value)
				    child.set_feature_type("continuous")
				    if i == 1:
					child.set_greater_or_less("less")
				    elif i == 2:
					child.set_greater_or_less("greater")

				child_nodes.append(child)
			parent_node=DecisionTreeNode(split_feature)
			parent_node.set_childs(child_nodes)
			if self.feature_type[split_feature] == "continuous":
			    parent_node.set_feature_type("continuous")
			#������ɢ����������Ҫ���ú��ӽڵ������ֵ��
			#������������������Ҫ���õ�ǰ����ڵ������ֵ
			#��ʵ��������������Ҳ�������ú��ӽڵ��ֵ������ҪԼ�����ı��Ǵ��ڣ��ı���С�ڣ������ڷ���ʱ����֪��Ӧ�����ı���
			#����,Ϊ��ͳһ������������õ�ǰ�ڵ������ֵ��һ����,��Ϊȫ�����ú��ӽڵ��ֵ
#			if self.feature_type[split_feature] == "discrete":
#			    parent_node.set_feature_value(split_feature_value)
			return parent_node
	def classify(self, testset):
		target=[]
		for sample in testset:
			print "sample", sample
			target.append(self.classify_a_sample(sample))
		return target
	def classify_a_sample(self, sample):
		node=self.root
	#	print "root", node.get_split_feature(), len(node.get_childs())
		print "sample", sample
		while True:
			if node.get_label() != None:
				print "label", node.get_label()
				return node.get_label()
			else:
				split_feature=node.get_split_feature() #��ǰ�ڵ��������ѵ�����
				split_feature_value=sample[split_feature] #�������÷���������ֵ
				print "node", split_feature, split_feature_value
				if self.feature_type[split_feature]=="discrete":
				    classified=True
				    for child in node.get_childs():
					    print "child", child.get_feature_value(), split_feature_value #, child.get_greater_or_less()
					    if child.get_feature_value() == split_feature_value:
						    node=child
						    classified=False
						    break
				    if classified==True:
					    return "failure"
				else:#����������
				    childs=node.get_childs()
				    print "childs", childs
				    for i in childs:
					print i, i.get_feature_value(),i.get_greater_or_less()
				#����һ��Ҫת��Ϊfloat����Ϊû��ʹ���κεĿ⣬ֱ�Ӵ��ļ��ж�ȡ���������е���������Ĭ�϶����ַ���������
				#�ڶ��������з���ʱ��Ҫ������ֵ�Ƚϴ�С����ʱ���������ֵת��Ϊ������
				    if float(split_feature_value) < float(childs[0].get_feature_value()):#��ǰ�����ĸ�������ֵС�ڷ��ѵ��ֵ��������
					print "type", type(split_feature_value)
					print split_feature_value, "less than childs[0].get_feature_value()", childs[0].get_feature_value()
					node=childs[0]
				    else:
					print "type", type(split_feature_value)
					print split_feature_value, "more than childs[0].get_feature_value()", childs[1].get_feature_value()
					node=childs[1]
	
	def print_tree(self):
		self.print_tree_sub(self.root, 1)
	#��ӡ��ʱ�����ô��ӡ
	def print_tree_sub(self, node, depth):
		if node.label != None:#Ҷ�ӽڵ�
			if node.get_feature_type()=="discrete":
			    print "depth" +"\t"+str(depth) + "\t"+ "label"+"\t"+str(node.get_label()) +"\t" +"feature_value" +"\t"+ node.get_feature_value()
			else:
			    if node.get_greater_or_less() == "greater":
				print "depth" +"\t"+str(depth) + "\t"+ "label"+"\t"+str(node.get_label()) +"\t" +"feature_value" +"\t"+ " >= " +"\t" + str(node.get_feature_value())
			    else:
				print "depth" +"\t"+str(depth) + "\t"+ "label"+"\t"+str(node.get_label()) +"\t" +"feature_value" +"\t"+ " < " +"\t" + str(node.get_feature_value())
				
		else: #��Ҷ�ӽڵ�
			if node.get_feature_type()=="discrete":
			    print "depth" +"\t"+str(depth) + "\t"+"split_feature"+"\t"+ str(node.get_split_feature())+"\t"+"feature_value", str(node.get_feature_value())
			else:#��������
			    if node.get_greater_or_less() == "greater":
				print "depth" +"\t"+str(depth) + "\t"+ "split_feature"+"\t"+ str(node.get_split_feature())+"\t" + "feature_value" +"\t"+ " >= " +"\t" + str(node.get_feature_value())
			    elif node.get_greater_or_less() == "less":
				print "depth" +"\t"+str(depth) + "\t"+ "split_feature"+"\t"+ str(node.get_split_feature())+"\t" + "feature_value" +"\t"+ " < " +"\t" + str(node.get_feature_value())
			    else:
				print "depth" +"\t"+str(depth) + "\t"+ "split_feature"+"\t"+ str(node.get_split_feature())+"\t" + "feature_value" +"\t"+ " = " +"\t" + str(node.get_feature_value())
			for  child in node.get_childs():
				self.print_tree_sub(child, depth+1)
	
	def majority(labels):
	#����ͳ��ÿ��label�ĳ��ִ���
	#Ȼ�����������label��ΪҶ�ӽڵ�����
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

#����ĳ��������ȡֵ�����ݼ��ֳɼ������֣�ÿ�����ָ�������ȡֵ��ͬ
#tmp_hash��¼��������ÿ��ȡֵ��Ӧ�������������±�
	def split_dataset_by_feature(self, dataset, feature_index, feature_type, feature_value):
		sample_number=len(dataset)
		tmp_hash={}
		if feature_type == "discrete":
		    feature_list=[ dataset[j][feature_index] for j in range( sample_number) ] #ÿ�������ĸ�������ȡֵ
		    for j in range(sample_number):
			    if feature_list[j] in tmp_hash:
				    tmp_hash[feature_list[j]].append(j)
			    else:
				    tmp_hash[feature_list[j]]=[j]
		return tmp_hash

#feature_index:�����ָ�������±�
#������ɢ��������Ϣ������
	def information_gain_ratio_for_discrete_feature(self, dataset, labels, feature_list, feature_index):
		sample_number=len(self.labels)
		dataset_entropy=self.entropy(labels)
		tmp_hash=self.split_dataset_by_feature(dataset, feature_index, "discrete", 0.0) #�����������������ɢ��������û���õ�

		#���ȣ����������������ȡֵ���Լ�ÿ��ȡֵ��Ӧ���±꼯��
		tmp_entropy=0.0
		for k in tmp_hash:
			tmp_label=[ labels[i] for i in tmp_hash[k]]
			tmp_entropy+=len(tmp_hash[k])/float(sample_number)*self.entropy(tmp_label)

		information_gain_ratio=(dataset_entropy-tmp_entropy)/float(dataset_entropy)

		return information_gain_ratio,tmp_hash

		
	#index��Ҫ�����ص�Ԫ�ؼ��ϵ��±깹�ɵ��б� 
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

	#ͳ��ÿ����������Ϣ�����ʣ���Ϣ��������������Ϊ��������
	#������Ϣ������ʱ����ɢ���������������Ĵ������ǲ�ͬ��
	#����������Ҫѡ��һ�����ŷ��ѵ㣬���з��ѵ����������ķ��ѵ㼴���ŷ��ѵ㣬�õ㴦����Ϣ��������Ϊ����������Ϣ������
	#dataset Ϊ���ݼ�
	#labelsΪ��Ӧ��Ŀ�����
	#feature_listΪ���������ָ����������ֵΪ1��ʾ�������������ڷ���
	#
	def choose_feature(self, dataset, labels, feature_list):
		print "in choose feature"
		target_feature_index=-1
		max_information_gain_ratio=0.0
		best_split_value=0.0
		best_subset_hash={}
		for i in range(len(feature_list)): 
			if feature_list[i] == 1: #ÿ�����õ�����
				#��������ʱ����Ҫ�������������ݼ��ֳɼ����֣���������������ɽڵ�ĺ��ӽڵ�ʱ���õ���Ϊ�����ظ����֣����Խ�������Ϊ�������
				if self.feature_type[i]== "discrete":
					info_gain_ratio, subset_hash=self.information_gain_ratio_for_discrete_feature(dataset, labels, feature_list, i)
				elif self.feature_type[i] == "continuous":
					split_value, info_gain_ratio, subset_hash=self.infomation_gain_ratio_for_continuous_feature(dataset, labels, feature_list, i)

				print "feature", i, "gain_ratio", info_gain_ratio

				if info_gain_ratio > max_information_gain_ratio:
					max_information_gain_ratio=info_gain_ratio
					target_feature_index=i
					best_subset_hash=subset_hash
					if self.feature_type[i] == "continuous":
					    best_split_value=split_value
		print "best_feature", target_feature_index
		return target_feature_index, best_split_value, best_subset_hash


	#Ϊ����������ѡ����ѵ�
	def infomation_gain_ratio_for_continuous_feature( self, dataset, labels, feature_list, feature_index ):
		#ÿ�������±�Ͷ�Ӧ�ĸ�������ȡֵ�Ͷ�Ӧ�����
		print "in infomation_gain_ratio_for_continuous_feature"
		print "feature_index", feature_index
		feature_value_label={}
		feature_value_set=set()
		for i in range(len(dataset)):
			feature_value=float(dataset[i][feature_index])
			feature_value_set.add(feature_value)
			feature_label=labels[i]
			feature_value_label[i]=[i, feature_value, feature_label]

		sorted_feature_value=list(feature_value_set)
		sorted_feature_value.sort()
		print "sorted_feature_value", sorted_feature_value

		feature_value_label_sorted = sorted(feature_value_label.items(), key=lambda x : x[1][1] ) #������ȡֵ��С��������, ����һ���б�, �б��ÿ��Ԫ����һ��Ԫ��
		print "feature_value_label_sorted", feature_value_label_sorted
		#�����еķ��ѵ���ѡ��Ϣ�������ķ��ѵ������ŷ��ѵ�,��ʱ����Ϣ������Ϊ��������������Ϣ���棬���ݴ˼�����Ϣ������
		max_info_gain=0.0
		best_split_value=0.0
		best_subset_hash={}
		split_value_number=len(sorted_feature_value)-1 #���ѵ����
		for i in range(split_value_number):
			split_value=(sorted_feature_value[i]+sorted_feature_value[i+1])/2
#			print "split_value_1", split_value
			info_gain, subset_hash=self.information_gain_for_continuous_feature(dataset, labels, feature_list, feature_value_label_sorted, feature_index, split_value , split_value_number )
			print "info_gain", info_gain
	
			if info_gain > max_info_gain :
				max_info_gain=info_gain
				best_split_value=split_value
				best_subset_hash=subset_hash
				print "best_split_value", best_split_value

		dataset_entropy=self.entropy(labels)

		best_gain_ratio=max_info_gain/dataset_entropy
		print "best_split_value", best_split_value
		print "best_gain_ratio", best_gain_ratio
		
		return best_split_value, best_gain_ratio, best_subset_hash

	#����ĳ��������ĳ�����ѵ����Ϣ����
	#�÷��ѵ�����ݼ��ֳ������֣����ڷ��ѵ����һ���֣����ڷ��ѵ��������һ���֣��������ַ����µ���Ϣ����
	#��һ�����������������ͬ�ķ��ѵ��ָ���ô����
	def information_gain_for_continuous_feature(self, dataset, labels, feature_list, feature_value_label_sorted,
		    feature_index, split_value, split_value_number):
		subset_hash=collections.OrderedDict() #{}
		split_feature_index=0
		for i in range( len(feature_value_label_sorted) ):
		    if feature_value_label_sorted[i][1][1] > split_value:
			split_feature_index=i
			break
		print "split_value", split_value
		print "split_feature_index", split_feature_index

		subset_hash[1]=[x[1][0] for x in feature_value_label_sorted[0:split_feature_index] ] #����ȡֵС��split_value�������±꼯��
		subset_hash[2]=[x[1][0] for x in feature_value_label_sorted[split_feature_index:] ] #����ȡֵС��split_value�������±꼯��

		subset_label_hash={}
		subset_label_hash[1]=[x[1][2] for x in feature_value_label_sorted[0:split_feature_index] ] #����ȡֵС��split_value��������꼯��
		subset_label_hash[2]=[x[1][2] for x in feature_value_label_sorted[split_feature_index:] ] #����ȡֵС��split_value��������꼯��
		sample_number=len(self.labels)
		dataset_entropy=self.entropy(labels)

		tmp_entropy=0.0
		for k in subset_hash:
			tmp_entropy+=len(subset_hash[k])/float(sample_number)*self.entropy(subset_label_hash[k])
		information_gain=dataset_entropy-tmp_entropy
		print "split_value_number-1", split_value_number-1
		if split_value_number-1 != 0:
		    information_gain-=math.log(split_value_number-1)/float(sample_number)
		#ʹ��log_2^(N-1)/|D|��������,NΪ���ѵ����
		#����������ԭ��ò������Ϊ���Ա��ⵥ��ʹ������ʱ������ѡ�������ͱ��� ??(����)
		return information_gain, subset_hash

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
#	dataset, labels=read_dataset('./545/train')
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

	dataset, labels =read_iris_dataset('iris')
	trainset_size=105
#	print "dataset", dataset
#	print "label", labels
	index=range(len(dataset))
#	print "index", index
	random.shuffle(index) #�����ݼ�����
#	print "index", index
	trainset=[ dataset[index[i]] for i in range(trainset_size) ]
	trainset_file=file('trainset', 'w')
	trainset_labels=[ labels[index[i]] for i in range(trainset_size) ]

	k=0
	for i in trainset:
		trainset_file.write("\t".join(i)+"\t"+str(trainset_labels[k])+"\n")
		k+=1

	testset=[ dataset[index[i]] for i in range(trainset_size, len(index)) ]
	testset_labels=[ labels[index[i]] for i in range(trainset_size, len(index)) ]
	k=0
	testset_file=file('testset', 'w')
	for i in testset:
		testset_file.write("\t".join(i)+"\t"+str(testset_labels[k])+"\n")
		k+=1

#	print "trainset", trainset
#	print "trainset_label", trainset_labels
#	print "testset", testset
#	print "testset_label", testset_labels
#	print "trainset", len(trainset)
#	print "trainset_label", len(trainset_labels)
#	print "testset", len(testset)
#	print "testset_label", len(testset_labels)
	test_tree=DecisionTree(trainset, trainset_labels)
	#print test_tree.entropy(labels)
	print "start to build tree"
	test_tree.build_tree()
	print "print tree"
	test_tree.print_tree()
	print "predict"
	print "testset", testset
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
	print "precision"
	print correct, len(pred_label), len(testset_labels), float(correct)/len(pred_label)
