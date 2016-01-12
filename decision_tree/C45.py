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
		self.feature_type="discrete" #默认是离散型特征
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
	#为连续型特征而设置的属性
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
		self.feature_value={} #记录每个特征都有哪些取值
		for i in range(len(self.dataset)):#每个样本
			for j in range(len(self.dataset[i])):#每个特征
				if j not in self.feature_value:
					self.feature_value[j]=set()
				self.feature_value[j].add(self.dataset[i][j])

		self.feature_type={} #记录特征是离散型还是连续型特征
		for i in self.feature_value:#特征下标
			continuous=False
			for value in self.feature_value[i]:#特征的所有取值
				if type(value) == float:#特征取值为浮点型
					self.feature_type[i]='continuous'
					continuous=True
					break
			if continuous== False : 
				if len(self.feature_value[i]) >= 10: #特征取值数目超过10个则视为连续型特征
					self.feature_type[i]='continuous'
				else:
					self.feature_type[i]='discrete'

	def build_tree(self):
		feature_list=[ 1 for i in range(len(self.dataset[0])) ] 
		self.root=self.build_tree_sub(self.dataset, self.labels, feature_list)	

	def build_tree_sub(self, dataset, labels, feature_list):
		#这里可以改成别的数，少于某个数则不再进行分裂,可以看做是一种预剪枝的方法
		if len(dataset) == 1: #只有一个样本，此时不需要再选择特征，直接将该样本的target作为该叶子节点的目标值
			leaf=DecisionTreeNode()
			leaf.set_label(labels[0])
			return leaf
		elif len(set(labels))==1: #剩余样本都属于同一类
			leaf=DecisionTreeNode()
			leaf.set_label(labels[0])
			return leaf
		elif feature_list.count(1) == 0:#没有特征可以使用
			leaf=DecisionTreeNode()
			leaf.set_label( self.majority( labels ) )
			return leaf
		else:
			split_feature, split_feature_value, subset_hash=self.choose_feature(dataset,labels, feature_list)
			#根据这个特征把数据集分成几个部分
			print "split_feature", split_feature
			print "split_feature_value", split_feature_value
			child_nodes=[]
			parent_feature_list=feature_list[:]
			#根据离散特征分支划分数据集时，子数据集中不再包含该特征（因为每个分支下的子数据集该特征的取值就会是一样的，信息增益或者Gini Gain将不再变化）；而根据连续特征分支时，各分支下的子数据集必须依旧包含该特征（当然，左右分支各包含的分别是取值小于、大于等于分裂值的子数据集），因为该连续特征再接下来的树分支过程中可能依旧起着决定性作用。
			if self.feature_type[split_feature] =="discrete": #如果特征是离散的，那后面该特征不再用来分裂,如果是连续特征则后面仍然
			#可以用来分割
			    parent_feature_list[split_feature]=-1 
			#当是连续特征时，需要确保小于分裂值的样本走到左分支，大于的走到右分支，这是一个约定
			#现在的实现是连续型特征的第一个孩子节点是左分支,第二个是右分支
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
			#对于离散型特征，需要设置孩子节点的特征值，
			#对于连续型特征，需要设置当前分类节点的特征值
			#其实，对连续型特征也可以设置孩子节点的值，但是要约定好哪边是大于，哪边是小于，这样在分类时可以知道应该往哪边走
			#现在,为了统一起见，舍弃设置当前节点的特征值这一方法,改为全都设置孩子节点的值
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
				split_feature=node.get_split_feature() #当前节点用来分裂的特征
				split_feature_value=sample[split_feature] #该样本该分裂特征的值
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
				else:#连续型特征
				    childs=node.get_childs()
				    print "childs", childs
				    for i in childs:
					print i, i.get_feature_value(),i.get_greater_or_less()
				#这里一定要转换为float，因为没有使用任何的库，直接从文件中读取，所以所有的数据类型默认都是字符串，所以
				#在对样本进行分类时需要对特征值比较大小，此时必须把特征值转换为浮点型
				    if float(split_feature_value) < float(childs[0].get_feature_value()):#当前样本的该特征的值小于分裂点的值，往左走
					print "type", type(split_feature_value)
					print split_feature_value, "less than childs[0].get_feature_value()", childs[0].get_feature_value()
					node=childs[0]
				    else:
					print "type", type(split_feature_value)
					print split_feature_value, "more than childs[0].get_feature_value()", childs[1].get_feature_value()
					node=childs[1]
	
	def print_tree(self):
		self.print_tree_sub(self.root, 1)
	#打印的时候该怎么打印
	def print_tree_sub(self, node, depth):
		if node.label != None:#叶子节点
			if node.get_feature_type()=="discrete":
			    print "depth" +"\t"+str(depth) + "\t"+ "label"+"\t"+str(node.get_label()) +"\t" +"feature_value" +"\t"+ node.get_feature_value()
			else:
			    if node.get_greater_or_less() == "greater":
				print "depth" +"\t"+str(depth) + "\t"+ "label"+"\t"+str(node.get_label()) +"\t" +"feature_value" +"\t"+ " >= " +"\t" + str(node.get_feature_value())
			    else:
				print "depth" +"\t"+str(depth) + "\t"+ "label"+"\t"+str(node.get_label()) +"\t" +"feature_value" +"\t"+ " < " +"\t" + str(node.get_feature_value())
				
		else: #非叶子节点
			if node.get_feature_type()=="discrete":
			    print "depth" +"\t"+str(depth) + "\t"+"split_feature"+"\t"+ str(node.get_split_feature())+"\t"+"feature_value", str(node.get_feature_value())
			else:#连续特征
			    if node.get_greater_or_less() == "greater":
				print "depth" +"\t"+str(depth) + "\t"+ "split_feature"+"\t"+ str(node.get_split_feature())+"\t" + "feature_value" +"\t"+ " >= " +"\t" + str(node.get_feature_value())
			    elif node.get_greater_or_less() == "less":
				print "depth" +"\t"+str(depth) + "\t"+ "split_feature"+"\t"+ str(node.get_split_feature())+"\t" + "feature_value" +"\t"+ " < " +"\t" + str(node.get_feature_value())
			    else:
				print "depth" +"\t"+str(depth) + "\t"+ "split_feature"+"\t"+ str(node.get_split_feature())+"\t" + "feature_value" +"\t"+ " = " +"\t" + str(node.get_feature_value())
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
#tmp_hash记录该特征的每个取值对应的所有样本的下标
	def split_dataset_by_feature(self, dataset, feature_index, feature_type, feature_value):
		sample_number=len(dataset)
		tmp_hash={}
		if feature_type == "discrete":
		    feature_list=[ dataset[j][feature_index] for j in range( sample_number) ] #每个样本的该特征的取值
		    for j in range(sample_number):
			    if feature_list[j] in tmp_hash:
				    tmp_hash[feature_list[j]].append(j)
			    else:
				    tmp_hash[feature_list[j]]=[j]
		return tmp_hash

#feature_index:用来分割的特征下标
#计算离散特征的信息增益率
	def information_gain_ratio_for_discrete_feature(self, dataset, labels, feature_list, feature_index):
		sample_number=len(self.labels)
		dataset_entropy=self.entropy(labels)
		tmp_hash=self.split_dataset_by_feature(dataset, feature_index, "discrete", 0.0) #最后面两个参数对离散型特征是没有用的

		#首先，计算该特征的所有取值，以及每个取值对应的下标集合
		tmp_entropy=0.0
		for k in tmp_hash:
			tmp_label=[ labels[i] for i in tmp_hash[k]]
			tmp_entropy+=len(tmp_hash[k])/float(sample_number)*self.entropy(tmp_label)

		information_gain_ratio=(dataset_entropy-tmp_entropy)/float(dataset_entropy)

		return information_gain_ratio,tmp_hash

		
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

	#统计每个特征的信息增益率，信息增益最大的特征作为分裂特征
	#计算信息增益率时，离散特征和连续特征的处理方法是不同的
	#连续特征需要选择一个最优分裂点，所有分裂点中增益最大的分裂点即最优分裂点，该点处的信息增益率作为该特征的信息增益率
	#dataset 为数据集
	#labels为对应的目标变量
	#feature_list为可以用来分割的特征，若值为1表示该特征可以用于分裂
	#
	def choose_feature(self, dataset, labels, feature_list):
		print "in choose feature"
		target_feature_index=-1
		max_information_gain_ratio=0.0
		best_split_value=0.0
		best_subset_hash={}
		for i in range(len(feature_list)): 
			if feature_list[i] == 1: #每个可用的特征
				#计算增益时都需要根据特征把数据集分成几部分，而这个划分在生成节点的孩子节点时会用到，为避免重复划分，所以将划分作为结果返回
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


	#为连续型特征选择分裂点
	def infomation_gain_ratio_for_continuous_feature( self, dataset, labels, feature_list, feature_index ):
		#每个样本下标和对应的该特征的取值和对应的类标
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

		feature_value_label_sorted = sorted(feature_value_label.items(), key=lambda x : x[1][1] ) #按特征取值由小到大排列, 返回一个列表, 列表的每个元素是一个元组
		print "feature_value_label_sorted", feature_value_label_sorted
		#从所有的分裂点中选信息增益最大的分裂点做最优分裂点,此时的信息增益作为该连续特征的信息增益，并据此计算信息增益率
		max_info_gain=0.0
		best_split_value=0.0
		best_subset_hash={}
		split_value_number=len(sorted_feature_value)-1 #分裂点个数
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

	#计算某个特征的某个分裂点的信息增益
	#该分裂点把数据集分成两部分，基于分裂点的是一部分，大于分裂点的是另外一部分，计算这种分裂下的信息增益
	#有一个问题是如果存在相同的分裂点又该怎么处理
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

		subset_hash[1]=[x[1][0] for x in feature_value_label_sorted[0:split_feature_index] ] #特征取值小于split_value的样本下标集合
		subset_hash[2]=[x[1][0] for x in feature_value_label_sorted[split_feature_index:] ] #特征取值小于split_value的样本下标集合

		subset_label_hash={}
		subset_label_hash[1]=[x[1][2] for x in feature_value_label_sorted[0:split_feature_index] ] #特征取值小于split_value的样本类标集合
		subset_label_hash[2]=[x[1][2] for x in feature_value_label_sorted[split_feature_index:] ] #特征取值小于split_value的样本类标集合
		sample_number=len(self.labels)
		dataset_entropy=self.entropy(labels)

		tmp_entropy=0.0
		for k in subset_hash:
			tmp_entropy+=len(subset_hash[k])/float(sample_number)*self.entropy(subset_label_hash[k])
		information_gain=dataset_entropy-tmp_entropy
		print "split_value_number-1", split_value_number-1
		if split_value_number-1 != 0:
		    information_gain-=math.log(split_value_number-1)/float(sample_number)
		#使用log_2^(N-1)/|D|进行修正,N为分裂点个数
		#这样修正的原因貌似是因为可以避免单纯使用增益时更容易选择连续型变量 ??(存疑)
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
	random.shuffle(index) #把数据集打乱
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
