#encoding=utf8
import sys
import math
import random
import collections
from numpy import inf

#可以在代码中加入预剪枝
#比如划分后两个分支的样本数都不能小于一定数值
#划分后的均方误差的减少程度不能小于某一数值
#以后要加上

class DecisionTreeNode(object):
	def __init__(self,split_feature=-1, feature_value=-1):
		self.split_feature=split_feature
		self.feature_value=feature_value
		self.childs=None
		self.target=None
		self.feature_type="discrete" 
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
	def get_target(self):
		return self.target
	def set_target(self, target):
		self.target=target
	#为连续型特征而设置的属性
	def get_feature_type(self):
	    return self.feature_type
	def set_feature_type(self, feature_type):
	    self.feature_type=feature_type
	def set_greater_or_less(self, greater_or_less):
	    self.greater_or_less=greater_or_less

	def get_greater_or_less(self):
	    return self.greater_or_less

	def get_gain(self):
		return self.gain
	def set_gain(self, gain):
		self.gain=gain

class DecisionTree(object):
	def __init__(self, dataset , target, least_sample_in_leaf=5):
		self.dataset=dataset
		self.target=target
		self.root=None
		self.least_sample_in_leaf=least_sample_in_leaf
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

	#cart树算法每次都要从所有特征中选择最优的，所以feature_list参数不需要了
	def build_tree(self, class_number):
		self.root=self.build_tree_sub(self.dataset, self.target, class_number )	

	def build_tree_sub(self, dataset, target, class_number ):
		#设置一些停止条件
		if len(dataset) <= 5: #样本少于5个时停止构建树
			leaf=DecisionTreeNode()
			#对于一般的cart回归树来说，叶子节点的预测值是属于这个叶子节点的所有样本的target的均值
			leaf.set_target(sum(map(float,target))/len(target))
#			class_number=len(set(self.target))
			target_sum=sum(target)
			denominator=0.0
			for i in range(len(target)):
			    denominator+=abs(target[i])*(1-abs(target[i]))
			#当用于gbdt时，叶子节点的预测值应该按下面设置，为了方便我在节点中加了一个gain属性用来记录，应该有更好的实现方法
			print "class_number", class_number, target_sum, denominator
			print "set_gain", (class_number-1)/float(class_number)
			print "sdf" , target_sum/float(denominator) 

			leaf.set_gain((class_number-1)/float(class_number)*target_sum/float(denominator) )
			return leaf
		else:
			split_feature, split_feature_value, subset_hash=self.choose_feature(dataset,target )
			#根据这个特征把数据集分成几个部分
			print "split_feature", split_feature
			print "split_feature_value", split_feature_value
			child_nodes=[]
			#根据离散特征分支划分数据集时，子数据集中不再包含该特征（因为每个分支下的子数据集该特征的取值就会是一样的，信息增益或者Gini Gain将不再变化）；而根据连续特征分支时，各分支下的子数据集必须依旧包含该特征（当然，左右分支各包含的分别是取值小于、大于等于分裂值的子数据集），因为该连续特征再接下来的树分支过程中可能依旧起着决定性作用。

			#当是连续特征时，需要确保小于分裂值的样本走到左分支，大于的走到右分支，这是一个约定
			#现在的实现是连续型特征的第一个孩子节点是左分支,第二个是右分支
			print "subset_hash_1", subset_hash, type(subset_hash)
			for i in subset_hash:
			#	print "i_1", i	
				subset=[ dataset[j] for j in subset_hash[i] ]
				corr_target=[ target[j] for j in subset_hash[i]]
				child=self.build_tree_sub(subset, corr_target, class_number )

				if self.feature_type[split_feature]=="discrete":
				    child.set_feature_value(split_feature_value[i])
				else:
				    child.set_feature_value(split_feature_value)
				#subset_hash中是属于该节点的所有样本


				child.set_feature_type(self.feature_type[split_feature])
				if self.feature_type[split_feature] == "continuous":
				    if i == 1:
					child.set_greater_or_less("less")
				    elif i == 2:
					child.set_greater_or_less("greater")

				child_nodes.append(child)
			parent_node=DecisionTreeNode(split_feature)
			parent_node.set_childs(child_nodes)
#			if self.feature_type[split_feature] == "continuous":
			parent_node.set_feature_type(self.feature_type[split_feature])
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
		print "sample", sample
		while True:
			if node.get_target() != None:
				print "label", node.get_target()
				return node.get_target()
			else:
				split_feature=node.get_split_feature() #当前节点用来分裂的特征
				split_feature_value=sample[split_feature] #该样本该分裂特征的值
				print "node", split_feature, split_feature_value
				if self.feature_type[split_feature]=="discrete":
				    for child in node.get_childs():
					    print "child", child.get_feature_value(), split_feature_value #, child.get_greater_or_less()
					    if split_feature_value in child.get_feature_value():
						    node=child
				else:#连续型特征
				    childs=node.get_childs()
				    print "childs", childs
				    for i in childs:
					print i, i.get_feature_value(),i.get_greater_or_less()
				#这里一定要转换为float，因为没有使用任何的库，直接从文件中读取，所以所有的数据类型默认都是字符串，所以
				#在对样本进行分类时需要对特征值比较大小，此时必须把特征值转换为浮点型
				    if float(split_feature_value) < float(childs[0].get_feature_value()):#当前样本的该特征的值小于分裂点的值，往左走
					print split_feature_value, "less than childs[0].get_feature_value()", childs[0].get_feature_value()
					node=childs[0]
				    else:
					print split_feature_value, "more than childs[0].get_feature_value()", childs[1].get_feature_value()
					node=childs[1]

	def get_gain(self, testset):
	    target=[]
	    for sample in testset:
		target.append(self.get_gain_a_sample(sample))
	    return target
	def get_gain_a_sample(self, sample):
		node=self.root
		print "sample", sample
		while True:
			if node.get_target() != None:
				print "label1", node.get_gain()
				return node.get_gain()
			else:
				split_feature=node.get_split_feature() #当前节点用来分裂的特征
				split_feature_value=sample[split_feature] #该样本该分裂特征的值
				print "node", split_feature, split_feature_value
				if self.feature_type[split_feature]=="discrete":
				    for child in node.get_childs():
					    print "child", child.get_feature_value(), split_feature_value #, child.get_greater_or_less()
					    if split_feature_value in child.get_feature_value():
						    node=child
				else:#连续型特征
				    childs=node.get_childs()
				    print "childs", childs
				    for i in childs:
					print i, i.get_feature_value(),i.get_greater_or_less()
				#这里一定要转换为float，因为没有使用任何的库，直接从文件中读取，所以所有的数据类型默认都是字符串，所以
				#在对样本进行分类时需要对特征值比较大小，此时必须把特征值转换为浮点型
				    if float(split_feature_value) < float(childs[0].get_feature_value()):#当前样本的该特征的值小于分裂点的值，往左走
					print split_feature_value, "less than childs[0].get_feature_value()", childs[0].get_feature_value()
					node=childs[0]
				    else:
					print split_feature_value, "more than childs[0].get_feature_value()", childs[1].get_feature_value()
					node=childs[1]
	
	def print_tree(self):
		self.print_tree_sub(self.root, 1)
	#打印的时候该怎么打印
	def print_tree_sub(self, node, depth):
		if node.target!= None:#叶子节点
			if node.get_feature_type()=="discrete":
			    print "depth" +"\t"+str(depth) + "\t"+ "label"+"\t"+str(node.get_target()) +"\t" +"feature_value" +"\t"+ node.get_feature_value()
			else:
			    if node.get_greater_or_less() == "greater":
				print "depth" +"\t"+str(depth) + "\t"+ "label"+"\t"+str(node.get_target()) +"\t" +"feature_value" +"\t"+ " >= " +"\t" + str(node.get_feature_value())
			    else:
				print "depth" +"\t"+str(depth) + "\t"+ "label"+"\t"+str(node.get_target()) +"\t" +"feature_value" +"\t"+ " < " +"\t" + str(node.get_feature_value())
				
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
	
#	def majority(target):
#	#首先统计每个label的出现次数
#	#然后求次数最多的label作为叶子节点的类标
#		count_hash={}
#		for i in target:
#			if i in count_hash:
#				count_hash[i]+=1
#			else:
#				count_hash[i]=1
#		maximum_count=0
#		majority_label=None
#		for i in count_hash:
#			if count_hash[i] > maximum_count:
#				maximum_count=count_hash[i]
#				majority_label=i
#		
#		return majority_label

##根据某个特征的取值将数据集分成几个部分，每个部分该特征的取值相同
##tmp_hash记录该特征的每个取值对应的所有样本的下标
#	def split_dataset_by_concrete_feature(self, dataset, feature_index):
#		sample_number=len(dataset)
#		feature_value_set=set()
#		for i in range(sample_number):
#		    feature_value_set.add(dataset[i][feature_index])
#		split_set_overall=all_split(feature_value_set)#所有的二元划分
#
#		tmp_hash={}
#		    feature_list=[ dataset[j][feature_index] for j in range( sample_number) ] #每个样本的该特征的取值
#		    for j in range(sample_number):
#			    if feature_list[j] in tmp_hash:
#				    tmp_hash[feature_list[j]].append(j)
#			    else:
#				    tmp_hash[feature_list[j]]=[j]
#		return tmp_hash



	    
	
	
#feature_index:用来分割的特征下标
#计算离散特征的最优划分
#首先要知道特征取值集合的所有划分
#该问题等价于求一个集合划分为两部分的所有划分
	def best_split_for_discrete_feature(self, dataset, target, feature_index):
		sample_number=len(target)
		feature_value_set=set()
		for i in range(sample_number):
		    feature_value_set.add(dataset[i][feature_index])

		split_set_overall_has_repetition=all_split(feature_value_set)#所有的二元划分

		split_set_overall=[]
		for i in split_set_overall_has_repetition:
		    if not split_set_overall: 
			split_set_overall.append(i)
		    else:
			if_in=False
			for j in split_set_overall:
			    if i[0] in j or i[1] in j:
				if_in=True
				break
			if if_in == False:
			    split_set_overall.append(i)
		print "split_set_overall"
		print split_set_overall, len(split_set_overall)

		min_mse=inf
		best_split=[]
		best_sample_hash=[]
		for split in split_set_overall: #对每个划分
		    target_hash={}
		    sample_hash={}
		    #划分后的样本的目标值
		    for i in range(sample_number):
			if dataset[i][feature_index] in split[0]: #sample_hash使用0，1做 key是为了能和split的下标相对应
			    if 0 not in target_hash:
				target_hash[0]=[]
				sample_hash[0]=[]

			    target_hash[0].append(target[i])
			    sample_hash[0].append(i)
			elif dataset[i][feature_index] in split[1]:
			    if 1 not in target_hash:
				target_hash[1]=[]
				sample_hash[1]=[]
			    target_hash[1].append(target[i])
			    sample_hash[1].append(i)
		    cur_mse=0.0
		    for i in target_hash:
			mean=sum(map(float, target_hash[i]))/len(target_hash[i])
			for j in target_hash[i]:
				    cur_mse+=(float(j)-mean)**2
		    if cur_mse < min_mse:
			min_mse=cur_mse
			best_split=split
			best_sample_hash=sample_hash

		return  best_split, best_sample_hash, min_mse 

	#统计每个特征的最小均方误差，均方误差最小的特征作为分裂特征
	#计算信息增益率时，离散特征和连续特征的处理方法是不同的
	#连续特征需要选择一个最优分裂点，所有分裂点中增益最大的分裂点即最优分裂点，该点处的信息增益率作为该特征的信息增益率
	#dataset 为数据集
	#target为对应的目标变量
	#feature_list为可以用来分割的特征，若值为1表示该特征可以用于分裂
	#
	#当最优特征为离散特征时，best_split_value是一个长度为2的数组，数组的两个元素是划分后的特征值的集合
	def choose_feature(self, dataset, target ):
		print "in choose feature"
		target_feature_index=-1
		min_mse=inf
		best_split_value=0.0
		best_subset_hash={}
		feature_number=len(dataset[0])
		for i in range(feature_number): 
			#计算增益时都需要根据特征把数据集分成几部分，而这个划分在生成节点的孩子节点时会用到，为避免重复划分，所以将划分作为结果返回
			if self.feature_type[i]== "discrete":
				split_value, subset_hash, min_mse_local=self.best_split_for_discrete_feature(dataset, target, i)
			elif self.feature_type[i] == "continuous":
				split_value, subset_hash, min_mse_local=self.best_split_for_continuous_feature(dataset, target, i) #该连续特征的最佳分裂点和对应的mse

			print "feature", i, "mse_local",min_mse_local 

			if min_mse_local< min_mse:
				min_mse=min_mse_local
				target_feature_index=i
				best_subset_hash=subset_hash
				best_split_value=split_value
		print "best_feature", target_feature_index
		return target_feature_index, best_split_value, best_subset_hash

	#为某个连续型特征选择最优分裂点
	def best_split_for_continuous_feature( self, dataset, target, feature_index ):
		#每个样本下标和对应的该特征的取值和对应的类标
		print "in best_split_for_continuous_feature"
		print "feature_index", feature_index
		feature_value_label={}
		feature_value_set=set()
		for i in range(len(dataset)):
			feature_value=float(dataset[i][feature_index])
			feature_value_set.add(feature_value)
			feature_label=target[i]
			feature_value_label[i]=[i, feature_value, feature_label]

		sorted_feature_value=list(feature_value_set)
		sorted_feature_value.sort()
		print "sorted_feature_value", sorted_feature_value

		feature_value_label_sorted = sorted(feature_value_label.items(), key=lambda x : x[1][1] ) #按特征取值由小到大排列, 返回一个列表, 列表的每个元素是一个元组
		print "feature_value_label_sorted", feature_value_label_sorted
		#从所有的分裂点中选平方损失最小的分裂点做最优分裂点
		min_mean_square_error=inf
		best_split_value=0.0
		best_subset_hash={}
		split_value_number=len(sorted_feature_value)-1 #分裂点个数
		for i in range(split_value_number):
			split_value=(sorted_feature_value[i]+sorted_feature_value[i+1])/2
			mean_square_error, subset_hash=self.mse( feature_value_label_sorted, split_value )
			print "mean_square_error", mean_square_error, "split_value", split_value, "min_mean_square_error", min_mean_square_error
	
			if mean_square_error < min_mean_square_error:
				print "here"
				min_mean_square_error=mean_square_error
				best_split_value=split_value
				best_subset_hash=subset_hash

		print "best_split_value", best_split_value
		
		return best_split_value, best_subset_hash, min_mean_square_error

	#计算某个特征的某个分裂点的平方损失
	#该分裂点把数据集分成两部分，基于分裂点的是一部分，大于分裂点的是另外一部分，每一部分的预测值为目标变量的均值，计算这种分裂下的平方损失
	def mse(self, feature_value_label_sorted,split_value):
		subset_hash=collections.OrderedDict() #{}
		#分裂点在feature_value_label_sorted中的下标
		split_sample_index=0
		for i in range( len(feature_value_label_sorted) ):
		    if feature_value_label_sorted[i][1][1] >= split_value:
			split_sample_index=i
			break
		print "split_value", split_value
		print "split_sample_index", split_sample_index

		subset_hash[1]=[x[1][0] for x in feature_value_label_sorted[0:split_sample_index] ] #特征取值小于split_value的样本下标集合
		subset_hash[2]=[x[1][0] for x in feature_value_label_sorted[split_sample_index:] ] #特征取值小于split_value的样本下标集合

		subset_label_hash={}
		subset_label_hash[1]=[x[1][2] for x in feature_value_label_sorted[0:split_sample_index] ] #特征取值小于split_value的样本目标值集合
		subset_label_hash[2]=[x[1][2] for x in feature_value_label_sorted[split_sample_index:] ] #特征取值小于split_value的样本目标值集合
	    
		mse=0.0
		for k in subset_label_hash:
#		    print "here", map(float, subset_label_hash[k])
		    mean=sum(map(float, subset_label_hash[k]))/len(subset_label_hash[k])
		    for target in subset_label_hash[k]:
			mse+=(mean-float(target))**2
		return mse, subset_hash

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

def all_split(feature_value_set):
	split_set_overall=[]
	for feature_value in feature_value_set:
	    cur_split_set=set()
	    cur_split_set.add(feature_value)
	    all_split_sub(feature_value_set-cur_split_set, cur_split_set, split_set_overall)
	return split_set_overall
	    

def all_split_sub(remained_feature_value, cur_split_set, split_set_overall ):
	if not remained_feature_value:
	    return
	new_split=[]
	new_split.append(cur_split_set)
	new_split.append(remained_feature_value)
	split_set_overall.append(new_split)
	for feature_value in remained_feature_value:
	    split_set=set(cur_split_set)
	    split_set.add(feature_value)
	#    print remained_feature_value
	#    print set().add(feature_value)
	    tmp_set=set()
	    tmp_set.add(feature_value)
	    all_split_sub(remained_feature_value-tmp_set,split_set, split_set_overall )

if __name__ == '__main__':

#	feature_set=set()
#	feature_set.add(1)
#	feature_set.add(2)
#	feature_set.add(3)
#	feature_set.add(4)
#	feature_set.add(5)
#	print "feature_set", feature_set
#	two_splits=all_split(feature_set)
#	print two_splits
#	
#	no_repetition=[]
#	for i in two_splits:
#	    if not no_repetition: 
#		no_repetition.append(i)
#	    else:
#		if_in=False
#		for j in no_repetition:
#		    if i[0] in j or i[1] in j:
#			if_in=True
#			break
#		if if_in == False:
#		    no_repetition.append(i)
#	print "no_repetition"
#	print no_repetition, len(no_repetition)
	    
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
	target=[0,0,1,1,0,0,0,1,1,1,1,1,1,1,0]
#	dataset, target=read_dataset('./545/train')
#	print dataset
#	print len(dataset), len(dataset[0])
#	print target
#	testset, testset_target=read_dataset('./545/test')
#	print testset
#	print len(testset), len(testset[0])
#	print testset_target
#
#	dataset, target=read_weather_dataset('weather_dataset')
#	testset=[dataset[-1]]

	dataset, target =read_iris_dataset('car')
	#trainset_size=105 #iris
	trainset_size=125 #wine
#	print "dataset", dataset
#	print "label", target
	index=range(len(dataset))
#	print "index", index
	random.shuffle(index) #把数据集打乱
#	print "index", index
	trainset=[ dataset[index[i]] for i in range(trainset_size) ]
	trainset_file=file('trainset', 'w')
	trainset_target=[ target[index[i]] for i in range(trainset_size) ]

	k=0
	for i in trainset:
		trainset_file.write("\t".join(i)+"\t"+str(trainset_target[k])+"\n")
		k+=1

	testset=[ dataset[index[i]] for i in range(trainset_size, len(index)) ]
	testset_target=[ target[index[i]] for i in range(trainset_size, len(index)) ]
	k=0
	testset_file=file('testset', 'w')
	for i in testset:
		testset_file.write("\t".join(i)+"\t"+str(testset_target[k])+"\n")
		k+=1

	test_tree=DecisionTree(trainset, trainset_target)
	#print test_tree.entropy(target)
	print "start to build tree"
	test_tree.build_tree()
	print "print tree"
	test_tree.print_tree()
	print "predict"
	print "testset", testset
#	sys.exit(0)
   # print test_tree.classify_a_sample(test_sample)
	pred_label=test_tree.classify(testset)
	#print "pred_label", str(pred_label)+"\t"+str(testset_target)
	print len(pred_label)
	for i in range(len(pred_label)):
		print str(pred_label[i])+"\t"+str(testset_target[i])

	correct=0
	for i in range(len(pred_label)):
		if pred_label[i] == testset_target[i]:
			correct+=1
	print "precision"
	print correct, len(pred_label), len(testset_target), float(correct)/len(pred_label)
