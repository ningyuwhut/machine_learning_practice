#encoding=utf8
import math
import random

class SVM(object):
    def __init__(self, dataset, target, C=0.6, tolerance=0.001):
	self.dataset=dataset
	self.target=target
	self.C=C
	self.tolerance=tolerance
	self.alpha=[0.0 for i in range(len(dataset))]
	self.E={}
	self.b=0.0
	self.w=[0.0 for i in range(len(dataset[0]))]
	
    def train(self):
	numChanged=0
	exampleAll=1

	trainset_size=len(self.dataset)
	iter=0
	while numChanged > 0 or exampleAll:
	    numChanged=0
	    if exampleAll: #检查整个数据集
		for i in range(trainset_size):
		    numChanged+=self.examineExample(i)
		iter+=1
	    else:#选择alpha在(0,c)区间的样本进行检查
		for i in range(trainset_size): 
		    numChanged+=self.examineExample(i)
		iter+=1
	    if exampleAll:
		exampleAll=0
	    elif numChanged == 0:#所有alpha在(0,c)中的样本都没有优化,那么下次就对整个数据集进行选择
		exampleAll=1
	    print "iter", iter
	for j in range(len(self.dataset[0])):
	    for i in range(trainset_size):
		self.w[j] +=self.alpha[i]*self.target[i]*self.dataset[i][j]
		
    #y*E=y(f_x-y)=y*f_x-1
    #所以，如果y*E>0，那么y*f_x>1,即函数间隔大于1，样本点在间隔边界以外，对应的alpha为0,此时如果alpha>0，那么该样本点违反了kkt条件。
    #如果y*E<0,那么y*f_x<1,即函数间隔小于1，样本点在间隔边界以内，对应的alpha为C,此时如果alpha小于C,那么该样本点违反了kkt条件
    #在实现时还要考虑舍入误差(round-off error )
    def examineExample(self, i2):
	alpha2=self.alpha[i2]
	y2=self.target[i2]
	e2=self.calculateE(i2)
	r2=e2*y2

	#选择违反kkt条件的样本
	if (r2 < -self.tolerance and self.alpha[i2] < self.C) or (r2 > self.tolerance and self.alpha[i2] > 0): #i2违反了kkt条件
	    i1=self.select_i1(i2,e2)
	    if self.takeStep(i1, i2):
		return 1
	    else: 
	    #按照上面的启发性方法选择的i1无法令目标函数有足够的下降,那么
	    #先在non-bound examples中寻找可以带来足够下降的样本
	    #如果non-bound examples不能满足要求，那么在所有样本中选择可以带来足够下降的样本
	    #选择时从一个随机的位置开始，避免每次都是选择前面的样本
	
		all_sample_index=[i for i in range(len(self.dataset)) ]
		random.shuffle(all_sample_index)

		for k in range(len(all_sample_index)):
		    i1=all_sample_index[k]
		    if self.alpha[i1] >0 and self.alpha[i1] < self.C:
			if self.takeStep(i1, i2):
			    return 1
		for k in range(len(all_sample_index)):
		    i1=all_sample_index[k]
		    if self.takeStep(i1,i2):
			return 1
	return 0
    
    def takeStep(self, i1, i2):
	if i1==i2:
	    return 0
	alpha1=self.alpha[i1]
	y1=self.target[i1]
	e1=self.calculateE(i1)

	alpha2=self.alpha[i2]
	y2=self.target[i2]
	e2=self.calculateE(i2)

	s=y1*y2

	if y1 != y2:
	    L=max(0, alpha2-alpha1)
	    H=min(self.C, self.C+alpha2-alpha1)
	if y1== y2:
	    L=max(0, alpha2+alpha1-self.C)
	    H=min(self.C, alpha2+alpha1)
	if L==H:
	    return 0
	k11=self.kernel(i1, i1)
	k12=self.kernel(i1, i2)
	k22=self.kernel(i2, i2)
	eta=k11+k22-2*k12

	if eta > 0:
	    self.alpha[i2]=alpha2+y2*(e1-e2)/eta
	    if self.alpha[i2] < L:
		self.alpha[i2]=L
	    if self.alpha[i2] >H:
		self.alpha[i2]=H

	    if abs(self.alpha[i2] - alpha2) < 0.00001: #变化幅度太小
		return 0

	    self.alpha[i1]=alpha1+s*(alpha2-self.alpha[i2])

	    b1=self.b-e1-y1*(self.alpha[i1]-alpha1)*self.kernel(i1,i1)-y2*(self.alpha[i2]-alpha2)*self.kernel(i1,i2)
	    b2=self.b-e2-y1*(self.alpha[i1]-alpha1)*self.kernel(i1,i2)-y2*(self.alpha[i2]-alpha2)*self.kernel(i2,i2)

	    if self.alpha[i1] >0 and self.alpha[i1] < self.C:
		self.b=b1
	    elif self.alpha[i2] > 0 and self.alpha[i2] < self.C:
		self.b=b2
	    else:
		self.b=(b1+b2)/2

	    self.E[i2]=self.calculateE(i2) #更新E
	    self.E[i1]=self.calculateE(i1)

	    return 1  #更新成功
	else:
	#当核函数不符合mercer's条件，目标函数为非正定的，该值会小于0
	#当数据集中存在相同的样本且核函数正确时，该值会为0
	#这种情况暂时不考虑
	    return 0

    #从(0,C)中选择最大步长
    def select_i1(self, i, Ei ):
	maxK=-1;
	maxDeltaE=0.0
	Ej=0

	self.E[i]=Ei #更新E

	for k in range(len(self.alpha)):
	    if self.alpha[k] >0 and self.alpha[k] < self.C:
		Ek=self.calculateE(k)
		deltaE=Ek-Ei
		if abs(deltaE) > maxDeltaE:
		    maxK=k
		    maxDeltaE=deltaE
		    Ej=Ek

	if maxK != -1:
	    return maxK
	else: #不存在
	    j=i
	    while j == i:
		j=random.randint(0, len(self.dataset))
	    return j

    def calculateE(self, i):
	f_x=0.0
	trainset_size=len(self.dataset)
	for k in range(trainset_size):
	    f_x+=(self.alpha[k]*self.target[k]*self.kernel(k,i))
	f_x+=self.b
	e_x=f_x-self.target[i]
	return e_x
	    
	
    def kernel(self, i, j): #样本i和j的核函数，暂时只实现线性核
	return sum([self.dataset[i][k]*self.dataset[j][k] for k in range(len(self.dataset[i]))])
	

    def test(self, testset, testset_target):
	precision=0.0
	correct=0
	for k in range(len(testset)):
	    sample =testset[k]
	    pred_value=0.0
	    for i in range(len(sample)):
		pred_value+=self.w[i]*sample[i]
	    pred_value+=self.b

	    if pred_value >= 0:
		label=1
	    else:
		label=-1
	    if testset_target[k] == label:
		correct+=1

	precision=correct/(float(len(testset_target)))

	return precision

def read_libsvm_format_file(dataset_filename):
	dataset_file=file(dataset_filename,'r')
	dataset_label=[]
	dataset=[]
	for line in dataset_file:
		splitted=line.strip().split()
		dataset_label.append(float(splitted[0]))
		sample=[]
		for i in range(1,len(splitted)):
			index_value=splitted[i].split(":")
			sample.append(float(index_value[1]))
		dataset.append(sample)
	return dataset, dataset_label

if __name__ == "__main__":
    dataset, target =read_libsvm_format_file('diabetes')

    trainset_size=500 #wine
    index=range(len(dataset))
    random.shuffle(index) #把数据集打乱
    trainset=[ dataset[index[i]] for i in range(trainset_size) ]
    trainset_target=[ target[index[i]] for i in range(trainset_size) ]

    testset=[ dataset[index[i]] for i in range(trainset_size, len(index)) ]
    testset_target=[ target[index[i]] for i in range(trainset_size, len(index)) ]

    svm=SVM(dataset, target)
    svm.train()
