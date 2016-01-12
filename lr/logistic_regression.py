#encoding=gbk
import random
import math

#读取数据集
def load_data(filename):
	dataset = file(filename)
	x_input=[]
	y_output=[]
	for line in dataset:
		a_x_input=[]
		splitted_line = line.strip().split(',')
		length=len(splitted_line)
		a_x_input=splitted_line[0:length-1]
		a_x_input.append(1)
		x_input.append(a_x_input)
		y_output.append(splitted_line[length-1])

	return (x_input, y_output)

def sigmoid(x):
	return 1.0/(1.0+math.pow(math.e, -x))

def compute_log_likelihood(trainset_x, trainset_y,weights):
	sample_num=len(trainset_x)
	feature_num=len(trainset_x[0])
	log_likelihood=0.0
	another_log_likelihood=0.0
	for i in range(0,sample_num):
		wx=0
		for j in range(0, feature_num):
			wx+=weights[j]*float(trainset_x[i][j])
		if float(trainset_y[i]) == 1.0:
			print float(trainset_y[i])
		log_likelihood+=(float(trainset_y[i])*wx-math.log(1+math.pow(math.e, wx)))
		another_log_likelihood+=( (float(trainset_y[i])*math.log(sigmoid(wx)))+(1-float(trainset_y[i])*math.log(1-sigmoid(wx))) )
		#两个似然其实是一样的，只是another计算的是原始形式，log_likelihood多推导了几步

	print "log_likelihood", log_likelihood
	print "another_log_likelihood", another_log_likelihood
		
	return log_likelihood

#梯度下降
#这里实现的是批梯度下降
def train(trainset_x, trainset_y, max_iter):
	sample_num=len(trainset_x)
	feature_num=len(trainset_x[0])
	weights=[1]*feature_num
	alpha=0.001 #步长，或叫learning rate
	print "sample_num, feature_num", sample_num, feature_num

	for m in range(max_iter): #0到max_iter-1
		print "m", m
		log_likelihood = compute_log_likelihood(trainset_x, trainset_y, weights)

		for k in range(0, feature_num): #0到feature_num-1
			gradient=0
			for i in range(0,sample_num): #0到sample_num-1 #每个样本
				tmp=0	
				for j in range(0, feature_num):
					tmp+=float(trainset_x[i][j])*weights[j] #计算w*x
				output=sigmoid(tmp) #1/(1+exp(-w*x))
				error=float(trainset_y[i])-output

				gradient+=error*float(trainset_x[i][k])

			weights[k]+=alpha*gradient

		new_log_likelihood = compute_log_likelihood(trainset_x, trainset_y, weights)

		if new_log_likelihood < log_likelihood :
			print "new", new_log_likelihood
			print "old", log_likelihood
			print "error"
			print (new_log_likelihood-log_likelihood)/log_likelihood
			break
		if (new_log_likelihood-log_likelihood)/log_likelihood < 0.001 :
			print (new_log_likelihood-log_likelihood)/log_likelihood
			break

	return weights

if __name__ == "__main__":
	dataset_file="ds1.10.csv"
	x_input, y_output = load_data(dataset_file)
	print len(x_input)

	trainset_x=[]
	trainset_y=[]

	testset_x=[]
	testset_y=[]
	max_iter=500

	dataset_size=len(x_input)
	#随机抽取80%做训练集
	trainset_size=int(dataset_size*0.8)
	testset_size=dataset_size-trainset_size

	index=range(len(x_input))
	random.shuffle(index)
	trainset_x = [ x_input[index[i]] for i in range(trainset_size)]
	trainset_y = [ y_input[index[i]] for i in range(trainset_size)]
	testset_x = [ x_input[index[i]] for i in range(trainset_size,dataset_size)]
	testset_y = [ y_input[index[i]] for i in range(trainset_size,dataset_size)]

	print len(trainset_x),len(trainset_y)
	print len(testset_x), len(testset_y)

	weights = train(trainset_x,trainset_y,max_iter)

