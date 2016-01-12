#encoding=gbk
from numpy import *
#读取数据集
def load_data(filename):
	dataset = file(filename)
	x_input=[]
	y_output=[]
	for line in dataset:
		a_x_input=[]
		splitted_line = line.strip().split(',')
#		print splitted_line
		length=len(splitted_line)
		a_x_input=splitted_line[0:length-1]
		a_x_input.append(1)
		x_input.append(a_x_input)
		y_output.append(splitted_line[length-1])
#		print a_x_input
#		print splitted_line[length-1]

	return mat(x_input), mat(y_output).transpose()

def sigmoid(x):
	return 1.0/(1+exp(-x))

def compute_log_likelihood(trainset_x, trainset_y,weights):
	sample_num, feature_num = shape(trainset_x)
	log_likelihood=0.0
	another_log_likelihood=0.0

	for i in range(0,sample_num):
		wx=0
		for j in range(0, feature_num):
			wx+=weights[j]*float(trainset_x[i][j])
		log_likelihood+=(float(trainset_y[i])*wx-math.log(1+math.pow(math.e, wx)))
		another_log_likelihood+=( (float(trainset_y[i])*math.log(sigmoid(wx)))+(1-float(trainset_y[i])*math.log(1-sigmoid(wx))) )

	print "log_likelihood", log_likelihood
	print "another_log_likelihood", another_log_likelihood
		
	return log_likelihood

def train(trainset_x, trainset_y, max_iter):
	sample_num, feature_num = shape(trainset_x)
	print sample_num, feature_num
	sample_num=len(trainset_x)
	feature_num=len(trainset_x[0])
	print sample_num, feature_num
	weights=ones((feature_num, 1))
#	weights=[1]*feature_num
	for k in range(max_iter):
		log_likelihood = compute_log_likelihood(trainset_x, trainset_y, weights)
		output=sigmoid(trainset_x*weights)
		error= trainset_y-output
		weights = weights+alpha*trainset_x.transpose()*error
		new_log_likelihood = compute_log_likelihood(trainset_x, trainset_y, weights)

		if new_log_likelihood < log_likelihood :
			print "new", new_log_likelihood
			print "old", log_likelihood
			print "error"
			print (new_log_likelihood-log_likelihood)/log_likelihood
			break
	#	if (new_log_likelihood-log_likelihood)/log_likelihood < 0.001 :
	#		print (new_log_likelihood-log_likelihood)/log_likelihood
	#		break
	return weights

if __name__ == "__main__":
	dataset_file="ds1.10.csv"
	x_input, y_output = load_data(dataset_file)
	max_iter=500

	train(x_input, y_output, max_iter)
