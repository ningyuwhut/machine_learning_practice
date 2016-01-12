#encoding=utf8
from sklearn.svm import SVC
import random

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

    clf=SVC(C=0.6,kernel="linear", tol=0.001)
    clf.fit(trainset, trainset_target)
    print clf.score(testset, testset_target)
#    svm=SVM(dataset, target)
#    svm.train()
