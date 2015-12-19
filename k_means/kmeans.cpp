#include <iostream>
#include <ctime>
#include <vector>
#include <algorithm>
#include <float.h>
#include <math.h>
#include <map>
#include <string>
#include <sstream>
#include <fstream> //ifstream
#include <stdlib.h>
using namespace std;
void split(const string &s, char delim, vector<float> &elems) {
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(atof(item.c_str())); //读取时转化为浮点数
    }
}
//数据集从http://cs.joensuu.fi/sipu/datasets/页面中选择的aggregation数据集
//参数记录聚类中心和每个样本所属的中心
void kmeans(vector<vector<float> > dataset, int sample_number, int feature_number, int cluster_number, float** clusters, vector<int>& cluster_of_samples );

int main(int argc, char** argv){
    //读取数据集
    vector< vector<float> > dataset;
    string filename("aggregation");
    ifstream fin(filename.c_str());//ifstream构造函数不接受string类型，但是可以接受c-string
    string line;
    const char delimeter='\t';
    vector<float> line_vec;
    while( getline(fin, line) ){ //每次读取一行
	line_vec.clear();
	split(line, delimeter, line_vec);
//	for( int i=0; i < line_vec.size(); ++i )
//	    cout <<  line_vec[i] << " ";
//	cout << endl;
	vector<float> vec_without_cluster_index(line_vec.begin(), line_vec.end()-1);
	dataset.push_back(vec_without_cluster_index); //数据文件的最后一列是类的下标
    }
    for( int i=0; i< dataset.size(); ++i ){
	for( int j =0; j < dataset[i].size(); ++j ){
	    cout << dataset[i][j] << " ";
	}
	cout << endl;
    }

    int sample_number=dataset.size();
    int feature_number=dataset[0].size();
    int cluster_number=7;

    //为聚类中心分配空间
    float**clusters= new float*[cluster_number];
    int i;
    for( i=0; i<cluster_number; ++i ){
	clusters[i]=new float[feature_number];
    }
    vector<int> cluster_of_samples;
    kmeans( dataset, sample_number, feature_number, cluster_number, clusters, cluster_of_samples);
    for( i=0; i< cluster_of_samples.size(); ++i ){
	cout << i << " " << cluster_of_samples[i] << endl;
    }
    return 0;
}

void kmeans( vector<vector<float> >  dataset, int sample_number, int feature_number, int cluster_number, float** clusters, vector<int>& cluster_of_samples){
    if( sample_number <=0 || feature_number <=0 || cluster_number <=0 || clusters == NULL )
	return;
    if( sample_number < cluster_number )
	return;

    srand((unsigned)time(0));
    int cluster_index;
//用来保存初始化聚类中心的样本下标
    vector<int> cluster_index_vec;
    //初始化聚类中心
    for( int i =0; i < cluster_number; ++i ){
	cluster_index=rand()%sample_number; //随机选择一个样本作为聚类中心
	while( find( cluster_index_vec.begin(), cluster_index_vec.end(), cluster_index ) != cluster_index_vec.end() ){
	    cluster_index=rand()%sample_number; //随机选择一个样本作为聚类中心
	}
	cluster_index_vec.push_back(cluster_index);
	cout << i << "cluster_index" << cluster_index << " " << endl;
	for( int k =0; k< feature_number; ++k){
	    clusters[i][k]=dataset[cluster_index][k];
	    cout << clusters[i][k] << " ";
	}
	cout << endl;
    }

    int iteration=0;
    int max_iter=500;
    //三个终止条件
    //1.达到迭代次数上限
    //2.类内元素不再变化或者低于最低比例
    //3.每个样本到各自的聚类中心的欧氏距离之和(RSS)低于某个阈值
    float rss=0.0;
    while( true ){
	//每个样本计算和每个聚类中心的距离，从而找到最近的那个聚类中心
	cluster_of_samples.clear();
	float new_rss=0.0;
	for( int i=0; i< sample_number; ++i ){//每个样本
	    float minimum_distance=FLT_MAX;
	    int corr_cluster=-1;
	    for( int j=0; j< cluster_number; ++j ){
		float distance=0.0;
		for( int k = 0; k < feature_number; ++k ){
		    distance+=pow(dataset[i][k]-clusters[j][k], 2);
		}
		if( distance < minimum_distance ){
		    minimum_distance=sqrt(distance);
		    corr_cluster=j;
		    new_rss+=minimum_distance;
		}
	    }
	    cluster_of_samples.push_back(corr_cluster);
	}
	cout << "iteration" << iteration << " " << new_rss << endl;
	//rss变化比例低于0.001就退出
	if( fabs(new_rss - rss ) / rss < 0.00000001 ){//fabs是针对浮点型数据,abs针对整数
	    break;
	}else{
	    rss=new_rss;
	}
	//记录每个聚类中心下的样本
	map<int, vector<int> > samples_in_each_cluster;
	for( int i=0; i < cluster_of_samples.size(); ++i ){//每个样本所属的聚类中心
	    if( samples_in_each_cluster.find(cluster_of_samples[i]) != samples_in_each_cluster.end() ){
		samples_in_each_cluster[cluster_of_samples[i]].push_back(i);
	    }else{
		vector<int> sample_index;
		sample_index.push_back(i);
		samples_in_each_cluster[cluster_of_samples[i]]=sample_index;
	    }
	}
	//计算每个聚类中心下的新中心
	for( map<int, vector<int> >::iterator iter=samples_in_each_cluster.begin(); iter != samples_in_each_cluster.end(); ++iter){
	    int cluster_index=iter->first;
	    vector<int> sample_index=iter->second; //该类下的所有样本

	    for( int j =0; j< feature_number; ++j){
		clusters[cluster_index][j]=0.0;
	    }
	    for( int i =0; i < sample_index.size(); ++i ){
		for( int j=0; j < feature_number; ++j ){
		    clusters[cluster_index][j]+=dataset[sample_index[i]][j];
		   // cout << clusters[cluster_index][j] <<e;
		}
	    }
	    for( int i=0; i < feature_number; ++i ){
//		cout <<"feature_number" << feature_number << endl;
		cout << clusters[cluster_index][i]<< " ";

		clusters[cluster_index][i]=float(clusters[cluster_index][i])/float(sample_index.size());
	    }
	    cout << endl;
	}
	for( int i=0; i< cluster_number; ++i ){
	    for( int j =0; j < feature_number; ++j ){
		cout << clusters[i][j] << " " ;
	    }
	    cout << endl;
	}

	iteration++;

	if( iteration > max_iter )
	    break;
    }
    return;
}
