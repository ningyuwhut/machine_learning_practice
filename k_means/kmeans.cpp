#include <iostream>
#include <ctime>
#include <vector>
#include <algorithm>
#include <float.h>
#include <math.h>
#include <map>
using namespace std;

//应该返回聚类中心和每个样本所属的中心
int** kmeans(int** dataset, int sample_number, int feature_number, int cluster_number);
int main(int argc, char** argv){

}
int** kmeans(int** dataset, int sample_number, int feature_number, int cluster_number){
    if( dataset == NULL || sample_number <=0 || feature_number <=0 || cluster_number <=0 )
	return;
    if( sample_number < cluster_number )
	return;
    int **clusters= new int*[cluster_number];
    int i;
    for( i=0; i<cluster_number; ++i ){
	clusters[i]=new int[feature_number];
    }
    srand((unsigned)time(0));

    int cluster_index;
    vector<int> cluster_index_vec;
    //初始化聚类中心
    for( i =0; i < cluster_number; ++i ){
	cluster_index=rand()%sample_number; //随机选择一个样本作为聚类中心
	while( find( cluster_index_vec.begin(), cluster_index_vec.end(), cluster_index ) != cluster_index_vec.end() ){
	    cluster_index=rand()%sample_number; //随机选择一个样本作为聚类中心
	}
	cluster_index_vec.push_back(cluster_index);
	for( int k =0; k< feature_number; ++k){
	    clusters[i][k]=dataset[cluster_index][k];
	}
    }

    vector<int> cluster_of_samples;//记录每个样本所属的聚类中心
    int iter=0;
    int max_iter=500;
    //三个终止条件
    //1.达到迭代次数上限
    //2.类内元素不再变化或者低于最低比例
    //3.每个样本到各自的聚类中心的欧氏距离之和(RSS)低于某个阈值
    float rss=0.0;
    while( true ){
	//每个样本计算和每个聚类中心的距离，从而找到最近的那个聚类中心
	float new_rss=0.0;
	for( int i=0; i< sample_number; ++i ){
	    float minimum_distance=FLT_MAX;
	    int corr_cluster=-1;
	    for( int j=0; j< cluster_number; ++j ){
		float distance=0.0;
		for( int k = 0; k < feature_number; ++k ){
		    distance+=math.power(dataset[i][k]-clusters[j][k], 2);
		}
		if( distance < minimum_distance ){
		    minimum_distance=distance;
		    corr_cluster=j;
		    new_rss+=minimum_distance;
		}
	    }
	    cluster_of_samples.push_back(corr_cluster);
	}
	//rss变化比例低于0.001就退出
	if( abs(new_rss - rss ) / rss < 0.00 1 ){
	    break;
	}
	//记录每个聚类中心下的样本
	map<int, vector<int> > samples_in_each_cluster;
	for( int i=0; i < cluster_of_samples.size(); ++i ){
	    if( samples_in_each_cluster.find(cluster_of_samples[i]) != samples_in_each_cluster.end() ){
		samples_in_each_cluster[cluster_of_samples[i] ].push_back(i);
	    }else{
		vector<int> sample_index;
		sample_index.push_back(i);
		samples_in_each_cluster[cluster_of_samples[i]]=sample_index;
	    }
	}
	//计算每个聚类中心下的新中心

	for( map<int, vector<int> > iter=samples_in_each_cluster.begin(); iter != samples_in_each_cluster.end(); ++iter){
	    int cluster_index=iter->first;
	    vector<int> sample_index=iter->second; //该类下的所有样本

	    for(int i=0; i< cluster_number; ++i ){
		for( int j =0; j< feature_number; ++j){
		    clusters[i][j]=0.0;
		}
	    }
	    for( int i =0; i < sample_index.size(); ++i ){
		for( int j=0; j < feature_number; ++j ){
		    clusters[cluster_index][j]+=dataset[sample_index[i]][j]
		}
	    }
	    for( int i=0; i <=feature_number; ++j ){
		clusters[cluster_index]/=sample_index.size();
	    }
	}

	iter++;

	if( iter > max_iter )
	    break;
    }
    return clusters;
}
