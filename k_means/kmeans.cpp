#include <iostream>
#include <ctime>
#include <vector>
#include <algorithm>
#include <float.h>
#include <math.h>
using namespace std;

void kmeans(int** dataset, int sample_number, int feature_number, int cluster_number);
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

    vector<int> cluster_of_samples;
    while( true ){
	//每个样本计算和每个聚类中心的距离，从而找到最近的那个聚类中心
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
		}
	    }
	    cluster_of_samples.push_back(corr_cluster);
	}
	for( vector<int>::iterator iter=


    }
}
