import pandas as pd
import numpy as np
import sklearn.cluster as clusterKmeans
import seaborn as sns
import matplotlib.pyplot as plt
import pyfpgrowth
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.preprocessing import StandardScaler
from clustviz.chameleon.chameleon import cluster
from sklearn import metrics
import time


def elbowMethod(df):
    keep = []
    it = range(1,15)
    for cl in it:
        km = clusterKmeans.KMeans(n_clusters=cl)
        km.fit(df)
        keep.append(km.inertia_)
    plt.figure(figsize=(15,7))
    plt.plot(it, keep, 'bx-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Within-Cluster Sum of Square')
    plt.title('Elbow Method to find optimum number of clusters')
    plt.show()
   

# I tried to implement it but I couldn't understand how can I apply the clustering operation
# using FP-Growth    
"""
def frequent_pattern_growth(df):
    patterns = pyfpgrowth.find_frequent_patterns(df, 2)
    print(patterns)
     
"""

def DB_scan(df):
      
    if(df.shape[1] < 20):
        df = StandardScaler().fit_transform(df)
        dbscan_info = DBSCAN(eps=0.3, min_samples=30).fit_predict(df)
        plt.title("After DBSCAN Clustering for DS1")
        plt.scatter(df[:,0], df[:,1], c=dbscan_info)
        sc_ds1 = metrics.silhouette_score(df, dbscan_info, metric='euclidean')
        print('Silhouette Coefficient for DS1 with DBSCAN: %.3f\n' % sc_ds1)
        
        
    elif(df.shape[1] > 20):
        df = StandardScaler().fit_transform(df)
        dbscan_info = DBSCAN(eps=6, min_samples=100).fit_predict(df)
        print("\nClusters of DS2 with DBSCAN")
        unique, counts = np.unique(dbscan_info, return_counts=True)
        print(dict(zip(unique, counts)))
        sc_ds2 = metrics.silhouette_score(df,dbscan_info, metric='euclidean')
        print('\nSilhouette Coefficient for DS2 with DBSCAN: %.3f\n' % sc_ds2)
        
        
        
def Kmeans(df):
    
    if(df.shape[1] < 20):   # for two-dimensional dataset
        elbowMethod(df)
        kmeans = clusterKmeans.KMeans(n_clusters = 3, init ="k-means++")
        kmeans = kmeans.fit(df)
        df['Clusters'] = kmeans.labels_
        sns.scatterplot(x = 'x',y ='y',hue = 'Clusters',data = df)
        ax = plt.gca()
        ax.set_title("After Kmeans Clustering for DS1")
        sc_ds1 = metrics.silhouette_score(df, kmeans.labels_, metric='euclidean')
        print('Silhouette Coefficient for DS1 with Kmeans: %.3f\n' % sc_ds1)
        
        
    elif(df.shape[1] > 20):  # for dataset that has more than 20 dimensions 
        elbowMethod(df)
        kmeans = clusterKmeans.KMeans(n_clusters = 2, init ="k-means++")
        kmeans = kmeans.fit(df)
        df['Clusters'] = kmeans.labels_
        print("Clusters of DS2 with Kmeans")
        print(df['Clusters'].value_counts())
        print()
        sc_ds2 = metrics.silhouette_score(df, kmeans.labels_, metric='euclidean')
        print('Silhouette Coefficient for DS2 with Kmeans: %.3f\n' % sc_ds2)
        

def Chameleon(df):
    
    if(df.shape[1] < 20):
        chameleonModel,h = cluster(pd.DataFrame(df),k=2,knn=15,m=10,alpha=2,plot=False)
        chameleonModel.plot(kind='scatter', c=df['cluster'], cmap='jet' , x=0, y=1,title = 'After Chameleon Clustering with DS1')
        print()
        plt.show()
        sc_ds1 = metrics.silhouette_score(df, df['cluster'], metric='euclidean')
        print('Silhouette Coefficient for DS1 with CHAMELEON: %.3f\n' % sc_ds1)
        
    elif(df.shape[1] > 20):
        chameleonModel,h = cluster(pd.DataFrame(df),k=2,knn=15,m=10,alpha=2,plot=False)
        chameleonModel.plot(kind='scatter', c=df['cluster'], cmap='jet' , x=0, y=1,title = 'After Chameleon Clustering with DS2')
        plt.show()
        sc_ds2 = metrics.silhouette_score(df, df['cluster'], metric='euclidean')
        print('Silhouette Coefficient for DS2 with CHAMELEON: %.3f\n' % sc_ds2)
        

 

if __name__ == "__main__":
    dataFrameDS1 = pd.read_csv("DS1.csv")  
    dataFrameDS2 = pd.read_csv("DS2.csv")  
    
    # KMEANS
    start_kmeans_ds1 = time.time()
    #kmeans with ds1
    Kmeans(dataFrameDS1)
    end_kmeans_ds1 = time.time()
    print("Computational time for DS1 with Kmeans: ",end_kmeans_ds1 - start_kmeans_ds1)
    print()
    
    start_kmeans_ds2 = time.time()
    #kmeans with ds2
    Kmeans(dataFrameDS2)
    end_kmeans_ds2 = time.time()
    print("Computational time for DS2 with Kmeans: ",end_kmeans_ds2 - start_kmeans_ds2)
    print()
    
    #DBSCAN
    start_dbscan_ds1 = time.time()
    #dbscan with ds1
    DB_scan(dataFrameDS1)
    end_dbscan_ds1 = time.time()
    print("Computational time for DS1 with DBSCAN: ",end_dbscan_ds1 - start_dbscan_ds1)
    print()
    
    start_dbscan_ds2 = time.time()
    #dbscan with ds2
    DB_scan(dataFrameDS2)
    end_dbscan_ds2 = time.time()
    print("Computational time for DS2 with DBSCAN: ",end_dbscan_ds2 - start_dbscan_ds2)
    print()


    #CHAMELEON
    start_chameleon_ds1 = time.time()
    #chameleon with ds1
    Chameleon(dataFrameDS1)
    end_chameleon_ds1 = time.time() 
    print("Computational time for DS1 with CHAMELEON: ",end_chameleon_ds1 - start_chameleon_ds1)
    print()
    
    start_chameleon_ds2 = time.time()
    #chameleon with ds2
    Chameleon(dataFrameDS2)
    end_chameleon_ds2 = time.time() 
    print("Computational time for DS2 with CHAMELEON: ",end_chameleon_ds2 - start_chameleon_ds2)
    print()


