import sklearn
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import v_measure_score
from scipy import stats
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import homogeneity_score
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from fcmeans import FCM
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
from scipy.stats import entropy
from scipy import stats
from sklearn.metrics.cluster import adjusted_mutual_info_score
import matplotlib.cm as cm
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import IsolationForest

def get_data(data):

    train_x1 = np.genfromtxt(data, dtype=float, delimiter=',')
    #label = train_x1[:, -1]
    #train_x1 = np.delete(train_x1, 8, axis=1)
    train_x1 = train_x1.astype(float)
    #label = label.astype(int)
    return train_x1

def minmax(train_x):
    scaler = MinMaxScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    return train_x

def standarization(train_x):
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    return train_x

def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matr ix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def plot_clusters(train_x1,labels):
    u_labels = np.unique(labels)

    # plotting the results:

    for i in u_labels:
        plt.scatter(train_x1[labels == i, 0], train_x1[labels == i, 1], label=i)
    plt.legend()
    plt.show()
    return None

def kmeans(train_x1,train_y,num_class):
    print('k means')
    kmeans = KMeans(n_clusters=num_class).fit(train_x1)
    m = kmeans.labels_
    # ps = purity_score(train_y, m)
    ss = silhouette_score(train_x1, m)
    # ff=f1_score(train_y.flatten(), m,average='weighted' )
    # ri =rand_score(train_y.flatten(), m)
    ari = adjusted_rand_score(train_y.flatten(), m)
    # mi=mutual_info_score(train_y.flatten(), m)
    # print("purity",ps)
    print("silo", ss)
    # print("f1 score",ff)
    # print("ri", ri)
    print("ari", ari)
    # print("mi", mi)
    plot_clusters(train_x1, m)
    return ari

def fuuzyc(train_x1,train_y,num_class):
    print('fuzz')
    fcm = FCM(n_clusters=num_class)
    fcm.fit(train_x1)
    m = fcm.predict(train_x1)
    # ps = purity_score(train_y, m)
    ss = silhouette_score(train_x1, m)
    # ff=f1_score(train_y.flatten(), m,average='weighted' )
    # ri =rand_score(train_y.flatten(), m)
    ari = adjusted_rand_score(train_y.flatten(), m)
    # mi=mutual_info_score(train_y.flatten(), m)
    # print("purity",ps)
    print("silo", ss)
    # print("f1 score",ff)
    # print("ri", ri)
    print("ari", ari)
    # print("mi", mi)
    # plot_clusters(train_x1, m)
    return ari

def gmm(train_x1,train_y,num_class):
    print('gmm')
    gm = GaussianMixture(n_components=num_class).fit(train_x1)
    m = gm.predict(train_x1)
    # ps = purity_score(train_y, m)
    ss = silhouette_score(train_x1, m)
    # ff=f1_score(train_y.flatten(), m,average='weighted' )
    # ri =rand_score(train_y.flatten(), m)
    ari = adjusted_rand_score(train_y.flatten(), m)
    # mi=mutual_info_score(train_y.flatten(), m)
    # print("purity",ps)
    print("silo", ss)
    # print("f1 score",ff)
    # print("ri", ri)
    print("ari", ari)
    # print("mi", mi)
    #plot_clusters(train_x1, m)
    return ari

def spec(train_x1,train_y,num_class):
    print('spec')
    clustering = SpectralClustering(n_clusters=num_class).fit(train_x1)
    m = clustering.labels_
    # ps = purity_score(train_y, m)
    ss = silhouette_score(train_x1, m)
    # ff=f1_score(train_y.flatten(), m,average='weighted' )
    # ri =rand_score(train_y.flatten(), m)
    ari = adjusted_rand_score(train_y.flatten(), m)
    # mi=mutual_info_score(train_y.flatten(), m)
    # print("purity",ps)
    print("silo", ss)
    # print("f1 score",ff)
    # print("ri", ri)
    print("ari", ari)
    # print("mi", mi)
    # plot_clusters(train_x1, m)
    return ari


def hirr(train_x1,train_y,num_class):
    print('hirr')
    clustering = AgglomerativeClustering(n_clusters=num_class).fit(train_x1)
    m = clustering.labels_
    # ps = purity_score(train_y, m)
    ss = silhouette_score(train_x1, m)

    # ff=f1_score(train_y.flatten(), m,average='weighted' )
    # ri =rand_score(train_y.flatten(), m)
    ari = adjusted_rand_score(train_y.flatten(), m)
    # mi=mutual_info_score(train_y.flatten(), m)
    # print("purity",ps)
    print("silo", ss)
    # print("f1 score",ff)
    # print("ri", ri)
    print("ari", ari)
    # print("mi", mi)

    # plot_clusters(train_x1, m)
    return ari

def remove_outliars(set_points):
    clf= IsolationForest(random_state=0).fit_predict(set_points)
    counter=0
    for i in range(len(clf)):
        if clf[i-counter] < 0:
            set_points= np.delete(set_points, i-counter, axis=0)
            clf= np.delete(clf, i-counter, axis=0)
            counter+=1

    return set_points

def data_after_removel(train_x):


    np.random.shuffle(train_x)
    label = train_x[:,8]
    train_x= np.delete(train_x, 8, axis=1)

    return train_x,label


if __name__ == '__main__':

    r=1
    # 1 for remove outliers
    # 0 for not

    data1_final="final_data1.csv"
    train_x=get_data(data1_final)


    train1 = train_x[0:16259, :]
    test1 = remove_outliars(train1)
    train2 = train_x[16260:, :]
    test2 = remove_outliars(train2)

    const=2

    if r==1 :
        #with removal of outliars
        new_train=np.concatenate((test1,test2),axis=0)

    else:
        #without removal of outliars
        new_train=np.concatenate((train1,train2),axis=0)

    arir = np.zeros([5, 10])


    for i in range(10):

        train_x3, label1 = data_after_removel(new_train)

        train_x = train_x3[0:12000, :]
        label = label1[0:12000]


        train_x = standarization(train_x)
        # train_x = minmax(train_x)

        pca = PCA(n_components=2)
        pca.fit(train_x)
        print(pca.explained_variance_ratio_)
        train_x = pca.transform(train_x)



        r1=kmeans(train_x,label,const)
        arir[0,i]=r1
        r2=fuuzyc(train_x, label, const)
        arir[1, i] = r2
        r3 = gmm(train_x, label, const)
        arir[2, i] = r3
        r4=spec(train_x, label, const)
        arir[3, i] = r4
        r5=hirr(train_x, label, const)
        arir[4, i] = r5

    np.savetxt("final_data1_results.csv", arir, fmt='%f', delimiter=',')







