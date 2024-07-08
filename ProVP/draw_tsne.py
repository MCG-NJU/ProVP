
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
DataPath='/storage/data1/xuchen/ProVP_ori/ProVP/output/base2new/test_base/ucf101/shots_16/ProVP/vit_b16_ep50/'
# 加载Iris数据集 鸢尾花
def draw_tsne(DataPath,mode,dataset):
    data_x=[]
    data_y=[]
    for i in range(3):
        X = np.load(DataPath+'seed'+str(i+1)+'/img_feats.npy',allow_pickle=True)
        y = np.load(DataPath+'seed'+str(i+1)+'/labels.npy',allow_pickle=True)
        X = np.vstack((X[0], X[1]))
        y = np.concatenate((y[0],y[1]))
        data_x.append(X)
        data_y.append(y)

    data_x=np.vstack((data_x[0],data_x[1],data_x[2]))
    data_y=np.concatenate((data_y[0],data_y[1],data_y[2]))
    data_y=1/np.max(data_y)*data_y
    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, perplexity=200,random_state=10)
    X_2d = tsne.fit_transform(data_x)
    print(X_2d.shape,y.shape)
    fig = plt.figure()
    plt.scatter(X_2d[:,0], X_2d[:,1], c=data_y,cmap='cividis',s=5)
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.show()
    plt.savefig(dataset+mode+'.png',bbox_inches='tight', pad_inches=0)
# 可视化结果
dataset="ucf101"
DataBasePath='/storage/data1/xuchen/ProVP_ori/ProVP/output/base2new/test_base/'+dataset+'/shots_16/ProVP/vit_b16_ep50/'
DataNewPath='/storage/data1/xuchen/ProVP_ori/ProVP/output/base2new/test_new/'+dataset+'/shots_16/ProVP/vit_b16_ep50/'
draw_tsne(DataBasePath,"base",dataset)
draw_tsne(DataNewPath,"new",dataset)
