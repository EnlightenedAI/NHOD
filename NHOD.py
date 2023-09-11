import math
import time
import pyod
from scipy.spatial import KDTree
import numpy as np
np.set_printoptions(threshold=np.inf)
import scipy.io as scio    
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
#a=np.array(range(1,21))*10
K=np.array([61])  
name=np.array([1])
# K= np.arange(10,201,10)
for na in range(name.shape[0]):
    parts = ['../data/data','.mat']
    namen=str(name[na])
    print('Dataset:'+namen)
    dataFile = parts[0]+namen+parts[1]
    data = scio.loadmat(dataFile)
    data1=data['data'][:,0:-1]
    lable=data['data'][:,-1]
    w=data1.shape[1]
    print('number of objects：', data1.shape[0])
    print('number of dimensions：',w)
    T=np.sum(lable==1)          #number of outliers
    print('number of outliers:',T)
    min_max_scaler = MinMaxScaler()
    Tdata = min_max_scaler.fit_transform(data1)
    tree = KDTree(Tdata,leafsize=100000)
    trandata = np.array([[row[i] for row in Tdata] for i in range(len(Tdata[0]))])
    dist, ind = tree.query(Tdata, K[K.shape[0] - 1] + 1)
    fpauc=open(r'./auc/auc.txt',"a+")
    fpacc=open(r'./acc/ac.txt',"a+")
    fptime=open(r'./time/time.txt',"a+")
    #fpout=open(r'./out/out.txt',"a+")
    print('data' + namen, 'K:', K, 'AUC',file=fpauc)
    print('data' + namen, 'K:', K,'AC', file=fpacc)
    print('data' + namen, 'K:', K, 'time', file=fptime)
    #print('data' + namen, 'K:', K, 'out', file=fpout)

    for k in K:
        start = time.time()
        print('k:', k)
        N = 100
        n = 0
        print(math.ceil(Tdata.shape[0] / N))
        for r in range(math.ceil(Tdata.shape[0] / N)):
            if r < math.ceil(Tdata.shape[0] / N)-1:
                minidata = Tdata[n:n + N, :]
                miniind = ind[n:n + N, :]
            else:
                minidata = Tdata[n:Tdata.shape[0], :]
                miniind = ind[n:Tdata.shape[0], :]
            #print(minidata.shape)
            n = n + N
            minitrandata = np.array([[row[i] for row in minidata] for i in range(len(minidata[0]))])
            # trandata= np.array([[row[i] for row in Tdata] for i in range(len(Tdata[0])) ])#type:[2,1043]
            lcfk = np.zeros((w, len(minidata))) 
            lcfkk = [] 
            # data_1=data1/mdata
            for kk in range(1, k + 1):
                k1 = miniind[:, kk] 
                k1data = trandata[:, k1]
                moe = np.array(np.linalg.norm(minitrandata - k1data, axis=0, keepdims=True), dtype=np.float128)
                lcf = (k1data - minitrandata) * (moe ** (w-2))
                if kk==1:
                    lcfk=lcf 
                else:
                    lcfk = lcfk+lcf  
            if r == 0:
                LCF = lcfk
            else:
                LCF = np.hstack((LCF, lcfk))
        lcod = np.linalg.norm(LCF, axis=0, keepdims=True)
        end = time.time()
        rLcf = lcod.T
        for kk in range(1,k+1):
            if kk==1:
                LDCD=lcod[0,ind[:,kk]]
            else:
                LDCD=LDCD+lcod[0,ind[:,kk]]
        LDCD=np.array(LDCD).reshape(len(data1),1)
        LDCD=rLcf/LDCD
        AUC=roc_auc_score(lable,LDCD)
        y_pred = pyod.utils.utility.get_label_n(lable,LDCD, T)
        AC=accuracy_score(lable, y_pred)
        print('Accuracy:',AC)
        runtime=end-start
        print('time:',runtime)
        print(runtime, file=fptime)
        print(AUC, file=fpauc)
        print(AC, file=fpacc)
    fpauc.close()
    fpacc.close()
    fptime.close()
