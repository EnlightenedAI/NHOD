import math
import time
import pyod
from scipy.spatial import KDTree
import numpy as np
import scipy.io as scio
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score

# Configuration parameters
k_values = np.array([61])
dataset_indices = np.array([1])
batch_size = 100

for idx in dataset_indices:
    dataset_name = str(idx)
    print(f'Dataset: {dataset_name}')
    
    # Load dataset
    data_file = f'../data/data{dataset_name}.mat'
    mat_data = scio.loadmat(data_file)
    features = mat_data['data'][:, 0:-1]
    labels = mat_data['data'][:, -1]
    
    num_samples = features.shape[0]
    num_features = features.shape[1]
    num_outliers = np.sum(labels == 1)
    
    print(f'Number of objects: {num_samples}')
    print(f'Number of dimensions: {num_features}')
    print(f'Number of outliers: {num_outliers}')
    
    # Feature scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)
    
    # Build KDTree for neighbor search
    tree = KDTree(scaled_data, leafsize=100000)
    transposed_data = scaled_data.T
    
    # Query nearest neighbors
    max_k = k_values[-1]
    distances, indices = tree.query(scaled_data, max_k + 1)
    
    # Open result files for appending
    with open('./auc/auc.txt', "a+") as f_auc, \
         open('./acc/ac.txt', "a+") as f_acc, \
         open('./time/time.txt', "a+") as f_time:
        
        print(f'data{dataset_name}, K: {k_values}, AUC', file=f_auc)
        print(f'data{dataset_name}, K: {k_values}, AC', file=f_acc)
        print(f'data{dataset_name}, K: {k_values}, time', file=f_time)
        
        for k in k_values:
            start_time = time.time()
            print(f'k: {k}')
            
            # Batch process for LCF calculation
            num_batches = math.ceil(num_samples / batch_size)
            all_lcf = None
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                
                mini_data = scaled_data[start_idx:end_idx, :]
                mini_indices = indices[start_idx:end_idx, :]
                mini_data_t = mini_data.T
                
                batch_lcf = np.zeros((num_features, mini_data.shape[0]))
                
                # Calculate LCF contribution for each neighbor
                for neighbor_idx in range(1, k + 1):
                    neighbor_indices = mini_indices[:, neighbor_idx]
                    neighbor_data = transposed_data[:, neighbor_indices]
                    
                    # Calculate Local Correlation Factor components
                    diff = neighbor_data - mini_data_t
                    moe = np.array(np.linalg.norm(diff, axis=0, keepdims=True), dtype=np.float128)
                    lcf_contribution = diff * (moe ** (num_features - 2))
                    
                    if neighbor_idx == 1:
                        batch_lcf = lcf_contribution
                    else:
                        batch_lcf += lcf_contribution
                
                if i == 0:
                    all_lcf = batch_lcf
                else:
                    all_lcf = np.hstack((all_lcf, batch_lcf))
            
            # Calculate final LCOD scores
            lcod_scores = np.linalg.norm(all_lcf, axis=0, keepdims=True)
            end_time = time.time()
            
            # Aggregate scores across the neighborhood (NHOD)
            nhod_scores = np.zeros((num_samples, 1))
            for neighbor_idx in range(0, k + 1):
                nhod_scores[:, 0] += lcod_scores[0, indices[:, neighbor_idx]]
            
            # Evaluate performance
            auc_score = roc_auc_score(labels, nhod_scores)
            pred_labels = pyod.utils.utility.get_label_n(labels, nhod_scores, num_outliers)
            accuracy = accuracy_score(labels, pred_labels)
            
            runtime = end_time - start_time
            print(f'Accuracy: {accuracy}')
            print(f'Time: {runtime}')
            
            # Write results to files
            print(runtime, file=f_time)
            print(auc_score, file=f_auc)
            print(accuracy, file=f_acc)
