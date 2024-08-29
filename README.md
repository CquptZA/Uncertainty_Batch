# Uncertainty_Batch
Batch Selection for Multi-Label Classification: Guided by Dynamic Uncertainty and Label Correlations

# Package:
python==3.7
skmultilearn==1.2.1
pytorch==1.11.0 
numpy== 1.21.5

# Usage

## run UncertainBatch.ipynb cell by cell
   
In detail, we can train on different datasets for the last cell: 

give all dataset used in paper: path_to_arff_files = ["emotions","scene","yeast", "Corel5k","rcv1subset1","rcv1subset2","rcv1subset3","yahoo-Business1","yahoo-Arts1","bibtex",'tmc2007','enron','cal500','LLOG-F']

give the label count of corresponding dataset: label_counts = [6, 6,14,374,101,101,101,28,25,159,22,53,174,75]

give the feature retention ratio of the corresponding: dataset=select_feature=[1,1,1,1,0.02,0.02,0.02,0.05,0.05,1,0.01,1,1,1]

The warm epoch is set to 10 by default.

For $$\lambda_{1}$$ and $$\lambda_{2}$$ default:

$$
E[idx, j] = \frac{1}{2} \times meandiffs + \frac{1}{2} \times currententropy, \quad where \quad \lambda_{1} = \lambda_{2} = \frac{1}{2}
$$


## key code:


\textbf{def update_H(H, y_pred, ids, max_history_length=5):}

    y_pred_numpy = y_pred.detach().cpu().numpy() 
    
    for i, idx in enumerate(ids):
    
        if idx not in H:
        
            H[idx] = deque(maxlen=max_history_length) 
            
        H[idx].append(y_pred_numpy[i])   
        
    return H

max_history_length is the size of the sliding window, H is a queue used to store the latest sliding window size predictions


def update_E(H, E, ids, label_dim):

    for idx in ids:
    
        current_predictions_history = np.array(H[idx])
        
        last_row_index = len(current_predictions_history) - 1
        
        for j in range(label_dim): 
        
            diffs = np.abs(np.diff(current_predictions_history[:, j]))
            
            mean_diffs = np.sum(diffs)/len(diffs)
            
            current_entropy = -1 / np.log(2) * (
                current_predictions_history[last_row_index][j] * np.log(current_predictions_history[last_row_index][j]) 
                + (1 - current_predictions_history[last_row_index][j]) * np.log(1 - current_predictions_history[last_row_index][j])
            )
            
            E[idx,j] = 1/2 * mean_diffs + 1/2 * current_entropy
            
    return E
    
E is a two-dimensional array of n (number of samples) * q (number of labels), which updates the uncertainty of each sample and label in current epoch.

def update_U(E, U, epoch):

    E[E > 1] = 1
    
    if epoch >= 5:
    
        bins = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        
        discrete_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        
        indices = np.digitize(E, bins) - 1
        
        indices = np.clip(indices, 0, len(discrete_values) - 1)
        
        E = discrete_values[indices]
        
    else:
    
        I = np.ones((E.shape[1], E.shape[1]))
        
    I = np.ones((E.shape[1], E.shape[1]))
    
    U = np.dot(E, I)
    
    w = np.sum(U, axis=1)
    
    return w





## analyse.ipynb includes all comparison methods,You can run the corresponding cell to obtain the results
