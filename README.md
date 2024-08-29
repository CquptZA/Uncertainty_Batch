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

$$
E[\text{idx}, j] = \frac{1}{2} \times \text{mean\_diffs} + \frac{1}{2} \times \text{current\_entropy} \quad where \lambda_1=\lambda_2=1/2
$$



## analyse.ipynb includes all comparison methods,You can run the corresponding cell to obtain the results
