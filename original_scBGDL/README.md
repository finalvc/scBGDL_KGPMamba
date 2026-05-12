# Graph-based Deep Learning for Integrating Single-Cell and Bulk Transcriptomic Data to Identify Clinical Cancer Subtypes
scBGDL integrates single-cell and bulk transcriptomic data to construct sample-specific gene graphs and utilizes graph-based deep learning with self-attention mechanisms to enhance survival prediction and cancer subtype classification, thereby improving the precision of cancer subtype evaluation.The workflow of the proposed scBGDL is illustrated in Figure：
<div align="center">
  <img src="https://github.com/user-attachments/assets/3dd3dda8-899c-4cf6-b0e5-a048733fbbab" alt="workflow" width="500">
</div>
<HR>

### Datasets  
+ Survival_data: Contains survival information for each cohort, stored in CSV format. The file must include at least three columns: `id`, `survival_time`, and `survival_state`.  
+ RNA-seq Expression: Contains RNA-seq expression data for each cohort.  
+ Single-Cell Expression: Contains single-cell expression data.  
+ Key Genes: Stores key genes identified for each clinical sample.  
+ Graphs: Contains graph representations of clinical samples in each cohort.  
<HR>

### Usage  
Follow the steps below to utilize scBGDL effectively:  

+ Configure the Environment  
   Set up the required software and dependencies for scBGDL.  
+ Prepare Datasets  
   Create a folder in the `datasets` directory for your data.  
   Download the required datasets and place them in the created folder.  
+ Identify Key Genes  
   Run `get_Sample-Specific_KeyGenes.py` to identify key genes specific to each clinical sample.  
+ Generate Graph Representations  
   Use `conceptualize_Sample-Specific_graph.py` to create graph representations for each clinical sample.  
+ Train the Predictive Model  
   Execute `train_predictive_model.py` to train the graph-based deep learning (GDL) model.  
+ Calculate Risk Scores  
   Once the model training is complete, use `model_evaluation.py` to calculate the risk score for each clinical sample based on the trained GDL model.  

