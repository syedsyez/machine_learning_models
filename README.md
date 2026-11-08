# Machine Learning Assignment#2

## 1. Problem Statement

The objective is to build binary classification models using
measurements computed from digitized images of fine needle aspirates
(FNA) of breast masses.

-   Benign (B): mapped to `0`
-   Malignant (M):mapped to `1`

The models are compared using Accuracy, AUC, Precision, Recall,
F1-score, and Matthews Correlation Coefficient (MCC).

## 2. Dataset

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

The dataset contains 569 observations and 30 numeric input features,
along with an identifier and the diagnosis target.

### Data preparation

1.  Loaded the dataset from `data.csv`.
2.  Removed the `id` column because it is an identifier, not a
    predictive feature.
3.  Removed the `Unnamed: 32` column because it does not contain useful
    feature information.
4.  Checked for missing values and duplicate records.
5.  Converted diagnosis labels: `B` to `0` and `M` to `1`.
6.  Separated input features from the target variable.
7.  Split the data into training and testing sets using an 80:20
    stratified split with `random_state=42`.

### Dataset split

  Dataset            Records
  ---------------- ---------
  Total                  569
  Training (80%)         455
  Testing (20%)          114
  Input features          30

The stratified split preserves the approximate class distribution in
both subsets.

## 3. GitHub Repository link:


## 4. Machine-Learning Models

Five classification algorithms were implemented:

1.  Logistic Regression -- a linear classification algorithm used as
    a baseline. `StandardScaler` was applied.
2.  Decision Tree -- a tree-based model that makes predictions using
    feature-based decision rules.
3.  K-Nearest Neighbors (KNN) -- classifies an observation based on
    the labels of its nearest neighbours. `StandardScaler` was applied.
4.  Gaussian Naive Bayes -- a probabilistic classifier based on
    Bayes' theorem and the Gaussian feature-distribution assumption.
5.  Support Vector Machine (SVM) -- implemented with an RBF kernel.
    `StandardScaler` was applied.

## 5. Model Evaluation

All five models were evaluated on the same held-out test set of 114
records.

  ------------------------------------------------------------------------------------------
  Model            Accuracy          AUC    Precision       Recall     F1-score          MCC
  ------------ ------------ ------------ ------------ ------------ ------------ ------------
  Logistic           0.9649   **0.9960**       0.9750       0.9286       0.9512       0.9245
  Regression                                                                    

  Decision           0.9298       0.9246       0.9048       0.9048       0.9048       0.8492
  Tree                                                                          

  KNN                0.9561       0.9823       0.9744       0.9048       0.9383       0.9058

  Gaussian           0.9386       0.9934       1.0000       0.8333       0.9091       0.8715
  Naive Bayes                                                                   

  SVM (RBF)      **0.9737**       0.9947   **1.0000**   **0.9286**   **0.9630**   **0.9442**
  ------------------------------------------------------------------------------------------

### Observations
### Model Name	Observation about the Model's Performance:
##** Support Vector Machine (SVM):	## The best performer overall, with the highest accuracy (97.37%), F1-score (0.9630) and MCC (0.9442). It reached 100% precision, so it raised no false alarms, and recall was 92.86%. It correctly identified 39 of the 42 malignant cases, missing only 3.

Logistic Regression	A very close second, with 96.49% accuracy and the highest AUC of 0.9960, showing it separates the two classes extremely well. Precision (97.50%) and recall (92.86%) were both strong, and its results were consistent across metrics.

KNN	Solid results with 95.61% accuracy and an AUC of 0.9823. Precision was high at 97.44%, but recall of 90.48% means it missed a few more malignant cases than Logistic Regression and SVM.

Naive Bayes	Reached 93.86% accuracy and a high AUC of 0.9934, with perfect precision (100%), so every case it flagged as malignant truly was. The trade-off was a lower recall of 83.33%, meaning it missed 7 actual malignant cases, which is a notable weakness in a medical setting.

Decision Tree	The weakest of the five, with 92.98% accuracy and the lowest AUC (0.9246). It performed reasonably well, but its precision, recall and F1-score all trailed the other models.

## Overall Winner for Your Dataset
Best overall model: SVM
Support Vector Machine (SVM) is the overall winner because it achieved the highest accuracy (97.37%), F1-score (0.9630), and MCC (0.9442). It also maintained excellent precision and recall, making it the strongest overall performer among the five models.

-   
## 6. Project Structure

Machine_Learning_Models/
├── app.py
├── requirements.txt
├── README.md
├── 01_data_preprocessing.ipynb
└── 02_model_implementation.ipynb
└── data.csv
├── data/
│   └──raw
        ├── data.csv
    └──processed/
│       ├── train_data.csv
│       └── test_data.csv
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   └── svm.pkl

    
    

