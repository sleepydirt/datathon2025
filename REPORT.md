## Datathon TM-74 Report

### 1. Data analysis

From the data, we note that there is a severe class imbalance with the Suspect and Pathologic classes having ~300 total combined compared to ~1600 examples for the Normal class.

Additionally, we identified the CLASS column to be highly correlated with our target NSP, hence the column was dropped. The dataset also contained 11 duplicate rows, which are dropped as well.

- LB vs Mean: Lower LBs and lower Mean tends to be associated with pathological cases (NSP = 3) while higher LBs and higher Mean tends to be a mix of healthy (NSP = 1) and suspect (NSP = 2) cases.
- Width vs Variance: Higher widths & variance tend to be associated with pathological cases (NSP = 3).
- MSTV vs ALTV: Lower MSTV and higher ALTV tends towards suspect (NSP = 2) and pathological cases (NSP = 3).
- Mean vs Variance: Low Mean tends to pathological cases (NSP = 3), quite clear separation while higher Mean tends to suggest suspect cases (NSP = 2).

### 2. Baseline models

We trained a total of four models of different architectures:

- Logistic Regression
- Support Vector Machines
- Random Forest
- Neural Networks

In general, tree-based methods like Random Forest proved to be a good baseline, showing relatively balanced accuracy among all classes despite the class imbalance. For other models, they are either too simple to learn the features during training, or overfit to the training data easily. Additionally, they suffer from class imbalance issues as the Recall for the Suspect and Pathologic classes are way lower than the average accuracy. For statistics, please refer to the notebook in `data_exploration/datathon.ipynb`.

### 3. Feature engineering

Indicator variables are added to help the model learn specific features. Based on our analysis in `1`, columns on decelerations tend to have many zeroes. We identified that the presence of decelerations are likely to result in suspect or pathologic cases.

Other statistical features like the variance-to-mean ratio and deviation from mean are also added, given the presence of histogram features (`Mean, Max, Min, Width ...`)

### 4. Training

Building on our hypothesis that tree-based models are best suited for our dataset, we explored [LightGBM](https://github.com/microsoft/LightGBM) and [XGBoost](https://github.com/dmlc/xgboost), both of which are [Gradient Boosted Decision Trees](https://developers.google.com/machine-learning/decision-forests/intro-to-gbdt) algorithms which combine multiple weak learners to form a strong model. These models tend to be more complex than random forests, which helps it to learn the features better.

To address the class imbalance, a 80/20% train test stratified split was performed. To avoid data leakage, the 20% test set was withheld and used only for evaluation. This gives us an estimate of the model's performance on unseen data.

Additionally, we computed [class weights](https://www.geeksforgeeks.org/machine-learning/how-does-the-classweight-parameter-in-scikit-learn-work/) which gives a higher weightage to under-represented classes in an imbalanced dataset, helping to improve the macro-F1 and balanced accuracy for our Suspect and Pathologic classes.

Further hyperparameter tuning is also done to increase the accuracy of our model further. We identified LightGBM to be the best performing model, achieving a macro-F1 score of 93.14% and balanced accuracy of 91.76%:

_LightGBM, 20% test set evaluation_

```
Accuracy: 0.9645
Precision (weighted): 0.9643
Recall (weighted): 0.9645
F1-score (macro): 0.9314
Balanced Accuracy: 0.9176
              precision    recall  f1-score   support

           1       0.98      0.99      0.98       330
           2       0.89      0.88      0.89        58
           3       0.97      0.89      0.93        35

    accuracy                           0.96       423
   macro avg       0.95      0.92      0.93       423
weighted avg       0.96      0.96      0.96       423
```

_Random Forest, 20% test set evaluation_

```
Accuracy: 0.9527
Precision (weighted): 0.9517
Recall (weighted): 0.9527
F1-score (macro): 0.9159
Balanced Accuracy: 0.8927
              precision    recall  f1-score   support

           1       0.96      0.99      0.97       330
           2       0.90      0.78      0.83        58
           3       0.97      0.91      0.94        35

    accuracy                           0.95       423
   macro avg       0.94      0.89      0.92       423
weighted avg       0.95      0.95      0.95       423
```

We prioritise a balance between precision and recall over simply maximising accuracy. $ recall = \dfrac{TP}{TP+FN}$, which means that a lower recall indicates a larger number of false negatives. We aim to improve this since false negatives are a great cause for concern in the medical industry. In comparison to the Random Forest model, the LightGBM model achieves 10% greater recall on the Suspect class, despite a 1% decrease in precision, which is probably within expected margin of error.

For more comparisons against other models, please refer to the notebook in `data_exploration/datathon.ipynb`.

### Citations

Citations are given inline where possible.
