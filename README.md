## Datathon 2025: TM-74

Dataset: [Cardiotocography](https://archive.ics.uci.edu/dataset/193/cardiotocography)

Team TM-74:
- Aaron Soh Jun Qi
- Iain Demetriuss Chin Yi Rong
- A.Siva Meyyappan

### Usage
1. Install required libraries
```bash
$ pip install -r requirements.txt
```

2. Train the model in `datathon.ipynb`

3. Evaluate the model using the provided evaluation harness
```bash
$ python eval.py -m models/lgbm_final.pkl -e data/CTG.csv
```

### Results
LightGBM, 20% hidden test set, stratified
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