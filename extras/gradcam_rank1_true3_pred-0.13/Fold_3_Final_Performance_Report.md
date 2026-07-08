# Fold 3 Final Performance Report

## 1. Validation (During Training)
- Best Epoch: 25
- Val QWK: 0.8663
- Val Loss: 0.3591

## 2. Test Set Evaluation (EyePACS 2015 - 53,576 images)
- QWK: 0.8380
- Accuracy: 0.8051

### Classification Report (EyePACS)
```text
              precision    recall  f1-score   support

           0     0.9166    0.9690    0.9421     39533
           1     0.3394    0.3246    0.3318      3762
           2     0.7997    0.2378    0.3665      7861
           3     0.2129    0.8896    0.3436      1214
           4     0.8432    0.5439    0.6613      1206

    accuracy                         0.8051     53576
   macro avg     0.6224    0.5930    0.5291     53576
weighted avg     0.8413    0.8051    0.7949     53576

```

## 3. Test Set Evaluation (APTOS 2019 - 3,662 images)
- QWK: 0.9298
- Accuracy: 0.7851

### Classification Report (APTOS)
```text
              precision    recall  f1-score   support

           0     0.9977    0.9806    0.9891      1805
           1     0.7689    0.4946    0.6020       370
           2     0.7516    0.5816    0.6558       999
           3     0.2139    0.6684    0.3241       193
           4     0.7737    0.7186    0.7452       295

    accuracy                         0.7851      3662
   macro avg     0.7012    0.6888    0.6632      3662
weighted avg     0.8481    0.7851    0.8044      3662

```
