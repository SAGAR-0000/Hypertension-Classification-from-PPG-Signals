# Hypertension Classification from PPG Signals

## Final Results: 87.88% Accuracy

Advanced machine learning system for hypertension screening using Photoplethysmography (PPG) signals with **subject-independent validation**.

### Key Metrics
- **Accuracy**: 87.88% (Binary Classification)
- **Specificity**: 93.84% (Excellent for screening)
- **Sensitivity**: 62.69%
- **Validation**: Subject-independent (realistic performance)

---

## Project Structure

```
project/
├── 📓 Notebooks (Core Files)
│   ├── data_preparation.ipynb              # Baseline feature extraction
│   ├── data_preparation_advanced.py        # APG + Frequency features
│   ├── hypertension_classification.ipynb   # Baseline models (67%)
│   └── advanced_classification.ipynb       # Final models (87.88%) ⭐
│
├── Data
│   ├── data/                              # Raw .mat files (not included)
│   └── processed_data/
│       ├── ppg_features.csv               # Baseline features
│       └── ppg_features_advanced.csv      # 29 features, 25,201 samples
│
├── Results
│   └── results/
│       ├── binary_classification_results.csv
│       ├── individual_models_comparison.csv
│       └── selected_features.csv
│
└── Documentation
    ├── FINAL_RESULTS_SUMMARY.md           # Comprehensive results
    ├── COMPREHENSIVE_COMPARISON.md        # All approaches compared
    ├── project_report.md                  # Technical report
    └── walkthrough.md                     # Implementation guide
```

---


## Results Summary

| Approach | Accuracy | Key Finding |
|----------|----------|-------------|
| **Binary XGBoost** | **87.88%** | Best overall (Normal vs Hypertensive) |
| Binary Optimized | 87.55% | Feature selection didn't help |
| 3-Class XGBoost | 72.24% | Best 3-class performance |
| 3-Class RF | 68.20% | Solid baseline |
| Baseline | ~67% | Initial implementation |

---

## Technical Highlights

### Advanced Features (29 total)
1. **APG Features** (Most Important!)
   - b/a ratio (arterial stiffness)
   - a-wave, b-wave amplitudes
   - Statistical moments

2. **Frequency Domain**
   - LF/HF ratio (autonomic balance)
   - Spectral entropy
   - Dominant frequency

3. **Baseline Features**
   - Morphological (pulse intervals)
   - Physiological (HR, HRV)
   - Statistical (mean, std, skewness)

### Validation Strategy
✅ Subject-Independent (GroupShuffleSplit)  
✅ No Data Leakage (Group-based splitting)  
✅ Robust Preprocessing (RobustScaler + SMOTE)  
✅ Comprehensive Testing (7 different approaches)

---

## Documentation

### For Quick Overview
- **FINAL_RESULTS_SUMMARY.md** - All results and comparisons

### For Technical Details
- **project_report.md** - Methodology and clinical interpretation
- **walkthrough.md** - Implementation phases and decisions

### For Comparison
- **COMPREHENSIVE_COMPARISON.md** - Why each approach performed as it did

---

## Clinical Application

### Use Case: Primary Screening Tool
1. **Continuous PPG monitoring** detects suspicious cases
2. **High specificity (93.84%)** minimizes false alarms
3. **Flagged individuals** get clinical BP confirmation
4. **Normal cases** require minimal follow-up

### Advantages
✅ Non-invasive (PPG sensor only)  
✅ Fast (real-time inference)  
✅ Cost-effective (minimal hardware)  
✅ Clinically viable (93.84% specificity)

---

## Dependencies

```python
numpy>=1.20
pandas>=1.3
scikit-learn>=1.0
xgboost>=1.5
imbalanced-learn>=0.9
matplotlib>=3.5
seaborn>=0.11
scipy>=1.7
```

Install all:
```bash
pip install numpy pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn scipy
```

---

## Dataset

**Source**: Kaggle Blood Pressure Dataset (MIMIC-derived)  
**Link**: https://www.kaggle.com/datasets/mkachuee/BloodPressureDataset

**Characteristics**:
- 25,201 PPG segments
- 125 Hz sampling rate
- Real clinical data (MIMIC-III)
- Includes ABP for ground truth labels

---

## Future Improvements

To reach 90%+ accuracy:

### Option 1: Deep Learning (Recommended)
- 1D CNN/LSTM on raw PPG signals
- Expected: 89-93% accuracy
- Effort: 1-2 days

### Option 2: Multi-Modal
- Add ECG, demographics
- Expected: 92-95% accuracy
- Effort: 1 week

### Option 3: Hyperparameter Tuning
- Bayesian optimization (200+ iterations)
- Expected: 88.5-89.5% accuracy
- Effort: 2-4 hours

---

## Citation

If using this project, please cite:
```
Hypertension Classification from PPG Signals
Advanced Feature Engineering with Subject-Independent Validation
Final Accuracy: 87.88% (Binary Classification)
Dataset: Kaggle Blood Pressure Dataset (MIMIC-derived)
```

---
