# Decision Trees and Random Forests - Exercises

This folder contains practical exercises on Decision Trees and Random Forests using real Kaggle datasets.

## Files

- `Exercises_Decision_Trees_RandomForest.ipynb` - Main exercises notebook

## Datasets Required

### 1. Regression Dataset (Exercise 2)

**Source**: [2022 Regression Data Challenge](https://www.kaggle.com/competitions/2022-regression-data-challenge/data)

**Download Instructions**:
1. Go to https://www.kaggle.com/competitions/2022-regression-data-challenge/data
2. Download `train.csv`
3. Create a `data` folder in this directory
4. Save the file as `data/regression_train.csv`

**Using Kaggle CLI** (alternative):
```bash
kaggle competitions download -c 2022-regression-data-challenge
unzip 2022-regression-data-challenge.zip -d data/
mv data/train.csv data/regression_train.csv
```

### 2. Classification Dataset (Exercises 3 & 4)

**Source**: [Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10/data)

**Download Instructions**:
1. Go to https://www.kaggle.com/competitions/playground-series-s6e10/data
2. Download both `train.csv` and `test.csv`
3. Create a `data` folder in this directory (if not already done)
4. Save the files as:
   - `data/classification_train.csv`
   - `data/classification_test.csv`

**Using Kaggle CLI** (alternative):
```bash
kaggle competitions download -c playground-series-s6e10
unzip playground-series-s6e10.zip -d data/
mv data/train.csv data/classification_train.csv
mv data/test.csv data/classification_test.csv
```

## Setup Kaggle CLI

If you want to use the Kaggle CLI for automatic downloads:

1. Install kaggle package:
```bash
pip install kaggle
```

2. Set up authentication:
   - Go to https://www.kaggle.com/settings/account
   - Click "Create New API Token"
   - This downloads `kaggle.json`
   - Move it to `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<YourUsername>\.kaggle\kaggle.json` (Windows)
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## Directory Structure

```
week.10/
├── Exercises/
│   ├── Exercises_Decision_Trees_RandomForest.ipynb
│   ├── README.md (this file)
│   ├── data/
│   │   ├── regression_train.csv
│   │   ├── classification_train.csv
│   │   └── classification_test.csv
│   └── (outputs):
│       ├── submission.csv
│       └── model_summary_report.txt
├── 16a-Decision_trees.ipynb
└── 16b-Random-Forest.ipynb
```

## Exercise Overview

### Exercise 1: Decision Tree Basics (Iris Dataset)
- Training/testing fundamentals
- Overfitting analysis
- Tree visualization

### Exercise 2: Regression with Real Data
- Load and explore regression dataset
- Tree depth optimization
- Decision Tree vs Random Forest comparison
- Feature importance analysis

### Exercise 3: Classification with Real Data
- Load and explore classification dataset
- Decision Tree vs Random Forest comparison
- Feature importance and hyperparameter tuning
- Effect of tree count on performance

### Exercise 4: Hyperparameter Tuning
- Systematic grid search
- Cross-validation analysis
- Final model evaluation
- Confusion matrix and classification metrics

### Exercise 5: Challenge & Kaggle Submission
- Generate predictions for test set
- Complete model analysis
- Write summary report
- Prepare for Kaggle competition

## Learning Outcomes

After completing these exercises, you will:

✓ Understand Decision Tree fundamentals (splits, depth, overfitting)
✓ Apply trees to regression and classification problems
✓ Compare single trees vs ensemble methods
✓ Interpret feature importance
✓ Perform hyperparameter tuning with cross-validation
✓ Evaluate models with multiple metrics
✓ Work with real-world Kaggle datasets
✓ Prepare predictions for submission

## Tips

1. **Start simple**: Begin with Exercise 1 using the Iris dataset
2. **Download early**: Get the Kaggle datasets before starting Exercise 2
3. **Read error messages**: They guide you to missing files or data issues
4. **Experiment**: Try different hyperparameters and observe the effects
5. **Document findings**: Take notes on what works and why
6. **Compare results**: Keep track of metrics across different models

## Resources

- [Scikit-learn Decision Trees Documentation](https://scikit-learn.org/stable/modules/tree.html)
- [Scikit-learn Random Forests Documentation](https://scikit-learn.org/stable/modules/ensemble.html#forests)
- [Kaggle Learn - Decision Trees](https://www.kaggle.com/learn/decision-trees)
- [Kaggle Learn - Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)

## Questions?

Refer to the lecture notes in `16a-Decision_trees.ipynb` and `16b-Random-Forest.ipynb` for theoretical background.
