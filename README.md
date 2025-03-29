# Feature Selection for Nearest Neighbors

## Introduction
In this project, our goal is to run the nearest neighbor algorithm on a dataset and find the specific feature combinations that provide us with the highest accuracy. We will accomplish this by iterating through specific combinations of features chosen by two methods: forward selection and backward elimination. 

The nearest neighbor algorithm uses leave-one-out cross-validation, which tests each data point against the others in the dataset to determine the accuracy of the selected features. We then analyze the performance of the two feature selection algorithms to determine which one is better under specific conditions.

## Algorithm Overview
### Forward Selection
Forward selection iteratively adds features to the feature set currently considered by the nearest neighbors algorithm. Initially, the feature set is empty, and the algorithm selects the feature that maximizes accuracy. This process continues until all features are included.

#### Small Dataset (6 Features)
- Default rate: **0.7**
- Adding feature **3**: Accuracy **83.4%**
- Adding feature **1**: Accuracy **98.2%** (Highest accuracy)
- Adding feature **6**: Accuracy **93.8%**
- Additional features reduce accuracy, final accuracy with all features: **80.4%**

#### Large Dataset (40 Features)
- Default rate: **66.7%**
- Adding feature **21**: Accuracy **85.3%**
- Adding feature **10**: Accuracy **96.3%** (Highest accuracy)
- Further additions cause a steady decline, final accuracy with all features: **67%**

### Backward Elimination
Backward elimination starts with all features and iteratively removes the least significant feature at each step.

#### Small Dataset (6 Features)
- Initial accuracy with all features: **80.4%**
- Removing feature **4**: Accuracy **85.2%**
- Removing feature **6**: Accuracy **90.2%**
- Removing feature **2**: Accuracy **95.4%**
- Removing feature **5**: Accuracy **98.2%** (Highest accuracy)

#### Large Dataset (40 Features)
- Initial accuracy: **67%**
- Removing feature **1**: Accuracy **68.8%**
- Removing feature **7**: Accuracy **69.7%**
- Small incremental increases until peak at **85.3%** with only feature **21**

## Runtime Analysis
| Dataset | Forward Search | Backward Elimination |
|---------|---------------|----------------------|
| Small (6 features, 500 instances) | 27.84 sec | 27.28 sec |
| Large (40 features, 1000 instances) | 1.16 hours | 1.159 hours |

Both methods take significant time due to leave-one-out cross-validation.

## Extra Credit: Wine Dataset
Dataset from UCI Machine Learning Repository: [Wine Dataset](https://archive.ics.uci.edu/dataset/109/wine)

- **Default rate**: **58.99%**
- Feature **7** (Flavonoids): **70.22%**
- Feature **10** (Color Intensity): **92.7%**
- Feature **13** (Proline): **96.63%**
- Maximum accuracy: **98.88%** with 8 features

## Conclusion
- Forward selection finds the optimal feature set **faster**.
- Adding more features to nearest neighbors **adds noise** rather than improving accuracy.
- Since nearest neighbors is computationally expensive, forward selection is preferable for early termination scenarios.

## Algorithm Trace
Traces for forward selection and backward elimination are omitted for space.

## GitHub Repository
The code for this project is available at: [GitHub Repository](https://github.com/KushMomaya/CS170_Project2_kmoma001-main)

### File Breakdown
- **Main**: Interface to select dataset and search method.
- **Feature Selection**: Implements forward selection and backward elimination.
- **Cross Validation**: Contains nearest neighbor logic.
- **Extra Credit**: Loads and normalizes wine dataset.
- **Graph**: Plots accuracy graphs for both search methods.
