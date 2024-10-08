# Credit-Card-Fraud-Detection-using-Scikit-Learn-and-Snap-ML

### Project Overview:
This project involves using machine learning techniques to build models that classify fraudulent transactions. It compares the performance of two classifiers—Decision Tree and Support Vector Machine (SVM)—implemented using both Scikit-Learn and Snap ML. The primary goal is to predict fraudulent credit card transactions, focusing on training speed and model performance.

### Dataset:
The dataset used in this project is the **Credit Card Fraud Detection** dataset from Kaggle. It contains highly imbalanced data, where fraudulent transactions account for only 0.172% of the total.

**Link to dataset**: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

### Project Structure:
The notebook is divided into the following sections:

1. **Introduction**: Explains the problem of predicting fraudulent credit card transactions and introduces the dataset.
2. **Data Preprocessing**: Handles loading and preparing the data for model training.
3. **Decision Tree Models**: Implements Decision Tree classifiers using Scikit-Learn and Snap ML.
4. **Support Vector Machine (SVM) Models**: Implements SVM classifiers using Scikit-Learn and Snap ML.
5. **Model Evaluation**: Compares the performance of Scikit-Learn and Snap ML models using metrics such as accuracy, ROC-AUC score, and hinge loss.
6. **Conclusion**: Provides a summary of the results, highlighting the benefits of using Snap ML for faster model training.

### Requirements:
To run this notebook, you will need to install the following Python libraries:

- `numpy`
- `pandas`
- `scikit-learn`
- `snapml`
- `matplotlib`
- `seaborn`

You can install the required libraries using:
```bash
pip install numpy pandas scikit-learn snapml matplotlib seaborn
```

### Usage:
1. Download the dataset from Kaggle.
2. Install the required libraries.
3. Run the Jupyter notebook to train and evaluate the Decision Tree and SVM classifiers.
4. Compare the performance of Scikit-Learn and Snap ML models.

### Results:
The project demonstrates that Snap ML can accelerate the training process while maintaining compatibility with Scikit-Learn metrics and data preprocessors. Although the models built with Scikit-Learn and Snap ML yield similar results in terms of classification accuracy and ROC-AUC score, Snap ML significantly reduces the training time, making it advantageous for large datasets.
