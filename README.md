# Stroke Prediction 🚀

Welcome to my project! This is all about predicting strokes using some cool machine learning algorithms and comparing how well they perform. I used a healthcare dataset with features like age, hypertension, and smoking status to train and test models.

## 🔍 Dataset Overview
The dataset, sourced from Kaggle, contains 4,910 rows and 11 features. Missing values in the BMI column were handled by removing incomplete rows. Categorical variables were converted to factors for compatibility with machine learning models.

## 📊 Algorithms Used
I tested out the following models:
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes**
- **Decision Tree**
- **Logistic Regression**
- **Random Forest**

## 📈 Results
Here’s how the models performed:
| **Model**             | **Accuracy** | **Precision** | **Recall** | **F1 Score** |
|-----------------------|--------------|---------------|------------|--------------|
| **KNN**               | 95.25%       | 95.83%        | 99.36%     | 97.56%       |
| **Naive Bayes**       | 90.84%       | 96.63%        | 93.69%     | 95.14%       |
| **Decision Tree**     | 95.72%       | 95.72%        | 100%       | 97.81%       |
| **Random Forest**     | 95.72%       | 95.72%        | 100%       | 97.81%       |

## 📊 Key Observations
- **Decision Tree** and **Random Forest** tied for the best performance, with perfect recall and high F1 scores.
- **KNN** also did a solid job with high recall, while **Naive Bayes** had great precision but slightly lower recall.

- # Stroke Prediction using Machine Learning 🚀

Welcome to my project! This is all about predicting strokes using some cool machine learning algorithms and comparing how well they perform. I used a healthcare dataset with features like age, hypertension, and smoking status to train and test models.

## 🔍 Dataset Overview
The dataset, sourced from Kaggle, contains 4,910 rows and 11 features. Missing values in the BMI column were handled by removing incomplete rows. Categorical variables were converted to factors for compatibility with machine learning models.

## 📊 Algorithms Used
I tested out the following models:
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes**
- **Decision Tree**
- **Logistic Regression**
- **Random Forest**

## 📈 Results
Here’s how the models performed:
| **Model**            | **Accuracy** | **Precision** | **Recall** | **F1 Score** |
|-----------------------|--------------|---------------|------------|--------------|
| **KNN**              | 95.25%       | 95.83%        | 99.36%     | 97.56%       |
| **Naive Bayes**       | 90.84%       | 96.63%        | 93.69%     | 95.14%       |
| **Decision Tree**     | 95.72%       | 95.72%        | 100%       | 97.81%       |
| **Random Forest**     | 95.72%       | 95.72%        | 100%       | 97.81%       |

## 📊 Key Observations
- **Decision Tree** and **Random Forest** tied for the best performance, with perfect recall and high F1 scores.
- **KNN** also did a solid job with high recall, while **Naive Bayes** had great precision but slightly lower recall.

## ⚙️ How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/YourUsername/Stroke-Prediction.git
   ```
2. Open the project in RStudio.
3. Install required libraries:
```bash
install.packages(c("caTools", "class", "e1071", "rpart", "randomForest", "caret", "ggplot2", "tidyr"))
```
4. Run the script and check out the results.

## Future Scope 💡 
Add a larger dataset to enhance the model's learning.
Try deep learning models for better predictions.
Fine-tune hyperparameters for optimized performance.


## Acknowledgments🤝 
Thanks to Kaggle for the dataset and to everyone who helped me learn machine learning along the way. 😄
