# Loan Eligibility Prediction using Machine Learning Algorithms

![A client receiving a form to sign](https://github.com/hazelapondi/FUTURE_DS_03/blob/main/imgs/pexels-olly-resize.png)

First begin by importing the essential libraries for data manipulation (pandas, numpy) and visualization (matplotlib).

The dataset is then loaded from a CSV file containing loan data.

* **dataset.head()** shows the first few rows of the dataset to get an idea of the structure.
* **dataset.shape** reveals that the dataset contains 614 rows and 13 columns.
* **dataset.info()** provides a summary of the dataset, showing data types and missing values in columns like Gender, Married, and Credit_History.

Create a crosstab to examine how  *Credit_History*  influences  *Loan_Status*  (approved or not).
Then use Boxplots and histograms to explore the distributions of  *ApplicantIncome* ,  *CoapplicantIncome* , and  *LoanAmount* . Since  *LoanAmount*  is right-skewed, apply a log transformation to normalize it and so on.

Missing values are handled by filling categorical features like  *Gender* ,  *Married* , and  *Credit_History*  with their mode (most frequent value), and numerical features like *LoanAmount*  with the mean.

Create a new feature  *TotalIncome* , by combining  *ApplicantIncome*  and  *CoapplicantIncome* , then apply log transformation to normalize it ( *TotalIncome_log* ).

Ensure to encode categorical variables before proceeding to modelling. For instance the target variable  *Loan_Status*  is encoded to binary values (1 for 'Y', 0 for 'N').

Split the data into training and testing sets using train_test_split (80% for training, 20% for testing)

Apply various models to determine the best. The model's accuracy is calculated on the test set using accuracy_score.

We opt to use the **Naive Bayes** algorithm which has an accuracy of 82.93% as opposed to the **Decision Tree Classifier**'s of 70.73%.
