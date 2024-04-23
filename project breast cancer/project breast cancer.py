import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




df = pd.read_csv("Breast_cancer_data.csv")
print(df.head(5))
print(df.info())
print(df.isnull().sum())


selected_columns = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']
selected_data = df[selected_columns]
# Calculating the correlation matrix
correlation_matrix = selected_data.corr()
# Creating the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",
            xticklabels=selected_columns, yticklabels=selected_columns)
plt.title('Correlation Heatmap')
plt.show()

#we wiil not use mean_perimeter and mwan_area

df = df[["mean_radius", "mean_texture", "mean_smoothness", "diagnosis"]]



fig, axes = plt.subplots(1,3,figsize=(18,6), sharey=True)
sns.histplot(df, ax=axes[0], x="mean_radius", kde=True, color='r')
sns.histplot(df, ax=axes[1], x="mean_smoothness", kde=True, color='b')
sns.histplot(df, ax=axes[2], x="mean_texture", kde=True)
plt.show()

x= df.drop(columns='diagnosis', axis=1)
y= df['diagnosis']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test)
accuracy = accuracy_score(y_test,Y_pred)*100
print('accuracy_Naive Bayes: %.3f' %accuracy)



