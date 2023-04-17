import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


df = pd.read_csv('star_classification.csv')
print(df.head())


print(df.info())


duplicated_rows = df[df.duplicated()]
print(duplicated_rows.shape)


print(df.isna().sum())


enc = OrdinalEncoder()
df['class'] = enc.fit_transform(df[['class']])
df['class'].head(10)


X = df.drop(columns=['class'])
y = df.loc[:, ['class']]
minmax = MinMaxScaler()
scaled = minmax.fit_transform(X)


best_feature = SelectKBest(score_func=chi2)
fit = best_feature.fit(scaled, y)


feature_score = pd.DataFrame({
    'feature' : X.columns,
    'score': fit.scores_
})


feature_score.sort_values(by=['score'], ascending=False, inplace=True)
print(feature_score)


std = StandardScaler()
scaled = std.fit_transform(X)
scaled = pd.DataFrame(scaled, columns=X.columns)
print(scaled.head())


data_standardization = y.join(scaled)


X = data_standardization.loc[:, ['redshift', 'plate', 'spec_obj_ID', 'z', 'MJD', 'i', 'u', 'r', 'g']]
y = data_standardization.loc[:, 'class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


k_neighbors = 30
metrics = ['euclidean', 'manhattan']


accuracy_total = []
for k in range(1, k_neighbors+1, 1):
    accuracy_k = []
    for metric in metrics:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric=metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_k.append(accuracy)
    accuracy_total.append(accuracy_k)


accuracy_df = pd.DataFrame(np.array(accuracy_total), columns=metrics)
k_df = pd.DataFrame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], columns=['k'])
accuracy_join= k_df.join(accuracy_df)


plt.plot(accuracy_join['k'], accuracy_join['euclidean'],'o' ,label='euclidean')
plt.plot(accuracy_join['k'], accuracy_join['manhattan'],'o', label='manhattan')

plt.legend()
plt.xlabel('k')
plt.ylabel('accuracy score')
plt.show()


knn = KNeighborsClassifier(n_neighbors=6, weights='distance', metric='manhattan')
print(knn.fit(X_train, y_train))


y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))