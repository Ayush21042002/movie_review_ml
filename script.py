import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import pickle

# reading the data from tsv file
df = pd.read_csv("moviereviews.tsv",sep="\t",)

# dropping those reviews where review is null
df.dropna(inplace=True)

# dropping those reviews where review is " "
blanks = []

for i,l,r in df.itertuples():
    if r.isspace():
        blanks.append(i)

df.drop(blanks,inplace=True)

# creating training and test data
X = df["review"]

y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# making a pipeline for handling feature extraction and training the model
text_clf = Pipeline([('tfdif',TfidfVectorizer()),('clf',LinearSVC())])

# training the model
text_clf.fit(X_train,y_train)

# getting the predictions
predictions = text_clf.predict(X_test)

# confusion matrix
print(confusion_matrix(y_test,predictions))
# classification report
print(classification_report(y_test,predictions))
# accuracy score
print(accuracy_score(y_test,predictions))

# saving the trained model
pickle.dump(text_clf,open("text_clf",'wb'))