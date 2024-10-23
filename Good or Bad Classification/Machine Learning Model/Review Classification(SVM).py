#import libraries
import pandas as pd         
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

#loading dataset
df = pd.read_csv(r'D:/MY AI PROJECTS/Good or Bad Classification/Dataset/TestReviews.csv')

#check class balance
print("Classs distribution:\n",df['condition'].value_counts())

#preprocessing the data (represent my 2 column names as x and y)
x = df['review']
y = df['condition']

#train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

#vectorizing the text data
vectorizer = TfidfVectorizer(max_features=10000)   #adjust it if needed
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

# Save the vectorizer
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

#initializing SVM model
svm = SVC(kernel='linear') #we can also try other kernels like RBF,Sigmoid kernels etc..

#training the model
svm.fit(x_train_vectorized,y_train)

#making predictions
y_pred = svm.predict(x_test_vectorized)

#evaluating the model
print('Accuracy :', accuracy_score(y_test,y_pred))
print('Classification Report:\n', classification_report(y_test,y_pred))
joblib.dump(svm, 'svm_review_classification_model.h5')
