import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,hamming_loss,zero_one_loss
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pdb

OUTPUT_PATH = "cleaned_data.csv"
HEADERS = ["clean_tweet","label"]
print (len(HEADERS))
dataset = pd.read_csv(OUTPUT_PATH)
print ('The shape of our features is:', dataset.shape)
vectorizer = TfidfVectorizer(stop_words=None,decode_error='ignore',min_df=5, max_df = 0.8,use_idf=True,ngram_range=(1,3),norm='l2')

def split_dataset(dataset, train_percentage, feature_headers, target_header):
    x=vectorizer.fit_transform(dataset['clean_tweet'].values.astype('U'))
    g = dataset['label']
    y=np.array([int(lines) for lines in g])
    train_x, test_x, train_y, test_y = train_test_split(x,y,train_size = train_percentage,shuffle = False)
    return train_x, test_x, train_y, test_y

def random_forest_classifier(features, target):
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf

train_x, test_x, train_y, test_y = split_dataset(dataset, 0.1, HEADERS[0:-1], HEADERS[-1])

print ("Train_x Shape :: ", train_x.shape)
print ("Train_y Shape :: ", train_y.shape)
print ("Test_x Shape :: ", test_x.shape)
print ("Test_y Shape :: ", test_y.shape)

trained_model = random_forest_classifier(train_x, train_y)
print ("Trained model :: ", trained_model)
predictions = trained_model.predict(test_x)
for i in range(0, 10):
    print ("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))


print ("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
print ("Test Accuracy  :: ", accuracy_score(test_y, predictions))
print ("Test f1 score :: ",f1_score(test_y,predictions))
print ("Test hamming_loss :: ",hamming_loss(test_y,predictions))
print ("Test zero_one_loss :: ",zero_one_loss(test_y,predictions))
print (" Confusion matrix ", confusion_matrix(test_y, predictions))
