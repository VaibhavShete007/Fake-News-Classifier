
import numpy
df= pd.read_csv('data.csv')
df=df.dropna()
## get independent features
x= df.drop('Label',axis=1)
y=df['Label']
x.shape
y.shape
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

# vocabularry size
voc_size=5000
# # one hot representation
messages= x.copy()
messages.reset_index(inplace=True)
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

from tensorflow.keras.preprocessing.text import one_hot
onehot_repr= [one_hot(words,voc_size)for words in corpus]
print(onehot_repr)
sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)
len(embedded_docs)

## creating model
embedding_vector_features= 40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length)) # making embedding layer
model.add(LSTM(100))  # one LSTM Layer with 100 neurons
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

import numpy as np
x_final= np.array(embedded_docs)
y_final= np.array(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_final,y_final,test_size=0.33,random_state=0)

# # model training
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=64)

y_pred= model.predict_classes(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm= confusion_matrix(y_test,y_pred)
print(cm)
ac= accuracy_score(y_test,y_pred)
print(ac)

# # adding dropout 

from tensorflow.keras.layers import Dropout

## creating model
embedding_vector_features= 40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length)) # making embedding layer
model.add(Dropout(0.3))
model.add(LSTM(100))  # one LSTM Layer with 100 neurons
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

import numpy as np
x_final= np.array(embedded_docs)
y_final= np.array(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_final,y_final,test_size=0.33,random_state=0)
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=64)
y_pred= model.predict_classes(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cmm= confusion_matrix(y_test,y_pred)
print(cmm)
ac= accuracy_score(y_test,y_pred)
print(ac)





