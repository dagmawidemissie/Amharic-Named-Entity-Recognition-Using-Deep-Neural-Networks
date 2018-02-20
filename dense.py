################Final experiment of ANER using Dense layer and embedding Layer#################

import pandas
import numpy as numpy
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.utils.np_utils import to_categorical
from keras.layers import LSTM,Embedding,Bidirectional,TimeDistributed,Input,merge
from keras.preprocessing import sequence
from keras.models import Model
from keras import backend as K
from sklearn import model_selection
from sklearn.metrics import classification_report



# fix random seed for reproducibility
seed = 7
#numpy.random.seed(seed)
maxlength=75

# load dataset
raw = pandas.read_csv("correctedindex.csv", header=None)
sentence = raw.values
seq=pandas.read_csv("labels.csv", header=None)
labels = seq.values


####load training data
Sentences = sentence[:,].astype(int)
Labels = labels[:,].astype(int)
Sentences=numpy.asarray(Sentences,dtype='int32')
Labels=numpy.asarray(Labels,dtype='int32')



# split the arrays in to 10 folds

splitedX=numpy.vsplit(Sentences,10)
splitedY=numpy.vsplit(Labels,10)


# prepare folds
#fold 1
X_train1=numpy.concatenate((splitedX[1],splitedX[2],splitedX[3],splitedX[4],splitedX[5],splitedX[6],splitedX[7],splitedX[8],splitedX[9]), axis=0)
Y_train1=numpy.concatenate((splitedY[1],splitedY[2],splitedY[3],splitedY[4],splitedY[5],splitedY[6],splitedY[7],splitedY[8],splitedY[9]), axis=0)

X_test1=splitedX[0]
Y_test1=splitedY[0]


#fold 2
X_train2=numpy.concatenate((splitedX[0],splitedX[2],splitedX[3],splitedX[4],splitedX[5],splitedX[6],splitedX[7],splitedX[8],splitedX[9]), axis=0)
Y_train2=numpy.concatenate((splitedY[0],splitedY[2],splitedY[3],splitedY[4],splitedY[5],splitedY[6],splitedY[7],splitedY[8],splitedY[9]), axis=0)

X_test2=splitedX[1]
Y_test2=splitedY[1]

#fold 3
X_train3=numpy.concatenate((splitedX[0],splitedX[1],splitedX[3],splitedX[4],splitedX[5],splitedX[6],splitedX[7],splitedX[8],splitedX[9]), axis=0)
Y_train3=numpy.concatenate((splitedY[0],splitedY[1],splitedY[3],splitedY[4],splitedY[5],splitedY[6],splitedY[7],splitedY[8],splitedY[9]), axis=0)

X_test3=splitedX[2]
Y_test3=splitedY[2]



#fold 4
X_train4=numpy.concatenate((splitedX[0],splitedX[1],splitedX[2],splitedX[4],splitedX[5],splitedX[6],splitedX[7],splitedX[8],splitedX[9]), axis=0)
Y_train4=numpy.concatenate((splitedY[0],splitedY[1],splitedY[2],splitedY[4],splitedY[5],splitedY[6],splitedY[7],splitedY[8],splitedY[9]), axis=0)

X_test4=splitedX[3]
Y_test4=splitedY[3]


#fold 5
X_train5=numpy.concatenate((splitedX[0],splitedX[1],splitedX[2],splitedX[3],splitedX[5],splitedX[6],splitedX[7],splitedX[8],splitedX[9]), axis=0)
Y_train5=numpy.concatenate((splitedY[0],splitedY[1],splitedY[2],splitedY[3],splitedY[5],splitedY[6],splitedY[7],splitedY[8],splitedY[9]), axis=0)

X_test5=splitedX[4]
Y_test5=splitedY[4]


#fold 6
X_train6=numpy.concatenate((splitedX[0],splitedX[1],splitedX[2],splitedX[3],splitedX[4],splitedX[6],splitedX[7],splitedX[8],splitedX[9]), axis=0)
Y_train6=numpy.concatenate((splitedY[0],splitedY[1],splitedY[2],splitedY[3],splitedY[4],splitedY[6],splitedY[7],splitedY[8],splitedY[9]), axis=0)

X_test6=splitedX[5]
Y_test6=splitedY[5]

#fold 7
X_train7=numpy.concatenate((splitedX[0],splitedX[1],splitedX[2],splitedX[3],splitedX[4],splitedX[5],splitedX[7],splitedX[8],splitedX[9]), axis=0)
Y_train7=numpy.concatenate((splitedY[0],splitedY[1],splitedY[2],splitedY[3],splitedY[4],splitedY[5],splitedY[7],splitedY[8],splitedY[9]), axis=0)

X_test7=splitedX[6]
Y_test7=splitedY[6]

#fold 8
X_train8=numpy.concatenate((splitedX[0],splitedX[1],splitedX[2],splitedX[3],splitedX[4],splitedX[5],splitedX[6],splitedX[8],splitedX[9]), axis=0)
Y_train8=numpy.concatenate((splitedY[0],splitedY[1],splitedY[2],splitedY[3],splitedY[4],splitedY[5],splitedY[6],splitedY[8],splitedY[9]), axis=0)

X_test8=splitedX[7]
Y_test8=splitedY[7]

#fold 9
X_train9=numpy.concatenate((splitedX[0],splitedX[1],splitedX[2],splitedX[3],splitedX[4],splitedX[5],splitedX[6],splitedX[7],splitedX[9]), axis=0)
Y_train9=numpy.concatenate((splitedY[0],splitedY[1],splitedY[2],splitedY[3],splitedY[4],splitedY[5],splitedY[6],splitedY[7],splitedY[9]), axis=0)

X_test9=splitedX[8]
Y_test9=splitedY[8]

#fold 10
X_train10=numpy.concatenate((splitedX[0],splitedX[1],splitedX[2],splitedX[3],splitedX[4],splitedX[5],splitedX[6],splitedX[7],splitedX[8]), axis=0)
Y_train10=numpy.concatenate((splitedY[0],splitedY[1],splitedY[2],splitedY[3],splitedY[4],splitedY[5],splitedY[6],splitedY[7],splitedY[8]), axis=0)

X_test10=splitedX[9]
Y_test10=splitedY[9]

##########splitting finished###################
########## Concatenate training and test folds in to different arrays

Sentence_concat=numpy.concatenate((X_train1,X_train2,X_train3,X_train4,X_train5,X_train5,X_train6,X_train7,X_train8,X_train9), axis=0)
Label_concat=numpy.concatenate((Y_train1,Y_train2,Y_train3,Y_train4,Y_train5,Y_train5,Y_train6,Y_train7,Y_train8,Y_train9), axis=0)

Sentence_concattest=numpy.concatenate((X_test1,X_test2,X_test3,X_test4,X_test5,X_test6,X_test7,X_test8,X_test9,X_test10),axis=0)
Label_concattest=numpy.concatenate(((Y_test1,Y_test2,Y_test3,Y_test4,Y_test5,Y_test6,Y_test7,Y_test8,Y_test9,Y_test10)),axis=0)


# divide to equal 10 folds to get original 2 d arrays
# final training datas
Sentence_train=numpy.vsplit(Sentence_concat,10)
Label_train=numpy.vsplit(Label_concat,10)

# final Test datas
Sentence_test=numpy.vsplit(Sentence_concattest,10)
Label_test=numpy.vsplit(Label_concattest,10)

Sentence_train=numpy.asarray(Sentence_train,dtype='int32')
Label_train=numpy.asarray(Label_train,dtype='int32')

Sentence_test=numpy.asarray(Sentence_test,dtype='int32')
Label_test=numpy.asarray(Label_test,dtype='int32')

# open file stream to save accuracies for each training Fold
F=open('result.txt','a')

# Iterate through each training data and train the network
# take the mean of the accuracies to get over all accuracy

# find the length of the longest sentence:
seq_length = max([len(s) for s in Sentences])
# find the number of categories, by streaming each y into
# a set and count the number of unique y values
no_cat = set([ys for sent in Labels for ys in sent])
no_cat =len(no_cat)
Label =  numpy.array([np_utils.to_categorical(seq, no_cat) for seq in Labels])
print 'Network Training Started ..... \n'
print '=================================================\n'
# an array to store each iterations accuracy
scores=numpy.zeros(0,dtype=float)
for index in range(0,10) :
     Ytrain =  numpy.array([np_utils.to_categorical(seq, no_cat) for seq in  Label_train[index]])
     Xtrain =  Sentence_train[index]
     Xtest=Sentence_test[index]
     Ytest =  numpy.array([np_utils.to_categorical(seq, no_cat) for seq in  Label_test[index]])	 
     print  'Training on Fold ',index+1, '\n'
     # Defining the model
     Input_layer = Input(shape=(seq_length,), dtype='int32')
     Embedding_layer = Embedding(input_dim=4600, output_dim = 100, input_length=seq_length, dropout=0.2, mask_zero=True)(Input_layer)
     #Recurrent_layer=LSTM(100,return_sequences=True)(Embedding_layer)
     Dense_layer= TimeDistributed(Dense(128, activation='tanh'))(Embedding_layer)
     Output_layer = TimeDistributed(Dense(5, activation='softmax'))(Dense_layer)
     # create the model
     model = Model(input=Input_layer, output=Output_layer)
     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
     # Start building the model
     model.fit(Xtrain, Ytrain, epochs=20)
     score=model.evaluate(Xtest,Ytest, batch_size=32, verbose=1)
     # evaluating using scikit learn to get detailed metrics
     Predict=model.predict(Xtest)
     #Convert the final outputs (probablity distributions of  softmax layer to originally encoded values (0,1,2,3,4))
     Decoded=numpy.argmax(Predict,axis=-1)
     Ytrue=Label_test[index]
     Decoded=numpy.asarray(Decoded,dtype='int32')
     Decoded=Decoded.flatten()
     print Decoded.shape
     ##print Ytrue.shape
     Ytrue=Ytrue.flatten()
     Ytrue=numpy.asarray(Ytrue,dtype='int32')
     report = classification_report(Ytrue, Decoded)
     print(report)
     foldscore=score[1]*100
     scores=numpy.append(scores,foldscore)
     print 'Accuracy at fold ',index+1,'= ',foldscore,' %',score[0],'\n'
print 'Network training finished .... \n'
print 'Calculating Overall Accuracy of the model...\n'
print 'Summary of Network accuracy Scores...\n'
for ind in range(len(scores)):
    print 'Fold ',ind+1,'-------------------',scores[ind],' %'
print '======================================================\n'
print 'Overall Network Accuracy =',numpy.mean(scores),' %\n'
print 'Calculating Detailed metric for the network \n'




	 
	 










