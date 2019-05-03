# cs5293sp19-project2

Title of the project: The UnRedactor

Description:
In the previous project the sensitive data like names, gender etc. were redacted. In this project we are going to unredact the data that was redacted i.e to predict the redacted name.

Modules/Packages needed:
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk
import glob
import io
from sklearn.feature_extraction.text import CountVectorizer
import re
import os
import pdb
import sys
from sklearn.metrics import accuracy_score
import numpy 

Procedure:
The project mainly involves two steps.
1.	Training the data
2.	Predicting the name by using test data
Firstly the names from the data are extracted using nltk i.e natural language toolkit. We have a parts of speech tag from where we could guess a whether a word is a noun,verb and etc. From there we extract the named entities by using ne_chunk that has a label called PERSON.
The first two functions in the program i.e  extraction() and extreactredact() are the functions to read the training and testing data.

Function to extract features from training dataset:
The main task in machine learning is to select the features correctly. The efficiency or accuracy depends on the
Features that we select. In our project I used features like length of the word without spaces, number of spaces ,number of words in the name, length of each word in a name if there are more than one words in a given name.
To tokenize the word I used nltk.word_tokenize and found the length of each word in a name. The function returns the features that I mentioned earlier.

Function to extract features from redacted dataset:
In this function we take the redacted document as the input. The idea behind the function is to extract the features of the names that are redacted. In order to get the number of spaces from the redacted document for the convenience I put a  special symbol i.e  @  to identify that it is a space. So while redacting  I replaced a name with more than one word with @ in place of space. My data now has names with @ in them instead of space. Now I split the data by using ‘ ‘ i.e space so that the words in the name do not get separated. Then I used contains() function to identify whether there is ‘@’ in the datafile to count the number of spaces as I replaced @ in the place of ‘ ’. This gives me number of spaces. Next if add 1 to the number of spaces I get the number of words in a name.The length of each word is given by the count of ‘blocks’ by using the count() function. The length of the word without spaces is given the length of the word-number of spaces in the word.

The next phase of the project deals with training the data. I used the datafiles in ‘train/*.txt’. The datafiles in here are read by using glob.glob(‘train/*.txt’). The names are sent into the x_train variable and the features are sent into y_train variable. I used GaussianNB to train the model. But before training the data we need to need to send the data in the form of an array. I used numpy to accomplish this task. Then the data is tested and the names are predicted.

Assumptions and Challenges:
1)The extraction of the number of spaces from a redacted file was a challenging task for me because if there is more than word in a name if word_tokenize is applied then there is chance of the name being split. The other thing is if the name that has more than one word is directly redacted it would be difficult to extract the spaces as the spaces also get redacted.
2)I found it difficult in getting the accuracy. The thing that I noticed is the features of the entity impact the accuracy score a lot. The features with which we trained the model if they are present in the testing datasets then we would be getting the predicted names.
3) Few names were not completely redacted.
4)I assumed k=1 for the top k entities.

Observations:
1)The thing that I felt is the accuracy can be improved if I would have taken the adjacent words like bi-grams, tri-grams etc.

References:

https://www.youtube.com/watch?v=OGxgnH8y2NM&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
https://www.youtube.com/watch?v=Up6KLx3m2ww



 
