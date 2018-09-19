import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
import io
import csv
import json
import re
import random

data = []
text_data = json.load(io.open(os.path.abspath('./data/text_data.json'), 'r', encoding='utf-8'))

with open('./data/train.txt', 'r') as train_csv:
	reader = csv.reader(train_csv, delimiter='\t', quotechar='|')
	for row in reader:
		data.append([row[0], row[1]])

x = [i[1] for i in data] 
y = [i[0] for i in data]

x_train, x_cvtest, y_train, y_cvtest = train_test_split(x, y, test_size=0.4)
x_cv, x_test, y_cv, y_test = train_test_split(x_cvtest, y_cvtest, test_size=0.5)


word_pattern = '\\w'
if 'western_emoticones' in text_data and text_data['western_emoticones']:
	word_pattern = '(?:\\w|"' + '"|"'.join(map(lambda x: re.escape(x), text_data['western_emoticones'])) + '")'

token_pattern = '(?u)\\b' + word_pattern + word_pattern + '+\\b'

vectorizer = CountVectorizer(
	analyzer='word',
	stop_words='english',
	lowercase=True,
	min_df=0.01,
	token_pattern=token_pattern
)

X_train = vectorizer.fit_transform(x_train).toarray()
X_cv = vectorizer.transform(x_cv).toarray()
X_test = vectorizer.transform(x_test).toarray()

model = LogisticRegression(C=5)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)

print('Score : ', score)

