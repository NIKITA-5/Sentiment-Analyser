# Cleaning Text Steps :
# 1. Create a text file and take text from it .
# 2. Convert the letter into lowercase
# 3. Remove punctuations like .,?!

import string
from collections import Counter
import matplotlib.pyplot as plt

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# text = open('read.txt', encoding = 'utf-8').read()

# Sample text data and corresponding categories
text_data = [
    "I love programming",
    "Machine learning is fascinating",
    "Python is a versatile language",
    "Painting is a creative activity",
    "Music brings joy"
]

categories = [
    "Programming",
    "Machine Learning",
    "Python",
    "Art",
    "Music"
]

# Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text_data)

# Create a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X, categories)

# Get user input
text = input("Enter a sentence: ")

# Transform user input to numerical features using the same vectorizer
text_features = vectorizer.transform([text])

# Predict the category for the user input
predicted_category = clf.predict(text_features)

print("Predicted Category:", predicted_category[0])

lower_case = text.lower()
#print(lower_case)

# print(string.punctuation)    ..... for printing all punctuations

# str1 : specifies the list of characters that needs to be replaced.
# str2 : specifies the list of characters that it needs to be replaced with.
# str3 : specifies the list of character that need to be deleted.
# Returns : Returns the translation table which specifies the conversions that can be used to

cleaned_text = lower_case.translate(str.maketrans('','',string.punctuation))
#print(cleaned_text)

tokenized_words = cleaned_text.split()
#print(tokenized_words)

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

final_words = []
for word in tokenized_words:
    if word not in stop_words:
        final_words.append(word)

#print(final_words)

# NLP Emotion Algorithm

# 1. Check if the word in the final world list is also present in the emotion.txt
#  -- open the emotion file
#  -- Loop through each line and clear it
#  -- Extract the word and emotion using split

# 2. If word is present --> Add emotion to the emotion list

# 3. Finally count each emotion in the emotion list

emotion_list = []
with open('emotion.txt','r') as file:
    for line in file :
        clear_line = line.replace('\n','').replace(',','') .replace("'",'') .strip() # for removing spaces , and '' in between lines in emotion.txt file
       #print(clear_line)                                                                                                 # otherwise  it would have been just print(line) instead of print(clear_line)
        word, emotion = clear_line.split(':')                       # for separating words and emotion
      # print("Word:" + word + "     " + "Emotion:" + emotion)

        if word in final_words:
            emotion_list.append(emotion)
print(emotion_list)

w = Counter(emotion_list)
print(w)

# plt.bar(w.keys(), w.values())
# plt.savefig('graph.png')
# plt.show()

plt.bar(w.keys(), w.values(), color='skyblue')
plt.xlabel('Emotions')
plt.ylabel('Values')
plt.title('Sentiment Analysis')

#fig = plt.figure(facecolor='mistyrose')

plt.xticks(rotation= 30, ha='right')       # rotation of x labels

for key, value in w.items():                 # data on top of the bars
    plt.text(key, value , str(value), ha='center', fontsize=8)

plt.grid(axis='y', linestyle='--', alpha=0.7 )

plt.savefig('graph.png')
plt.show()