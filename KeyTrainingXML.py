from lxml import etree
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string

xml_file = "xml_file.xml"
# Punctuation and stop-words which we must remove from text
# I use ['ha', 'wa', 'u', 'a'] as a stop words because lemmatization converts words has, was, us and as in incorrect way
# So result can be wrong
stop = stopwords.words('english') + list(string.punctuation) + ['ha', 'wa', 'u', 'a']
lemmatizer = WordNetLemmatizer()

# Read the xml file
root = etree.parse(xml_file).getroot()
elements = root[0]

headers = []
content =[]

#  Empty list for clean texts for vectorizing below
dataset = []

for el in elements:

    # Extract the headers and the text.
    headers.append(el[0].text)
    content.append(el[1].text)

for el in content:
    # Rewrite the text in lowercase letters
    text = el.lower()

    # Tokenize each text
    text = word_tokenize(text)

    # Lemmatize EACH word in the tokenize text.
    for idx, word in enumerate(text):
        word = lemmatizer.lemmatize(word)
        text[idx] = word

    # Get rid of punctuation, stopwords.
    text = [word for word in text if word not in stop]

    # Get rid not-noun words
    text = [nltk.pos_tag([word]) for word in text]
    text = [el[0][0] for el in text if el[0][1] == 'NN']

    # Add "noun only" text as a string to list of strings for vectorizing below
    dataset.append(' '.join(text))


# Count the TF-IDF metric for each word in all stories.
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(dataset)

# Create the vocabulary
terms = vectorizer.get_feature_names()

for i in range(len(headers)):
    # Print each header
    print(f'{headers[i]}:')

    # Encode the text data for machine learning with Pandas library
    df = pd.DataFrame(tfidf_matrix[i].toarray())

    # Transpose index and columns (reflect the DataFrame)
    # Sort data by values in descending order
    # Reset the index
    df2 = df.transpose().sort_values(by=0, ascending=False).reset_index()

    # Empty list of sorted words (by score)
    sorted_score_words = []
    for i in range(0, len(df2)):
        # Find the top rated word in vocabulary by index
        top_rated_word = terms[df2['index'][i]]
        sorted_score_words.append(top_rated_word)

    # Create new column of words in DataFrame
    df2['words'] = sorted_score_words
    # Sorting in order and alphabetically in case of a conflict of scores (in descending order)
    df2 = df2.sort_values(by=[0, 'words'], ascending=[False, False])

    string_to_print = ''
    for n in range(0, 5):
        word = df2.iloc[n]['words']
        string_to_print += word + ' '
    print(string_to_print + '\n')
