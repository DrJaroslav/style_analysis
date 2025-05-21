from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import requests
from bs4 import BeautifulSoup
import re

#import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

#------------------------------------------------------------------------------

def extract_html_block(link, pattern):
    options = Options()
    options.add_argument("--headless")

    driver = webdriver.Chrome(options = options)

    driver.get(link)
    html = driver.page_source

    html_block_match = re.search(pattern, html, re.DOTALL)
    html_block = html_block_match.group()

    return html_block

def extract_links(link):
    pattern = r'<ul id="sortable".*?</ul>'
    html_block = extract_html_block(link, pattern)

    pattern = r'href="(.*?)"'
    links = re.findall(pattern, html_block, re.DOTALL)

    for i, link in enumerate(links):
        links[i] = re.sub('//', 'https://', link)

    return links

def extract_poem(link):
    pattern = r'<p>(.*?)</p>'

    response = requests.get(link)
    if 200 == response.status_code:
        html_block_match = re.search(pattern, response.text, re.DOTALL)
        html_block = html_block_match.group()
        soup = BeautifulSoup(html_block, 'html.parser')
        poem = soup.get_text(separator = '\n', strip = True)
    else:
        print('extract_poem(): '
              f'Status code - {response.status_code}.')

    return poem

def extract_poems(main_link, train_file, test_file, minimum, number):
    all_links = extract_links(main_link)

    links = []
    for i in range(minimum):
        links.append(all_links[i])

    try:
        with open(train_file, 'a', encoding = 'utf-8') as train:
            try:
                with open(test_file, 'a', encoding='utf-8') as test:
                    for i, link in enumerate(links):
                        poem = extract_poem(link)
                        if i % number != 0:
                            train.write(poem)
                            train.write("\n\n")
                        else:
                            test.write(poem)
                            test.write("\n\n")
            except:
                print('extract_poems(): '
                      'Error occured while opening file.')
    except:
        print('extract_poems(): '
              'Error occured while opening train file.')

def lower_and_clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s\']', '', text)

    return text

def normalize_text(text):
    text = lower_and_clean_text(text)

    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    normalized_text = ' '.join(tokens)

    return normalized_text

def count_words(file):
    counter = 0
    try:
        with open(file, 'r', encoding = 'utf-8') as f:
            text = f.read()
            text = lower_and_clean_text(text)

            poems = text.split('\n\n')

            for poem in poems:
                words = word_tokenize(poem)
                words = [word for word in words if word.isalpha()]

                counter += len(words)
    except:
        print('count_words(): '
              'Error occured while opening file.')

    return counter

def count_minimum(main_links):
    minimum = len(extract_links(main_links[0]))

    for i in range(1, len(main_links)):
        links = extract_links(main_links[i])
        if len(links) < minimum:
            minimum = len(links)

    return minimum

def predict_author(files, names, poem_file):
    texts = []
    authors = []

    for file, name in zip(files, names):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()
                poems = text.split('\n\n')

                for poem in poems:
                    normalized_text = normalize_text(poem)
                    texts.append(normalized_text)
                    authors.append(name)
        except:
            print('predict_author(): '
                  'Error occured while opening file.')

    vectorizer = TfidfVectorizer(ngram_range = (1, 2), max_features = 3000)
    vectorized_texts = vectorizer.fit_transform(texts)

    model = MultinomialNB()
    model.fit(vectorized_texts, authors)

    try:
        with open(poem_file, 'r', encoding = 'utf-8') as f:
            poem = f.read()
            normalized_poem = normalize_text(poem)
            new_text = vectorizer.transform([normalized_poem])

            predicted_author = model.predict(new_text)
            print(f'Predicted author: {predicted_author[0]}\n')

            probas = model.predict_proba(new_text)[0]
            authors = model.classes_

            for author, prob in zip(authors, probas):
                print(f'{author}: {prob:.2%}')
    except:
        print('predict_author(): '
              'Error occured while opening file with poem.')
