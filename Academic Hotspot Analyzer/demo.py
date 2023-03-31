import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
class AcademicHotspotAnalyzer:
    def __init__(self, domain):
        self.domain = domain
        self.articles = []
        self.keywords = []
        self.stopwords = set(stopwords.words('english'))
    def fetch_articles(self):
        url = 'https://dblp.org/search?q=' + self.domain
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('li', {'class': 'entry'})
        for result in results:
            title = result.find('span', {'class': 'title'}).text
            abstract = result.find('div', {'class': 'abstract'}).text
            self.articles.append((title, abstract))
    def analyze_keywords(self):
        for article in self.articles:
            tokens = nltk.word_tokenize(article[1])
            words = [word.lower() for word in tokens if word.isalpha() and word.lower() not in self.stopwords]
            self.keywords += words
        self.keywords = [word for word in self.keywords if len(word) > 2]
        counter = Counter(self.keywords)
        self.keywords = counter.most_common(10)
    def plot_keywords(self):
        labels = [keyword[0] for keyword in self.keywords]
        values = [keyword[1] for keyword in self.keywords]
        plt.bar(labels, values)
        plt.title('Top 10 keywords in ' + self.domain)
        plt.xlabel('Keywords')
        plt.ylabel('Frequency')
        plt.show()

analyzer = AcademicHotspotAnalyzer('computer science')
analyzer.fetch_articles()
analyzer.analyze_keywords()
analyzer.plot_keywords()
