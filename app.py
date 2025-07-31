
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def preprocess_text(text):
    text = re.sub(r"[^��-����-�R\s]", "", str(text))
    return text

def get_keywords(df, column='������ �ǰ�', stopwords=[]):
    texts = df[column].dropna().apply(preprocess_text)
    vectorizer = CountVectorizer(stop_words=stopwords)
    matrix = vectorizer.fit_transform(texts)
    keywords = vectorizer.get_feature_names_out()
    freq = matrix.sum(axis=0).A1
    return pd.DataFrame({'Ű����': keywords, '��': freq}).sort_values(by='��', ascending=False)

def draw_wordcloud(df, title="����Ŭ����"):
    word_freq = dict(zip(df['Ű����'], df['��']))
    wc = WordCloud(font_path='/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                   background_color='white', width=800, height=400).generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    st.pyplot(plt)
