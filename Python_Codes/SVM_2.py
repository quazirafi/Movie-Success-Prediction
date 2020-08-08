corpus = []
corpus.append("My name is Rafi")
corpus.append("Their name is dana")
corpus.append("They are same same name")
corpus.append("WHO the hell are you")
corpus.append("Why do you need to see me?")
corpus.append("I am so happy for u")
corpus.append("I don't know how long You are just stupid.It's bad really bad")
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word', min_df = 0, stop_words = 'english')
tfidf_matrix = tf.fit_transform(corpus)
print(tfidf_matrix.shape)
print(tfidf_matrix[2])
print(tf.get_feature_names())