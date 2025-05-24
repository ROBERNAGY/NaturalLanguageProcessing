import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import gensim

# Sample corpus of documents
documents = [
    " Paris[a] is the capital and most populous city of France. With an official estimated population of 2,102,650 residents as of 1 January 2023[2] in an area of more than 105 km2 (41 sq mi),[5] Paris is the fourth-most populated city in the European Union and the 30th most densely populated city in the world in 2022.[6] Since the 17th century, Paris has been one of the world's major centres of finance, diplomacy, commerce, culture, fashion, and gastronomy. For its leading role in the arts and sciences, as well as its early and extensive system of street lighting, in the 19th century, it became known as the City of Light.[7] the City of Paris is the centre of the Île-de-France region, or Paris Region, with an official estimated population of 12,271,794 inhabitants on 1 January 2023, or about 19% of the population of France.[2] The Paris Region had a GDP of €765 billion (US$1.064 trillion, PPP)[8] in 2021, the highest in the European Union.[9] According to the Economist Intelligence Unit Worldwide Cost of Living Survey, in 2022, Paris was the city with the ninth-highest cost of living in the world.[10]Paris is a major railway, highway, and air-transport hub served by two international airports: Charles de Gaulle Airport (the third-busiest airport in Europe) and Orly Airport.[11][12] Opened in 1900, the city's subway system, the Paris Métro, serves 5.23 million passengers daily;[13] it is the second-busiest metro system in Europe after the Moscow Metro. Gare du Nord is the 24th-busiest railway station in the world and the busiest outside Japan, with 262 million passengers in 2015.[14] Paris has one of the most sustainable transportation systems[15] and is one of the only two cities in the world that received the Sustainable Transport Award twice.[16]",
    "C# (/ˌsiː ˈʃɑːrp/ see SHARP)[b] is a general-purpose high-level programming language supporting multiple paradigms. C# encompasses static typing,[16]: 4  strong typing, lexically scoped, imperative, declarative, functional, generic,[16]: 22  object-oriented (class-based), and component-oriented programming disciplines.[17]The C# programming language was designed by Anders Hejlsberg from Microsoft in 2000 and was later approved as an international standard by Ecma (ECMA-334) in 2002 and ISO/IEC (ISO/IEC 23270 and 20619[c]) in 2003. Microsoft introduced C# along with .NET Framework and Visual Studio, both of which were closed-source. At the time, Microsoft had no open-source products. Four years later, in 2004, a free and open-source project called Mono began, providing a cross-platform compiler and runtime environment for the C# programming language. A decade later, Microsoft released Visual Studio Code (code editor), Roslyn (compiler), and the unified .NET platform (software framework), all of which support C# and are free, open-source, and cross-platform. Mono also joined Microsoft but was not merged into .NET.",
    "German (Standard High German: Deutsch, pronounced [dɔʏ̯t͡ʃ] ⓘ)[10] is a West Germanic language in the Indo-European language family, mainly spoken in Western and Central Europe. It is the most widely spoken and official or co-official language in Germany, Austria, Switzerland, Liechtenstein, and the Italian province of South Tyrol. It is also an official language of Luxembourg and Belgium, as well as a recognized national language in Namibia. There further exist notable German-speaking communities in France (Alsace), the Czech Republic (North Bohemia), Poland (Upper Silesia), Slovakia (Košice Region, Spiš, and Hauerland), Denmark (North Schleswig), Romania and Hungary (Sopron).It is most closely related to other West Germanic languages, namely Afrikaans, Dutch, English, the Frisian languages, Scots. It also contains close similarities in vocabulary to some languages in the North Germanic group, such as Danish, Norwegian, and Swedish. Modern German gradually developed from the Old High German which in turn developed from Proto-Germanic during the Early Middle Ages. German is the second-most widely spoken Germanic and West Germanic language after English, both as a first and a second language.",
    "The dog (Canis familiaris or Canis lupus familiaris) is a domesticated descendant of the wolf. Also called the domestic dog, it is derived from extinct gray wolves, and the gray wolf is the dog's closest living relative. The dog was the first species to be domesticated by humans. Experts estimate that hunter-gatherers domesticated dogs more than 15,000 years ago, which was before the development of agriculture. Due to their long association with humans, dogs have expanded to a large number of domestic individuals and gained the ability to thrive on a starch-rich diet that would be inadequate for other canids.[4]The dog has been selectively bred over millennia for various behaviors, sensory capabilities, and physical attributes.[5] Dog breeds vary widely in shape, size, and color. They perform many roles for humans, such as hunting, herding, pulling loads, protection, assisting police and the military, companionship, therapy, and aiding disabled people. Over the millennia, dogs became uniquely adapted to human behavior, and the human–canine bond has been a topic of frequent study. This influence on human society has given them the sobriquet of mans best friend",
    "C++ (/ˈsiː plʌs plʌs/, pronounced C plus plus and sometimes abbreviated as CPP) is a high-level, general-purpose programming language created by Danish computer scientist Bjarne Stroustrup. First released in 1985 as an extension of the C programming language, it has since expanded significantly over time; as of 1997, C++ has object-oriented, generic, and functional features, in addition to facilities for low-level memory manipulation for making things like microcomputers or to make operating systems like Linux or Windows. It is almost always implemented as a compiled language, and many vendors provide C++ compilers, including the Free Software Foundation, LLVM, Microsoft, Intel, Embarcadero, Oracle, and IBM.[14]C++ was designed with systems programming and embedded, resource-constrained software and large systems in mind, with performance, efficiency, and flexibility of use as its design highlights.[15] C++ has also been found useful in many other contexts, with key strengths being software infrastructure and resource-constrained applications,[15] including desktop applications, video games, servers (e.g., e-commerce, web search, or databases), and performance-critical applications (e.g., telephone switches or space probes).[16"
]

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stop_words]
    return " ".join(tokens)

preprocessed_documents = [preprocess_text(doc) for doc in documents]


tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_documents)

lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_matrix = lda_model.fit_transform(tfidf_matrix)

feature_names = tfidf_vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda_model.components_):
    print(f"Topic {topic_idx + 1}:")
    top_features_idx = topic.argsort()[-3:][::-1]
    top_features = [feature_names[i] for i in top_features_idx]
    print(", ".join(top_features))


corpus = [doc.split() for doc in preprocessed_documents]
id2word = gensim.corpora.Dictionary(corpus)
corpus_bow = [id2word.doc2bow(doc) for doc in corpus]
lda_gensim = gensim.models.LdaModel(corpus_bow, num_topics=2, id2word=id2word, passes=10)
print(lda_gensim.print_topics())

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

num_clusters = 3

# Assuming lda_matrix contains the topic distribution matrix from LDA
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(lda_matrix)

# Add cluster labels to the preprocessed documents
clustered_documents = list(zip(preprocessed_documents, clusters))

# Separate documents by cluster
cluster_0_docs = [doc for doc, cluster in clustered_documents if cluster == 0]
cluster_1_docs = [doc for doc, cluster in clustered_documents if cluster == 1]
cluster_2_docs = [doc for doc, cluster in clustered_documents if cluster == 2]
cluster_3_docs = [doc for doc, cluster in clustered_documents if cluster == 3]
cluster_4_docs = [doc for doc, cluster in clustered_documents if cluster == 4]


# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=[0]*len(cluster_0_docs), y=cluster_0_docs, label='Cluster 0', marker='o')
sns.scatterplot(x=[1]*len(cluster_1_docs), y=cluster_1_docs, label='Cluster 1', marker='s')
sns.scatterplot(x=[2]*len(cluster_2_docs), y=cluster_2_docs, label='Cluster 2', marker='o')
sns.scatterplot(x=[3]*len(cluster_3_docs), y=cluster_3_docs, label='Cluster 3', marker='s')
sns.scatterplot(x=[4]*len(cluster_4_docs), y=cluster_4_docs, label='Cluster 4', marker='o')
plt.xlabel('Cluster')
plt.ylabel('Documents')
plt.title('Document Clustering')
plt.legend()
plt.show()

from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns

# Tokenize the preprocessed documents
top_features_idx = [doc.split() for doc in preprocessed_documents]
print(top_features_idx)
# Train Word2Vec model
word2vec_model = Word2Vec(top_features_idx, vector_size=100, window=5, min_count=1, sg=1)

# Get word vectors
word_vectors = word2vec_model.wv

# Cluster word vectors using K-means
word_vectors_array = word_vectors.vectors
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
word_clusters = kmeans.fit_predict(word_vectors_array)

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
word_vectors_pca = pca.fit_transform(word_vectors_array)

# Plot word clusters
df = pd.DataFrame(word_vectors_pca, columns=['x', 'y'])
df['word'] = word_vectors.index_to_key
df['cluster'] = word_clusters

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette='Set1', legend='full' , s=100)
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.legend(title='')
plt.show()