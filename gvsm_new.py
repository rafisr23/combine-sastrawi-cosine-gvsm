import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary

# Create a stemmer object
factory = StemmerFactory()
stemmer = factory.create_stemmer()
# Define the example documents
documents = [
    "The cat sat on the mat",
    "The dog chased the cat",
    "The rat ran across the mat"
]

# Preprocess the documents by stemming the words
processed_documents = []
for document in documents:
    processed_document = " ".join([stemmer.stem(word) for word in document.split()])
    processed_documents.append(processed_document)

# Create a vocabulary of all the unique terms in the documents
vocab = set()
for document in processed_documents:
    for word in document.split():
        vocab.add(word)

vocab = list(vocab)

# Create a matrix of document vectors, with each row representing a document and each column representing a term
matrix = np.zeros((len(processed_documents), len(vocab)))

# Calculate the term frequency of each term in each document
for i, document in enumerate(processed_documents):
    for j, term in enumerate(vocab):
        matrix[i, j] = document.split().count(term)

# Perform term weighting using the GVSM
matrix = matrix / np.sqrt(np.sum(matrix ** 2, axis=1, keepdims=True))

# Define the query and preprocess it by stemming the words
query = "cat chased"
processed_query = " ".join([stemmer.stem(word) for word in query.split()])

# Create a query vector and perform term weighting using the GVSM
query_vector = np.zeros(len(vocab))
for j, term in enumerate(vocab):
    query_vector[j] = processed_query.split().count(term)
query_vector = query_vector / np.sqrt(np.sum(query_vector ** 2))

# Calculate the cosine similarity between the query vector and each document vector
similarities = []
for i, document_vector in enumerate(matrix):
    dot_product = np.dot(query_vector, document_vector)
    query_length = np.sqrt(np.dot(query_vector, query_vector))
    doc_length = np.sqrt(np.dot(document_vector, document_vector))
    similarity = dot_product / (query_length * doc_length)
    similarities.append(similarity)

# Rank the documents by similarity and retrieve the most relevant ones
ranked_documents = sorted(zip(similarities, documents), reverse=True)
most_relevant_documents = ranked_documents[:2]

print("Most relevant documents:")
for similarity, document in most_relevant_documents:
    print(f"Document: {document} (similarity: {similarity:.2f})")