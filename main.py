# Library
import sys
import fitz  # Membaca file PDF
import nltk  # Natural Language Toolkit
import re  # Menghapus karakter angka.
import string  # Menghapus karakter tanda baca.
import matplotlib.pyplot as plt  # Menggambarkan Frekuensi Kemunculan
import docx  # Membaca File docx
# from docx2pdf import convert  # Meng-convert docx ke pdf
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory  # Stemming Sastrawi
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary  # Stopword Sastrawi
import os  # Membaca file di dalam folder
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

# library pyqt5 untuk GUI
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi


class ShowGUI(QMainWindow):
    def __init__(self):
        super(ShowGUI, self).__init__()
        loadUi('gui.ui', self)
        self.file = ""
        self.query = ""
        self.term_freq = {}
        self.actionOpen.triggered.connect(self.openClicked)
        self.buttonQuery.clicked.connect(self.insertQuery)
        self.buttonSearch.clicked.connect(self.showTermFreq)
        self.msg = QMessageBox()
        self.msg.setIcon(QMessageBox.Information)
        self.msg.setStandardButtons(QMessageBox.Ok)
        self.msg.setMinimumHeight(100)
        self.msg.setMinimumWidth(100)
        self.msg.setStyleSheet("QLabel{min-width: 200px; min-height: 200px;}")

    # Get Dir
    def openClicked(self):
        self.file = str(QFileDialog.getExistingDirectory(
            self, "Select Directory"))
        self.pathLabel.setText(str(self.file))
        count = 0
        # Iterate directory
        for path in os.listdir(self.file):
            # check if current path is a file
            if os.path.isfile(os.path.join(self.file, path)):
                # check if file is pdf or docx
                if path.endswith(".pdf"):
                    count += 1
                elif path.endswith(".docx"):
                    count += 1    
        self.totalFileLabel.setText(str(count))

    def insertQuery(self):
        self.query = self.queryLabel.toPlainText()
        print(self.query)
        if self.query == "":
            self.msg.about(self, "Information", "Query is empty")
        else:
            if self.file == "":
                self.msg.about(self, "Information", "File is empty")
            else:
                self.main()

    def showTermFreq(self):
        
        term = self.termLabel.toPlainText()
        if self.term_freq != {}:
            if term in self.term_freq:
                # self.termFreqLabel.setText(str(self.term_freq[term]))
                self.msg.about(self, "Information", str(self.term_freq[term]))
            else:
                self.msg.about(self, "Information", "Term not found")
        else:
            self.termFreqLabel.setText("")

    def retrieve_relevant_documents(self, folder_path, query):
        # Create a stemmer object
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        # Read the documents from the specified folder
        documents = []
        filenames = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                filenames.append(filename)
                with open(os.path.join(folder_path, filename), "rb") as f:
                    pdf_reader = PyPDF2.PdfFileReader(f)
                    num_pages = pdf_reader.getNumPages()
                    document = ""
                    for page_num in range(num_pages):
                        page = pdf_reader.getPage(page_num)
                        document += page.extractText()
                    documents.append(document)

            elif filename.endswith(".docx"):
                filenames.append(filename)
                doc = docx.Document(os.path.join(folder_path, filename))
                document = ""
                for paragraph in doc.paragraphs:
                    document += paragraph.text
                documents.append(document)

        # Preprocess the documents by stemming the words
        processed_documents = []
        for document in documents:
            # Remove all the special characters
            # Converting to Lowercase
            document = document.lower()
            document = re.sub(r'\W', ' ', str(document))
            # remove all single characters
            document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
            # Remove single characters from the start
            document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
            # Substituting multiple spaces with single space
            document = re.sub(r'\s+', ' ', document, flags=re.I)
            # Removing prefixed 'b'
            document = re.sub(r'^b\s+', '', document)
            # add stopword
            factory = StopWordRemoverFactory().get_stop_words()
            more_stopword = ['dari', 'yang', 'ke', 'di', 'dengan', 'untuk', 'pada', 'dalam', 'dan', 'atau', 'ataupun', 'adalah', 'adanya', 'adapun', 'agak', 'agaknya', 'agar', 'akan', 'akankah', 'akhir', 'akhiri', 'akhirnya', 'aku', 'akulah', 'amat', 'amatlah', 'anda', 'andalah', 'antar', 'antara', 'antaranya', 'apa', 'apaan', 'apabila', 'apakah', 'apalagi', 'apatah', 'artinya', 'asal', 'asalkan', 'atas', 'atau', 'ataukah', 'ataupun', 'awal', 'awalnya', 'bagai', 'bagaikan', 'bagaimana', 'bagaimanakah', 'bagaimanapun', 'bagainamakah', 'bagi', 'bagian', 'bahkan', 'bahwa', 'bahwasanya', 'baik', 'bakal', 'bakalan', 'balik', 'banyak']
            factory.extend(more_stopword)
            stopword = StopWordRemoverFactory().create_stop_word_remover()
            document = stopword.remove(document)
            # Stemming
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
                count = document.split().count(term)
                matrix[i, j] = document.split().count(term)
                if term in self.term_freq:
                    self.term_freq[term] += count
                else:
                    self.term_freq[term] = count

        # Print the term frequency of each term
        for term, frequency in self.term_freq.items():
            print(f"Term: {term}, Frequency: {frequency}")

        # Perform term weighting using the GVSM
        matrix = matrix / np.sqrt(np.sum(matrix ** 2, axis=1, keepdims=True))

        # Preprocess the query by stemming the words
        processed_query = processed_query.lower()
        processed_query = re.sub(r'\W', ' ', str(query))
        processed_query = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_query)
        processed_query = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_query)
        processed_query = re.sub(r'\s+', ' ', processed_query, flags=re.I)
        processed_query = re.sub(r'^b\s+', '', processed_query)
        factory = StopWordRemoverFactory().get_stop_words()
        more_stopword = ['dari', 'yang', 'ke', 'di', 'dengan', 'untuk', 'pada', 'dalam', 'dan', 'atau', 'ataupun', 'adalah', 'adanya', 'adapun', 'agak', 'agaknya', 'agar', 'akan', 'akankah', 'akhir', 'akhiri', 'akhirnya', 'aku', 'akulah', 'amat', 'amatlah', 'anda', 'andalah', 'antar', 'antara', 'antaranya', 'apa', 'apaan', 'apabila', 'apakah', 'apalagi', 'apatah', 'artinya', 'asal', 'asalkan', 'atas', 'atau', 'ataukah', 'ataupun', 'awal', 'awalnya', 'bagai', 'bagaikan', 'bagaimana', 'bagaimanakah', 'bagaimanapun', 'bagainamakah', 'bagi', 'bagian', 'bahkan', 'bahwa', 'bahwasanya', 'baik', 'bakal', 'bakalan', 'balik', 'banyak']
        factory.extend(more_stopword)
        stopword = StopWordRemoverFactory().create_stop_word_remover()
        processed_query = stopword.remove(processed_query)
        processed_query = " ".join([stemmer.stem(word) for word in query.split()])

        # Create a query vector and perform term weighting using the GVSM
        query_vector = np.zeros(len(vocab))
        for j, term in enumerate(vocab):
            query_vector[j] = processed_query.split().count(term)

        # Replace nan values with 0
        query_vector[np.isnan(query_vector)] = 0

        np.seterr(invalid='ignore')
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
        ranked_documents = sorted(
            zip(similarities, documents, filenames), reverse=True)
        return ranked_documents

    # Retrieve the most relevant documents
    def main(self):
        # Define the folder path containing the documents
        folder_path = self.file

        # Define the query
        query = self.query

        most_relevant_filenames = self.retrieve_relevant_documents(
            folder_path, query)

        print("Most relevant documents:")
        self.resultLabel.clear()
        for similarity, document, filename in most_relevant_filenames:
            # similarity = similarity.dropna()
            if np.isnan(similarity):
                similarity = 0
            similarity_percentage = round(similarity * 100)
            similarity = similarity
            self.resultLabel.append(f"Document: {filename} (similarity: {similarity}) ({similarity_percentage}%)")


app = QtWidgets.QApplication(sys.argv)
window = ShowGUI()
window.setWindowTitle('Information Retrieval')
window.show()
sys.exit(app.exec_())
