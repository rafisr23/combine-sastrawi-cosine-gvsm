# Library
import sys
import fitz  # Membaca file PDF
import nltk  # Natural Language Toolkit
import re  # Menghapus karakter angka.
import string  # Menghapus karakter tanda baca.
import matplotlib.pyplot as plt  # Menggambarkan Frekuensi Kemunculan
import docx  # Membaca File docx
# from docx2pdf import convert  # Meng-convert docx ke pdf
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory  # Stemming Sastrawi
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary  # Stopword Sastrawi
import os  # Membaca file di dalam folder
from sklearn.metrics.pairwise import cosine_similarity

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
        self.actionOpen.triggered.connect(self.openClicked)
        # self.buttonGvsm.clicked.connect(self.main)
        self.buttonQuery.clicked.connect(self.insertQuery)
        self.msg = QMessageBox()
        self.msg.setIcon(QMessageBox.Information)
        self.msg.setStandardButtons(QMessageBox.Ok)
        self.msg.setMinimumHeight(100)
        self.msg.setMinimumWidth(100)
        self.msg.setStyleSheet("QLabel{min-width: 200px; min-height: 200px;}")

    # Get Dir
    def openClicked(self):
        self.file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.pathLabel.setText(str(self.file))
        count = 0
        # Iterate directory
        for path in os.listdir(self.file):
            # check if current path is a file
            if os.path.isfile(os.path.join(self.file, path)):
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

    # Input File - DOCX
    def getDOCX(self, filename):
        doc = docx.Document(filename)
        fullText = []
        for para in doc.paragraphs:
            fullText.append(para.text)
        return '\n'.join(fullText)

    # Input File - PDF
    def getPDF(self, filename):
        with fitz.open(filename) as doc:  # Membuka File PDF
            kalimat = ""
            for page in doc:
                kalimat += page.get_text()  # Memasukkan Kata ke variabel Kalimat
        return kalimat

    # Proses Stemming
    def getStemming(self, kalimat):
        # Merubah format teks menjadi format huruf kecil semua (lowercase).
        lower_case = kalimat.lower()

        # Menghapus karakter angka.
        remove_number = re.sub(r"\d+", "", lower_case)

        # Menghapus karakter tanda baca.
        remove_punctuation = remove_number.translate(str.maketrans("", "", string.punctuation))

        # Menghapus karakter kosong.
        removing_whitespace = remove_punctuation.strip()

        # Menggunakan library NLTK untuk memisahkan kata dalam sebuah kalimat.
        tokens = nltk.tokenize.word_tokenize(removing_whitespace)

        hasil_tokens = ''
        for x in tokens:
            hasil_tokens += ' ' + x  # Memasukkan kata yang sudah dipisah (list), disatukan kembali menjadi string

        # Menghapus kata Stopword - Sastrawi
        stop_factory = StopWordRemoverFactory().get_stop_words()  # load default stopword
        more_stopword = ['vol', 'volume', 'issn', 'php', 'mysql', 'gunadarma']  # menambahkan stopword

        data = stop_factory + more_stopword  # menggabungkan stopword

        dictionary = ArrayDictionary(data)
        str_stopword = StopWordRemover(dictionary)
        hasil_stopword = nltk.tokenize.word_tokenize(str_stopword.remove(hasil_tokens))

        hasil_stopword_fix = ''
        for x in hasil_stopword:
            hasil_stopword_fix += ' ' + x  # Memasukkan kata yang sudah dipisah (list), disatukan kembali menjadi string

        # Stemming Sastrawi
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        hasil = stemmer.stem(hasil_stopword_fix)

        return hasil

    # Membuat Value String Menjadi List
    def convert(self, value):
        li = list(value.split(" "))
        return li

    # Memproses File Inputan
    def proses(self, file):
        doc = file
        if doc.endswith('.pdf'):
            hasil = self.getPDF(doc)
            hasil_stemming = self.getStemming(hasil)
            hasil_fix = self.convert(hasil_stemming)
            kemunculan = nltk.FreqDist(self.convert(hasil_stemming))
            # print('\nStemming hasil -> ' + str(doc) + ':')
            # print(kemunculan.most_common())

        elif doc.endswith('.docx'):
            hasil = self.getDOCX(doc)
            hasil_stemming = self.getStemming(hasil)
            hasil_fix = self.convert(hasil_stemming)
            kemunculan = nltk.FreqDist(self.convert(hasil_stemming))
            # print('\nStemming hasil -> ' + str(doc) + ':')
            # print(kemunculan.most_common())

        else:
            print('Harap Masukkan File PDF atau DOCX')

        return hasil_fix

    # Membuat Term-Document Matrix
    def term_document_matrix(self, list_of_documents):
        all_terms = []
        for doc in list_of_documents:
            all_terms += doc
        # Menghapus kembar
        terms = set(all_terms)
        # print("Terms ->" + str(terms))

        # Membuat Dictionaries
        dict_list = []
        for term in terms:
            temp = []
            for doc in list_of_documents:
                if term in doc:
                    temp.append(1)
                else:
                    temp.append(0)
            dict_list.append(temp)
        # print("dict list ->" + str(dict_list))
        return terms, dict_list

    # Memproses Query
    def proses_query(self, query):
        query_stemming = self.getStemming(query)

        return self.convert(query_stemming)

    # Membuat Vector Query
    def vector_query(self, query, terms):
        temp = []
        for term in terms:
            if term in query:
                temp.append(1)
            else:
                temp.append(0)

        return temp

    # Menghitung Similarity
    def similarity(self, vector_query, dict_list):
        # Hitung cosine similarity antara vector query dan setiap vektor dalam dict_list
        sim_scores = cosine_similarity([vector_query], dict_list)[0]

        return sim_scores

    # Menampilkan Hasil Akhir
    def show_result(self, list_of_documents, results):
        for i in range(len(results)):
            if results[i] > 0:
                print("Document ke-" + str(i+1) + " memiliki similarity sebesar: " + str(results[i]))
            else:
                print("Document ke-" + str(i+1) + " tidak memiliki similarity.")

    # Main Program
    def main(self):
        # Membaca file di dalam folder
        path = self.file
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                if '.docx' in file:
                    files.append(os.path.join(r, file))
                elif '.pdf' in file:
                    files.append(os.path.join(r, file))

        # Memproses File
        list_of_documents = []
        for doc in files:
            list_of_documents.append(self.proses(doc))

        # Membuat Term-Document Matrix
        terms, dict_list = self.term_document_matrix(list_of_documents)

        # Memasukkan Query
        query = self.query

        # Memproses Query
        query = self.proses_query(query)

        # Membuat Vector Query
        vector_query_fix = self.vector_query(query, terms)

        # Menghitung Similarity
        results = self.similarity(vector_query_fix, dict_list)

        # Menampilkan Hasil Akhir
        self.show_result(list_of_documents, results)

app = QtWidgets.QApplication(sys.argv)
window = ShowGUI()
window.setWindowTitle('Information Retrieval')
window.show()
sys.exit(app.exec_())

#main()


# # Jaccard - Similarity
# def measure(a,b):
#     doc1 = set(a)
#     doc2 = set(b)
#     intersection = doc1.intersection(doc2)
#     union = doc1.union(doc2)
#     hasil = float(len(intersection))/len(union)
#     print('Hasil Similarity menggunakan Jaccard: ' + str(hasil))
#     # print(hasil)
#     # print('\n')
#
#     return hasil
#     # return len(union)

# Main Proses
# query = input(str('Masukkan Query Yang Diinginkan : '))
# stemming_query = getStemming(query)
# list_query = convert(stemming_query)
# print('Query -> ' + str(list_query))
#
# # Main Proses - Read File from Directory
# path = 'D:\Materi Kuliah\Semester 5\IFB-307 DATA MINING DAN INFORMATION RETRIEVAL\Code\Jaccard Similarity\similarity index\daftar-file'
# files = os.listdir(path)
# fileAfterProcess = []
# checkFile = []
# for index, file in enumerate(files):
#     print('\n' + file)
#     fileAfterProcess.append(proses(path + "/" + file))
#     # checkFile.append(measure(list_query, fileAfterProcess[index]))
#
# print('\nUrutan Hasil Similarity: ' + str(sorted(checkFile, reverse=True)))
# print('*semakin mendekati 1, semakin baik hasil similarity-nya')
