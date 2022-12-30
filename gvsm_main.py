# Library
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

# Input File - DOCX
def getDOCX(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)

# Input File - PDF
def getPDF(filename):
    with fitz.open(filename) as doc:  # Membuka File PDF
        kalimat = ""
        for page in doc:
            kalimat += page.get_text()  # Memasukkan Kata ke variabel Kalimat
    return kalimat

# Proses Stemming
def getStemming(kalimat):
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
def convert(value):
    li = list(value.split(" "))
    return li

# Memproses File Inputan
def proses(file):
    doc = file
    if doc.endswith('.pdf'):
        hasil = getPDF(doc)
        hasil_stemming = getStemming(hasil)
        hasil_fix = convert(hasil_stemming)
        kemunculan = nltk.FreqDist(convert(hasil_stemming))
        # print('\nStemming hasil -> ' + str(doc) + ':')
        # print(kemunculan.most_common())

    elif doc.endswith('.docx'):
        hasil = getDOCX(doc)
        hasil_stemming = getStemming(hasil)
        hasil_fix = convert(hasil_stemming)
        kemunculan = nltk.FreqDist(convert(hasil_stemming))
        # print('\nStemming hasil -> ' + str(doc) + ':')
        # print(kemunculan.most_common())

    else:
        print('Harap Masukkan File PDF atau DOCX')

    return hasil_fix

# Membuat Term-Document Matrix
def term_document_matrix(list_of_documents):
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
def proses_query(query):
    query_stemming = getStemming(query)

    return convert(query_stemming)

# Membuat Vector Query
def vector_query(query, terms):
    temp = []
    for term in terms:
        if term in query:
            temp.append(1)
        else:
            temp.append(0)

    return temp

# Menghitung Similarity
def similarity(vector_query, dict_list):
    pembagi = (sum([i ** 2 for i in vector_query])) ** 0.5
    temp = []
    for i in range(len(dict_list[0])):
        atas = sum([vector_query[x] * dict_list[x][i] for x in range(len(dict_list))])
        bawah = ((sum([i ** 2 for i in dict_list[i]])) ** 0.5) * pembagi
        temp.append(atas/bawah)

    return temp

# Menampilkan Hasil Akhir
def show_result(list_of_documents, results):
    for i in range(len(results)):
        if results[i] > 0:
            print("Document ke-" + str(i+1) + " memiliki similarity sebesar: " + str(results[i]))
        else:
            print("Document ke-" + str(i+1) + " tidak memiliki similarity.")

# Main Program
def main():
    # Membaca file di dalam folder
    path = 'D:\Materi Kuliah\Semester 5\IFB-307 DATA MINING DAN INFORMATION RETRIEVAL\Code\Jaccard Similarity\similarity index\daftar-file'
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
        list_of_documents.append(proses(doc))

    # Membuat Term-Document Matrix
    terms, dict_list = term_document_matrix(list_of_documents)

    # Memasukkan Query
    query = input("Masukkan query: ")

    # Memproses Query
    query = proses_query(query)

    # Membuat Vector Query
    vector_query_fix = vector_query(query, terms)

    # Menghitung Similarity
    results = similarity(vector_query_fix, dict_list)

    # Menampilkan Hasil Akhir
    show_result(list_of_documents, results)

main()


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
