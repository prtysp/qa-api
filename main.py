import json
from flask import Flask, request, flash, redirect, url_for, render_template
from langchain_community.document_loaders import (BasePDFLoader, JSONLoader)
from langchain_chroma import Chroma
from langchain_opeanai import OpenAIEmbeddings
from langchain_core.documents import Document
from pathlib import Path
from typing import Optional

app = Flask(__name__)

# Should serve the form described below, so that a user can upload a corpus first
# before asking questions.
@app.route('/')
def home():
    return render_template("index.html")

# Handles the route of uploading a corpus document using a form.
# Assumes a form as such:
# <!doctype html>
# <title> Upload </title>
# <form action="/api/load_document" method="post" enctype="multipart/form-data">
#   <input type="file" name="corpus"/>
#   <input type="radio" id="pdf" value="pdf" name="type"/> <label for="pdf">PDF</label>
#   <input type="radio" id="json" value="pdf" name="type"/> <label for="json">JSON</label>
#   <input type="file" name="questions"/>
#   <input type="submit" value="Submit"/>
# </form>
# I.E It assumes the following as input:
# 1. A file upload for the corpus.
# 2. A radio button for file type.
# 3. A file upload for the questions.
# 4. A submit buttons
# TODO: support multiple corpus files at the same time.
@app.route('/api/load_document', methods=['POST'])
def loadDocument():
    if request.method == 'POST':
        # Check if file exists
        if 'corpus' not in request.files:
            flash('No corpus file found!')
            return redirect(request.url)
        if 'questions' not in request.files:
            flash('No question file found!')
            return redirect(request.url)
        corpus = request.files['corpus']
        questions = request.files['questions']
        op = request.form['type']
        # Save the files to local.
        corpus.save(corpus.filename)
        questions.save(questions.filename)
    
    # Call functions depending on the type of corpus file.    
    if op == 'json':
        docs = loadJson(filename=corpus.filename)
    elif op == 'pdf':
        docs = loadPdf(filename=corpus.filename)
    else:
        flash('Invalid type!')
        return redirect(request.url)
    
    # Load the corpus docs into a vector store.
    split_docs = splitDocuments(docs)
    vector_store = buildVectorStore(split_docs)

    # Loads the question from the locally saved json file iterates over it to answer questions.
    qs = loadQuestions(filename=questions.filename)

    # Generate the responses from vector store and dumps them into a json file.
    rs = genResponses(vector_store, qs)
    rs_json = json.dump(rs)

    return render_template("response.html", rs_json)

# Splits documents
def splitDocuments(docs: list[Document]) -> list[Document]:
    ts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    return ts.split_documents(docs)

# Builds a Vector Store for question answering using OpenAI embeddings.
def buildVectorStore(split_docs: list[Document], persist_directory: str = "./chroma_db") -> Chroma:
    vs = Chroma.from_documents(documents=split_docs, embedding=OpenAIEmbeddings(), persist_director=persist_directory)
    return vs

# Loads a locally stored PDF file into a Document list.
def loadPdf(filename: str): -> list[Document]
    loader = BasePDFLoader(
            file_path = filename)
    docs = loader.load()

# Loads a locally stored JSON file into a Document list.
def loadJson(filename: str): -> list[Document]
    loader = JSONLoader(
            file_path = filename,
            jqschema = '.content')
    docs = loader.load()

# Loads a locally stored JSON file into a list of questions strings.
def loadQuestions(filename: str) -> list[str]:
    with open(filename) as qs_file:
        qs = json.load(qs_file)
    return qs

# Takes a vector store of knowledge and a list of questions to generate answers for each question.
def genResponses(vs: Chroma, qs: list[str]) -> dict[str, str]:
    rs = dict()
    for q in qs:
        ans = vs.similarity_search(q)
        rs[q] = ' '.join(ans)
    return rs
    
if __name__ == '__main_':
    app.run(debug=True)
