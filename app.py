import json
import re
import uuid
import requests
from typing import List
from flask import Flask, jsonify, request
# from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import boto3
import config
import fitz
app = Flask(__name__)

# Ensure the Tesseract OCR path is set correctly
#pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

#with open(r"config.json", 'r') as config_file:
  #  configurations = json.load(config_file)
#GROQ_API_KEY = configurations['groq']


def preprocess_text(text):
    """
    Function to Preprocess Raw text extracted from PDFs
    :param text: Raw extracted Text
    :return: Preprocessed Text
    """
    try:
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    except Exception as e:
        print("Failed to preprocess text:", e)
        return ""


# def fetch_text_file_from_huggingface_space():
#     url = "https://huggingface.co/spaces/Luciferalive/goosev9/blob/main/extracted_text.txt"
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         text_content = response.text
#         print("Successfully downloaded the text file")
#         return text_content
#     except Exception as e:
#         print(f"Failed to download the text file: {e}")
#         return ""


def list_files_in_folder(bucket_name, prefix):
    """
    List all files in a given S3 bucket folder.

    :param bucket_name: str. Name of the S3 bucket.
    :param prefix: str. Folder path in the S3 bucket.
    :return: list. List of file paths.
    """
    files = []
    s3 = boto3.client('s3', aws_access_key_id=config.access_key, aws_secret_access_key=config.secret_key)
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    while response:
        for content in response.get('Contents', []):
            files.append(content['Key'])
        if response.get('IsTruncated'):  # Check if there are more keys to retrieve
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix,
                                          ContinuationToken=response['NextContinuationToken'])
        else:
            break

        return files

def fetch_files_from_s3(prefix, bucket_name):
    """
    Fetch files from an S3 bucket.
    :param prefix: Bucket Prefix
    :param bucket_name: Name of Bucket
    :return: All files list to fetch data from
    """
    s3 = boto3.client('s3', aws_access_key_id=config.access_key, aws_secret_access_key=config.secret_key)

    all_files = []
    for folder in prefix:
        all_files.extend(list_files_in_folder(bucket_name, folder))

    file_contents = []
    for obj in all_files:
        response = s3.get_object(Bucket=bucket_name, Key=obj)
        file_content = response['Body'].read()
        # print(f"Fetched file {obj} with size {len(file_content)} bytes.")
        file_contents.append(file_content)
    print(f"Total No of Files Fetched {len(file_contents)}")
    return file_contents

def create_vector_store(text_content, client_id):
    """
    Function to create vector store from extracted text
    :param text_content: content to create vector DB From
    :param client_id: Unique Client ID
    :return: Vector DB or None
    """
    embeddings = SentenceTransformerEmbeddings(model_name=config.MODEL_NAME)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text_content)
    if not texts:
        print("No text chunks created")
        return None

    collection_name = "vector_db_" + str(client_id)
    vector_store = Chroma.from_texts(texts, embeddings, collection_name=collection_name, persist_directory="./goose_vec_db")
    print("Vector DB Successfully Created!")
    return vector_store


def extract_text_from_pdf(pdf_content):
    """
    Extract text from PDF content.
    :param pdf_content:
    :return: Extracted PDF Text
    """
    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
            print("Text Extracted from PDF:", text[100])
            # pix = page.get_pixmap()
            # img = Image.open(io.BytesIO(pix.tobytes()))
            # text += pytesseract.image_to_string(img)
        print("Extracted text from PDF: ", text[:500])  # Print the first 500 characters
        return text

    except Exception as e:
        print("Failed to extract text from PDF:", e)
        return ""


def load_vector_store(client_id):
    """
    Function to load vector store if exists
    :param client_id:Unique Client ID to get Vector Store
    :return: Vector DB or None
    """

    embeddings = SentenceTransformerEmbeddings(model_name=config.MODEL_NAME)
    try:
        if client_id:
            collection_name = "vector_db_" + str(client_id)
            db = Chroma(embedding_function=embeddings, collection_name=collection_name,
                        persist_directory="./goose_vec_db")
            print("Vector DB Successfully Loaded!")
            return db
        else:
            return None
    except Exception as e:
        print("Failed to load Vector DB:", e)
        return None


def process_files(file_contents: List[bytes]):
    """
    Process and combine text from multiple files.
    :param file_contents: Extracted Content from PDF
    :return: Preprocessed Extracted Text
    """

    all_text = ""
    for file_content in file_contents:
        extracted_text = extract_text_from_pdf(file_content)
        preprocessed_text = preprocess_text(extracted_text)
        all_text += preprocessed_text + " "
        print("Combined preprocessed text: ", all_text[:500])  # Print the first 500 characters
        return all_text


def extract_text_from_pdf(pdf_content):
    """Extract text from PDF content using OCR."""
    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
            print(text)
            # pix = page.get_pixmap()
            # img = Image.open(io.BytesIO(pix.tobytes()))
            # text += pytesseract.image_to_string(img)
        print("Extracted text from PDF: ", text[:500])  # Print the first 500 characters
        return text
    except Exception as e:
        print("Failed to extract text from PDF:", e)
        return ""


# noinspection PyPackageRequirements
def get_file_contents(prefix, bucket_name):
    """

    :param prefix:
    :param bucket_name:
    :return:
    """
    file_contents = fetch_files_from_s3(prefix, bucket_name)
    if not file_contents:
        print("No files fetched from S3.")
        return None

    all_text = process_files(file_contents)
    if not all_text.strip():
        print("No text extracted from documents.")
    return all_text

def answer_query(query, client_id):
    """

    :param query:
    :param client_id:
        :return:
        """
    try:
        vector_store = load_vector_store(client_id)
        if not vector_store:
            return {"error": "Vector DB Does not exist"}
            # all_text = get_file_contents(prefix, bucket_name)
            # if all_text:
            #     vector_store = create_vector_store(all_text, client_id)
            # else:
            #     return None
        docs = vector_store.similarity_search(query)
        print(f"\n\nDocuments retrieved: {len(docs)}")

        if not docs:
            print("No documents match the query.")
            return {"error": "No documents match the query"}

        docs_content = [doc.page_content for doc in docs]
        all_docs_content = " ".join(docs_content)

        client = Groq(api_key=config.GROQ_API_KEY)
        template = """
                            ### [INST] Instruction:
                            You are an AI assistant named Goose. Your purpose is to provide accurate, relevant, and helpful information to users in a friendly, warm, and supportiv>                    - Retrieve relevant information from your knowledge base to formulate accurate and informative responses.
                            - Always maintain a positive, friendly, and encouraging tone in your interactions with users.
                            - Strictly write crisp and clear answers, don't write unnecessary stuff.
                            - Only answer the asked question, don't hallucinate or print any pre-information.
                            - After providing the answer, always ask for any other help needed in the next paragraph.
                            - Writing in bullet format is our top preference.
                            Remember, your goal is to be a reliable, friendly, and supportive AI assistant that provides accurate information while creating a positive user experi>                    ### Question: {question}
                            """

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": template.format(docs=all_docs_content, question=query)
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            model="llama3-8b-8192",
        )
        answer = chat_completion.choices[0].message.content.strip()
        return {"answer": answer}
    except Exception as e:
        print("An error occurred while getting the answer: ", str(e))
        return None


@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        # prefix = None
        # bucket_name = None
        # doc_path = None
        query = request.json.get('query')
        session_id = request.json.get('session_id')
        client_id = request.json.get('client_id')
        if not query:
            return jsonify({"error": "No query provided"}), 400
        if not session_id:
            session_id = uuid.uuid4()
        if not client_id:
            return jsonify({"error": "Client Id not provided"}), 400
        # if not (client_doc_path and bucket_name):
        #     prefix = client_doc_path
        #     vector_store = load_vector_store(client_id)
        #     if not vector_store:
        #         return jsonify(
        #             {"error": "Client Doc path or Bucket Name not provided as Vector Store does not exist"}), 400
        # bucket_name = config.bucket_name
        # prefix = config.prefix
        # else:
        #     bucket_name = config.bucket_name
        #     prefix = client_doc_path
        #     print("Client Doc path received:", client_doc_path)
        #     print("Type of Prefix:", type(prefix))

        response = answer_query(query, client_id)
        if response.get('error'):
            return jsonify({"error": response['error'], 'session_id': session_id}), 404
        if response:
            return jsonify({"answer": response['answer'], 'session_id': session_id}), 200
        else:
            return jsonify({"error": "No answer found", 'session_id': session_id}), 404

    except Exception as e:
        print("An error occurred while processing the request: ", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/loadVectorDb', methods=['POST'])
def vector_db():
    try:
        client_doc_path = request.json.get('client_doc_path')
        client_id = request.json.get('client_id')
        bucket_name = request.json.get('bucket_name')
        if client_id and client_doc_path and bucket_name:
            # bucket_name = config.bucket_name
            prefix = client_doc_path
            print("Client Doc path received:", client_doc_path)
            print("Type of Prefix:", type(prefix))
            all_text = get_file_contents(prefix, bucket_name)
            if all_text:
                response = create_vector_store(all_text, client_id)
                if response:
                    return jsonify({"response": "Vector DB Successfully Created"}), 200
                else:
                    return jsonify({"response": "Vector DB Not Created Successfully"}), 404
            else:
                return jsonify({"error": "No Data found in Docs"}), 400
        else:
            return jsonify({"error": "Client Id or Client Doc path or Bucket Name not provided"}), 400

    except Exception as e:
        print("An error occurred while processing the request: ", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)
