import gradio as gr
from Syncfusion.OCRProcessor import OCRProcessor
from Syncfusion.Pdf.Parsing import PdfLoadedDocument
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from PIL import Image
import os
import io
import re
import numpy as np
import boto3
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from groq import Groq

# Use environment variables for sensitive information
import os
# AWS access credentials
access_key = 'AKIAUI7N373AFR74QX5H'
secret_key = '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'

# S3 bucket details
bucket_name = 'sentinelx-prod'
prefix = 'LOTO/Documents/LOTOFormDocuments/'

GROQ_API_KEY = "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"

def extract_text_from_pdf(pdf_content):
    try:
        # Initialize the OCR processor
        processor = OCRProcessor()

        # Load the PDF document from the byte content
        stream = io.BytesIO(pdf_content)
        pdf_loaded_document = PdfLoadedDocument(stream)

        # Set the OCR language
        processor.Settings.Language = "English"

        # Perform OCR on the PDF document
        processor.PerformOCR(pdf_loaded_document)

        # Extract the text from the OCRed PDF
        text = ""
        for page in pdf_loaded_document.Pages:
            text += page.ExtractText()

        print("Extracted text from PDF: ", text[:500])
        return text
    except Exception as e:
        print("Failed to extract text from PDF:", e)
        return ""

def preprocess_text(text):
    try:
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        print("Preprocessed text: ", text[:500])
        return text
    except Exception as e:
        print("Failed to preprocess text:", e)
        return ""

def process_files(file_contents: List[bytes]):
    all_text = ""
    for file_content in file_contents:
        extracted_text = extract_text_from_pdf(file_content)
        preprocessed_text = preprocess_text(extracted_text)
        all_text += preprocessed_text + " "
    print("Combined preprocessed text: ", all_text[:500])
    return all_text

def compute_cosine_similarity_scores(query, retrieved_docs):
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_embeddings = model.encode(retrieved_docs, convert_to_tensor=True)
    cosine_scores = np.dot(doc_embeddings.cpu().numpy(), query_embedding.cpu().numpy().reshape(-1, 1))
    readable_scores = [{"doc": doc, "score": float(score)} for doc, score in zip(retrieved_docs, cosine_scores.flatten())]
    return readable_scores

def fetch_files_from_s3():
    s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    file_contents = []
    for obj in objects.get('Contents', []):
        if not obj['Key'].endswith('/'):
            response = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
            file_content = response['Body'].read()
            print(f"Fetched file {obj['Key']} with size {len(file_content)} bytes.")
            file_contents.append(file_content)
    return file_contents

def create_vector_store(all_text):
    embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(all_text)
    if not texts:
        print("No text chunks created.")
        return None

    vector_store = Chroma.from_texts(texts, embeddings, collection_name="insurance_cosine")
    print("Vector DB Successfully Created!")
    return vector_store

def load_vector_store():
    embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    try:
        db = Chroma(embedding_function=embeddings, collection_name="insurance_cosine")
        print("Vector DB Successfully Loaded!")
        return db
    except Exception as e:
        print("Failed to load Vector DB:", e)
        return None

def answer_query_with_similarity(query):
    try:
        vector_store = load_vector_store()
        if not vector_store:
            file_contents = fetch_files_from_s3()
            if not file_contents:
                print("No files fetched from S3.")
                return None

            all_text = process_files(file_contents)
            if not all_text.strip():
                print("No text extracted from documents.")
                return None

            vector_store = create_vector_store(all_text)
            if not vector_store:
                print("Failed to create Vector DB.")
                return None

        docs = vector_store.similarity_search(query)
        print(f"\n\nDocuments retrieved: {len(docs)}")

        if not docs:
            print("No documents match the query.")
            return None

        docs_content = [doc.page_content for doc in docs]
        for i, content in enumerate(docs_content, start=1):
            print(f"\nDocument {i}: {content[:500]}...")

        cosine_similarity_scores = compute_cosine_similarity_scores(query, docs_content)
        for score in cosine_similarity_scores:
            print(f"\nDocument Score: {score['score']}")

        all_docs_content = " ".join(docs_content)

        client = Groq(api_key=GROQ_API_KEY)
        template = """
                ### [INST] Instruction:
                You are an AI assistant named Goose. Your purpose is to provide accurate, relevant, and helpful information to users in a friendly, warm, and supportive manner, similar to ChatGPT. When responding to queries, please keep the following guidelines in mind:
                - When someone says hi, or small talk, only respond in a sentence.
                - Retrieve relevant information from your knowledge base to formulate accurate and informative responses.
                - Always maintain a positive, friendly, and encouraging tone in your interactions with users.
                - Strictly write crisp and clear answers, don't write unnecessary stuff.
                - Only answer the asked question, don't hallucinate or print any pre-information.
                - After providing the answer, always ask for any other help needed in the next paragraph.
                - Writing in bullet format is our top preference.
                Remember, your goal is to be a reliable, friendly, and supportive AI assistant that provides accurate information while creating a positive user experience, just like ChatGPT. Adapt your communication style to best suit each user's needs and preferences.
                ### Docs: {docs}
                ### Question: {question}
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
        return answer
    except Exception as e:
        print("An error occurred while getting the answer: ", str(e))
        return None


def process_query(query):
    try:
        response = answer_query_with_similarity(query)
        if response:
            return "Answer: " + response
        else:
            return "No answer found."
    except Exception as e:
        print("An error occurred while getting the answer: ", str(e))
        return "An error occurred: " + str(e)

# # Set up the Gradio interface
# iface = gr.Interface(
#     fn=process_query,
#     inputs=gr.Textbox(lines=7, label="Enter your question"),
#     outputs="text",
#     title="Goose AI Assistant",
#     description="Ask a question and get an answer from the AI assistant."
# )
#
# iface.launch()


process_query("Hi")