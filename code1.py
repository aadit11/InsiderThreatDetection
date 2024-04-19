from openai import AzureOpenAI
from langchain.vectorstores import FAISS
#from langchain.embeddings.openai import OpenAIEmbeddings  
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA  # Update as needed
import PyPDF2
import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI
import psycopg2
from psycopg2 import sql

import os




client = AzureOpenAI(
    api_version=GPT_4_API_VERSION,
    azure_endpoint=GPT_4_API_BASE,
    
)

def ask_model(prompt):
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt},
        ],
        model=GPT_4_ID  
    )
    return response


pdf_file_obj = open("data\Aadit(Code3).pdf", "rb")
pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
num_pages = len(pdf_reader.pages)
detected_text = ""

for page_num in range(num_pages):
    page_obj = pdf_reader.pages[page_num]
    detected_text += page_obj.extract_text() + "\n\n"

pdf_file_obj.close()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.create_documents([detected_text])

directory = "index_store"
vector_index = FAISS.from_documents(texts, AzureOpenAIEmbeddings(
    azure_endpoint = GPT_4_API_BASE,
    deployment=EmbeddingModelDeploymentName,
    ))  
vector_index.save_local(directory)

vector_index = FAISS.load_local("index_store", AzureOpenAIEmbeddings(
    azure_endpoint = GPT_4_API_BASE,
    deployment=EmbeddingModelDeploymentName))  


retriever = vector_index.as_retriever(search_type="similarity", search_kwargs={"k": 6})



qa_interface = RetrievalQA.from_chain_type(
    llm=AzureChatOpenAI(openai_api_key=GPT_4_API_KEY,api_version=GPT_4_API_VERSION,base_url=GPT_4_API_BASE,azure_deployment=GPT_4_ID),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# prompt = """
# Analyze the attached Board Resolution document and provide a JSON structure listing the approval matrix, if present. Focus on identifying transaction limits, approval authorities, and specific conditions for financial operations like account operations, internet banking, etc. If the document does not contain this information or sections are empty, include this in the JSON structure. Ensure the JSON captures the approval process details or the absence of such data, adapting to the document's format and content.
# "Important for output" :  start the output with the opening bracket of the json and finish with the closing bracket. Do not print anything before or after the brackets as I want to process the output later.
# Critical: The standard format of json I require
#           {
#                 "[table heading](Generate accoding to table context. please dont include any special characters)":
#                 [{
#                 "column_name" : ["column_value"],
#                  all the rest of the values of that row
#                 },
#                 {
#                 Same process to be repeated for every row in that table
#                 }],         
#           }
# Very Important: 
#     a) The same format should be followed for all values.

# Critical:
#     a) There might me some data or table whose officials dont come under any particular category, in this case i want you to make a separeate table called "Other Banking Facilities" and add those officials in that table and add it at the end of the json.
#     b) The name of the official had to be present in every table.
#     c) Make sure to print the tables in the exact same order as they appear in the document starting from the first page.
# """    





# prompt1 = """
# Analyze the attached Board Resolution document and provide a JSON structure listing only the authorized signatories along with their functions. If sections related to banking operations like account opening, account operations, internet banking, etc., are present, extract names and designations of the signatories. If no data is found in these sections, include this in the JSON structure. Omit transaction limits or approval processes. Adapt the JSON to the document's content and format, capturing specific roles and names as presented or noting the absence of data.
# "Important for output" : I require the ouput as the json version of the table inside the <json></json> tags.
# Critical: The standard format of json I require
#           {
#                 "sectionname_tablename(please dont include any special characters except '_')":
#                 [{
#                 "column_name" : ["column_value"],
#                  all the rest of the values of that row
#                 },
#                 {
#                 Same process to be repeated for every row in that table
#                 }],         
#           }
# Very Important: 
#     a) The same format should be followed for all values.
#     b) Do not include the "approved by" or "approval matrix" as it is different from
# """ 



prompt2 = """
I want you to study the pdf thoroughly.
"Important for output" : If you are able to detect any code present of any language, i want your response
to be "Insider Threat Detected" and also the explanation of that code, if no code is present in the file, I want your response to be "No threat Detected".
"Critical for output ": I also want you to compute the level of insider threat based on the length of the code, 
its a fatal threat if the code is more than 50 lines, and its a moderate threat if its less than 50 lines.
So along with the detection of the threat, also print the level of the threat.
"""       

# response = qa_interface(
#     prompt
# )

# response1 = qa_interface(
#     prompt1
# )

response2 = qa_interface(
    prompt2
)

# print(response["result"])
# print(response1["result"])
print(response2["result"])


