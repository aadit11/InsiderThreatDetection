from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import AzureChatOpenAI

def main():
    # Path to the CSV file
    csv_file_path = "data\employee_data (1).csv"

    # Load data from CSV
    loader = CSVLoader(file_path=csv_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()

    # Text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)

    # Sentence embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Convert text chunks into embeddings and save them into FAISS knowledge base
    docsearch = FAISS.from_documents(text_chunks, embeddings)

    # Load Language Model
    llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q4_0.bin",
                        model_type="llama",
                        max_new_tokens=1024,
                        temperature=0)

    # Query the CSV with the LLM
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())

    # Example prompts
    prompts = [
        "What is the business unit with the most employees?",
        "What is the department with the most employees",
        "Give me names of any 5 Area Sales Managers?"
    ]

    # Get answers for prompts
    for prompt in prompts:
        result = qa({'chat_history': '', 'question': prompt})
        print(f"Question: {prompt}")
        print(f"Answer: {result['answer']}")

if __name__ == '__main__':
    main()
