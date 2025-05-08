from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv
import os

# 1. Load .env for GROQ_API_KEY
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 2. Load vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vector_db", embeddings=embedding_model, allow_dangerous_deserialization=True)

# 3. Accept user query and search similar chunks
def query_vectorstore(user_query: str, k: int = 5):
    results = vectorstore.similarity_search_with_score(user_query, k=k)
    return [doc.page_content for doc, score in results]

# 4. Format prompt with context and question
def build_prompt(context_chunks, question):
    context_text = "\n\n".join(context_chunks)
    prompt = f"""
You are an expert AI tutor. Use the following context to answer the user's question. Be clear and concise.

Context:
{context_text}

Question: {question}
"""
    return prompt.strip()

# 5. Send prompt to Groq model
def get_groq_response(prompt):
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content
