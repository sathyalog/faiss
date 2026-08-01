from openrouter import OpenRouter
import faiss
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

def generate_embeddings(text:str):
    client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))
    response = client.embeddings.generate(
        model="openai/text-embedding-3-small",
        input=text
    )
    vectors = [item.embedding for item in response.data]
    # Under the hood faiss was written in C++ and on topped with python wrapper. C++ code expects data in low level format(float32 numbers). Hence numpy library will be used to do that conversion of python code to low level C++.
    return np.array(vectors, dtype="float32")

    # Sample documents
documents = [
    "The cat sat on the mat in the living room.",
    "A dog was resting on the rug near the fireplace.",
    "The stock market crashed sharply today.",
    "Shares fell across all major indices this afternoon.",
    "Python is a popular programming language for data science.",
    "Machine learning models require large amounts of training data.",
]

# Get embeddings from OpenAI model
embeddings = generate_embeddings(documents)

# Build faiss index using FlatL2
index = faiss.IndexFlatL2(1536)  # 1536 is the dimension of the openai/text-embedding-3-small model, if you use a different model you will find different dimensions
index.add(embeddings)
print("Vectos stored:", index.ntotal)

# search using query
query = "A pet was lying near the fire"
query_vector = generate_embeddings([query])

distances, indices = index.search(query_vector, 3)

for i in range(3):
    index = indices[0][i]
    distance = distances[0][i]
    print(f"{i+1}. {documents[index]}(distance={distance:.3f})")

