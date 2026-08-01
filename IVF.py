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

# sqrt(N) for real datasets in real time
# Build faiss index using IVF
nlist=2 # number of clusters we wanted to create
quantizer=faiss.IndexFlatL2(1536) # this is our main index and its only helper which cluster centre is closest to the query vector
index = faiss.IndexIVFFlat(quantizer, 1536, nlist) # this is our main index

# IVF needs this extra training step before adding vectors
index.train(embeddings)

#Store embeddings in index
index.add(embeddings)

# how many clusters you want to search(higher number more accurate but performance will be slower)
index.nprobe = 2

print("Vectos stored:", index.ntotal)


# search using query
query = "A pet was lying near the fire"
query_vector = generate_embeddings([query])

distances, indices = index.search(query_vector, 3)

for i in range(3):
    index = indices[0][i]
    distance = distances[0][i]
    print(f"{i+1}. {documents[index]}(distance={distance:.3f})")

