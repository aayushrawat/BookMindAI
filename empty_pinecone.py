import pinecone
from cnfg import PINECONE_API_KEY, PINECONE_API_ENV

pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV
    )
index_name = "bookimind-ai"
index = pinecone.Index(index_name=index_name)

index.delete(delete_all=True)