#%%

from langchain_groq import ChatGroq

import chromadb

#%%

llm = ChatGroq(
    model_name = "llama-3.3-70b-versatile",
    groq_api_key="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",
    temperature=0
)
chroma = chromadb.Client()
collection = chroma.create_collection(name="collection")

#%%

collection.add(
    documents = ["This document is about me", "This document is about him", "This document is about her", "This document is about you"],
    ids = ["id1", "id2", "id3", "id4"]
)

docs = collection.get()
doc1 = collection.get(ids=["id1"])
doc2and3 = collection.get(ids=["id2", "id3"])
print(doc1)
#collection.delete(ids=all_docs["ids"])

collection.query(
    #is doing a semantic search
    query_texts=["Query is about different ways to talk about a girl"],
    n_results=2
)

#%%

response = llm.invoke("three qusetions, three answers, go: is iran a morally right country, whats your text limit, whast the powerhoused of the cell")
print(response.content)
# %%
