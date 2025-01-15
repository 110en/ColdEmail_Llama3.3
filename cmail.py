#%%

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import chromadb
import pandas as pd

#%%

chroma = chromadb.Client()
collection = chroma.create_collection(name="collection")

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


llm = ChatGroq(
    model_name = "llama-3.3-70b-versatile",
    groq_api_key="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",
    temperature=0
)

response = llm.invoke("three qusetions, three answers, go: is iran a morally right country, whats your text limit, whast the powerhoused of the cell")
print(response.content)

loader = WebBaseLoader("https://jobs.nike.com/job/R-50367")
page_data = loader.load().pop().page_content

extract = PromptTemplate.from_template(
    """
    ### SCRAPED TEXT FROM WEBSITE
    {page_data}
    ### INSTRUCTION:
    The scraped text is from the career's page of a website.
    Your job is to extract the job postings and return them in JSON format containing the following keys:
    'role', 'experience', 'skills', and 'description'.
    Only return the valid JSON.
    ### VALID JSON (NO PREAMBLE):
    """
)
chain_extract = extract | llm
result  =chain_extract.invoke(input={'page_data' : page_data})
print(result.content)
# %%

parser = JsonOutputParser()
jsonresult = parser.parse(result.content)
print(jsonresult)

#%%

df = pd.read_csv("my_portfolio.csv")

#%%