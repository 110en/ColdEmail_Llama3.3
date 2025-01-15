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

df = pd.read_csv("testing/my_portfolio.csv")
print(df)


x = 0
client = chromadb.PersistentClient('vectorstore')
collect = client.get_or_create_collection(name='portfolio')
if not collect.count():
    for _, row in df.iterrows():
        collect.add(
        documents = row["Techstack"],
        metadatas = {"links" : row["Links"]},
        ids = ["id" + str(x)]
        )
        x += 1

print(
    collect.query(
        query_texts=["Experience in Python", "Expertise in React"],
        n_results = 2
    ).get("metadatas")
)

links = collect.query(
            query_texts=jsonresult["skills"],
            n_results = 2
        ).get("metadatas", [])

#%%

cmail = PromptTemplate.from_template(
    """
    ### JOB DESCRIPTION:
    {page_data}

    ### INSTRUCTION:
    You are JimBob, a business development executive at OpenAI. your job is to write a cold email to the client regarding the job 
    mentioned above describing the capability of OpenAI in fulfilling their needs.
    Also add the most relevant ones from the following links to showcase OpenAI's portfolio: {link_list}
    Remember, you are JimBob, BDE at OpenAI.

    ### EMAIL (NO PREAMBLE):
    """
)

chain_email = cmail | llm
emailresult = chain_email.invoke({"page_data" : page_data, "link_list" : links})
print(emailresult.content)
#%%