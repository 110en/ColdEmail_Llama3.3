#%%
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            model_name = "llama-3.3-70b-versatile",
            groq_api_key = os.getenv("GROQ_API_KEY"),
            temperature = 0
        )

    def extract(self, page_data):
        prompt_extract = PromptTemplate.from_template(
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
        chain_extract = prompt_extract | self.llm
        result  = chain_extract.invoke(input={"page_data" : page_data})
        parser = JsonOutputParser()
        jsonresult = parser.parse(result.content)
        return jsonresult if isinstance(jsonresult, list) else [jsonresult]
    
    def write(self, jsonresult, links):
        prompt_cmail = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {jsonresult}

            ### INSTRUCTION:
            You are JimBob, a business development executive at OpenAI. your job is to write a cold email to the client regarding the job 
            mentioned above describing the capability of OpenAI in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase OpenAI's portfolio: {link_list}
            Remember, you are JimBob, BDE at OpenAI.

            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_cmail | self.llm
        emailresult = chain_email.invoke({"jsonresult" : str(jsonresult), "link_list" : links})
        return emailresult.content


# %%

