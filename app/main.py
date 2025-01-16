import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chain import Chain
from portfolio import Portfolio


def create_app(llm, portfolio):
    st.title("✉️ Cold Email Generator")
    url = st.text_input("Job Post URL:", value = "https://jobs.nike.com/job/R-50367")
    submit_button = st.button("Submit")

    if submit_button:
        loader = WebBaseLoader([url])
        page_data = loader.load().pop().page_content
        portfolio.load_portfolio()
        jsonresult = llm.extract(page_data)
        for result in jsonresult:
            skills = result.get("skills", [])
            links = portfolio.ask(skills)
            cmail = llm.write(result, links)
            st.code(cmail, language = "markdown")
        
        

if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout = "wide", page_title = "Email Gen", page_icon = "📧")
    create_app(chain, portfolio)