import pandas as pd
import chromadb


class Portfolio:
    def __init__(self, file_path="app/resources/my_portfolio.csv"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.chroma = chromadb.PersistentClient('vectorstore')
        self.collection = self.chroma.get_or_create_collection(name='portfolio')

    def load_portfolio(self):
        x = 0
        if not self.collection.count():
            for _, row in self.data.iterrows():
                self.collection.add(
                    documents = row["Techstack"],
                    metadatas = {"links" : row["Links"]},
                    ids = ["id" + x]
                )
                x += 1

        

    def ask(self, skills):
        return self.collection.query(query_texts = skills,n_results = 2).get("metadatas", [])
    


    
    