import sqlite3
import pandas as pd
from langchain_community.utilities.sql_database import SQLDatabase

def run_database():
    CSV_PATH = '../data/supermarket_sales.csv'
    df = pd.read_csv(CSV_PATH)
    
    connection = sqlite3.connect("sales.db")
    
    df.to_sql(name="sales", con=connection, if_exists='replace', index=False)
    
    db = SQLDatabase.from_uri("sqlite:///sales.db")
    
    return db