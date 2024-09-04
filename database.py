import json
from dotenv import load_dotenv
import os
import sqlite3
from dataclasses import asdict, dataclass
from langchain_core.messages import AIMessage, HumanMessage
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Tuple, Union
import pandas as pd
from crewai import Agent, Crew, Process, Task
from crewai_tools import tool
# from google.colab import userdata
from langchain.schema import AgentFinish
from langchain.schema.output import LLMResult
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import streamlit as st

def run_database():
    db = SQLDatabase.from_uri("sqlite:///sales.db")
    return db