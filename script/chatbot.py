from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

from agents import config_crew
from llm import run_llm

llm = run_llm()
crew = config_crew()

human = "{text}"
prompt = ChatPromptTemplate.from_messages([("human", human)])

chain = prompt | llm
response = chain.invoke(
    {
        "text": "Which company has better models OpenAI or Anthropic? Respond with just the company name."
    }
)

print(response.content)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

load_dotenv()

st.set_page_config(page_title="Chat Data Analyst", page_icon=":speech_balloon:")

st.title("Chat Data Analyst")

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        result = crew.kickoff(inputs={"query": user_query})
        print(result)
        result = str(result)
        st.markdown(result)
        
    st.session_state.chat_history.append(AIMessage(content=result))

