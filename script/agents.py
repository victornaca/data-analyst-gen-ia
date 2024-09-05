from textwrap import dedent
import pandas as pd
from crewai import Agent, Crew, Process, Task
from crewai_tools import tool
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)

from database import run_database
from llm import run_llm

db = run_database()
llm = run_llm()

@tool("list_tables")
def list_tables() -> str:
    """List the available tables in the database"""
    return ListSQLDatabaseTool(db=db).invoke("")

@tool("tables_schema")
def tables_schema(tables: str) -> str:
    """
    Input is a comma-separated list of tables, output is the schema and sample rows
    for those tables. Be sure that the tables actually exist by calling `list_tables` first!
    Example Input: table1, table2, table3
    """
    tool = InfoSQLDatabaseTool(db=db)
    return tool.invoke(tables)

@tool("execute_sql")
def execute_sql(sql_query: str) -> str:
    """Execute a SQL query against the database. Returns the result"""
    return QuerySQLDataBaseTool(db=db).invoke(sql_query)

@tool("check_sql")
def check_sql(sql_query: str) -> str:
    """
    Use this tool to double check if your query is correct before executing it. Always use this
    tool before executing a query with `execute_sql`.
    """
    return QuerySQLCheckerTool(db=db, llm=llm).invoke({"query": sql_query})

class AgentsChatBot():
    def agent_sql_dev():
        sql_dev = Agent(
            role="Senior Database Developer",
            goal="Construct and execute SQL queries based on a request",
            backstory=dedent(
                """
                You are an experienced database engineer who is master at creating efficient and complex SQL queries.
                You have a deep understanding of how different databases work and how to optimize queries.
                Use the `list_tables` to find available tables.
                Use the `tables_schema` to understand the metadata for the tables.
                Use the `execute_sql` to check your queries for correctness.
                Use the `check_sql` to execute queries against the database.
            """
            ),
            llm=llm,
            tools=[list_tables, tables_schema, execute_sql, check_sql],
            allow_delegation=False,
        )
        return sql_dev

    def agent_data_analyst():
        data_analyst = Agent(
            role="Senior Data Analyst",
            goal="You receive data from the database developer and analyze it",
            backstory=dedent(
                """
                You have deep experience with analyzing datasets using Python.
                Your work is always based on the provided data and is clear,
                easy-to-understand and to the point. You have attention
                to detail and always produce very detailed work (as long as you need).
            """
            ),
            llm=llm,
            allow_delegation=False,
        )
        return data_analyst

    def agent_report_writer():
        report_writer = Agent(
            role="Senior Report Editor",
            goal="Write an executive summary type of report based on the work of the analyst",
            backstory=dedent(
                """
                Your writing still is well known for clear and effective communication.
                You always summarize long texts into bullet points that contain the most
                important details.
                """
            ),
            llm=llm,
            allow_delegation=False,
        )
        return report_writer

extract_data = Task(
    description="Extract data that is required for the query {query}.",
    expected_output="Database result for the query",
    agent=AgentsChatBot.agent_sql_dev(),
)

analyze_data = Task(
    description="Analyze the data from the database and write an analysis for {query}.",
    expected_output="Detailed analysis text",
    agent=AgentsChatBot.agent_data_analyst(),
    context=[extract_data],
)

write_report = Task(
    description=dedent(
        """
        Write an executive summary of the report from the analysis. The report
        must be less than 100 words.
        """
    ),
    expected_output="Markdown report",
    agent=AgentsChatBot.agent_report_writer(),
    context=[analyze_data],
)

def config_crew():
    crew = Crew(
        agents=[AgentsChatBot.agent_sql_dev(), AgentsChatBot.agent_data_analyst(), AgentsChatBot.agent_report_writer()],
        tasks=[extract_data, analyze_data, write_report],
        process=Process.sequential,
        verbose=True,
        memory=False,
        output_log_file="crew.log",
    )
    return crew