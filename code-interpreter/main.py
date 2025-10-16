from typing import Any

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI

load_dotenv()

# Configuration constants
PYTHON_AGENT_INSTRUCTIONS = """You are an agent designed to write and execute python code to answer questions.
You have access to a python REPL, which you can use to execute python code.
You have qrcode package installed
If you get an error, debug your code and try again.
Only use the output of your code to answer the question. 
You might know the answer without running any code, but you should still run the code to get the answer.
If it does not seem like you can write code to answer the question, just return "I don't know" as the answer."""

CSV_PATH = "episode_info.csv"

# Model configurations
GPT4_TURBO_CONFIG = {"temperature": 0, "model": "gpt-4-turbo"}
GPT4_CONFIG = {"temperature": 0, "model": "gpt-4"}


def create_python_agent() -> AgentExecutor:
    """Create and return a Python code execution agent."""
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=PYTHON_AGENT_INSTRUCTIONS)
    
    tools = [PythonREPLTool()]
    python_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(**GPT4_TURBO_CONFIG),
        tools=tools,
    )
    
    return AgentExecutor(agent=python_agent, tools=tools, verbose=True)


def create_csv_analysis_agent() -> AgentExecutor:
    """Create and return a CSV analysis agent."""
    return create_csv_agent(
        llm=ChatOpenAI(**GPT4_CONFIG),
        path=CSV_PATH,
        verbose=True,
        allow_dangerous_code=True
    )


def create_router_agent(python_agent: AgentExecutor, csv_agent: AgentExecutor) -> AgentExecutor:
    """Create and return the router agent that delegates to specialized agents."""
    def python_agent_wrapper(original_prompt: str) -> dict[str, Any]:
        return python_agent.invoke({"input": original_prompt})
    
    def csv_agent_wrapper(original_prompt: str) -> dict[str, Any]:
        return csv_agent.invoke({"input": original_prompt})
    
    tools = [
        Tool(
            name="Python Agent",
            func=python_agent_wrapper,
            description="""useful when you need to transform natural language to python and execute the python code,
                          returning the results of the code execution
                          DOES NOT ACCEPT CODE AS INPUT""",
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent_wrapper,
            description="""useful when you need to answer question over episode_info.csv file,
                         takes an input the entire question and returns the answer after running pandas calculations""",
        ),
    ]
    
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions="")
    router_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(**GPT4_TURBO_CONFIG),
        tools=tools,
    )
    
    return AgentExecutor(agent=router_agent, tools=tools, verbose=True)


def main():
    """Main entry point for the code interpreter system."""
    print("Starting Code Interpreter System...")
    
    # Create agents
    python_agent = create_python_agent()
    csv_agent = create_csv_analysis_agent()
    router_agent = create_router_agent(python_agent, csv_agent)

    # Process queries
    queries = [
        "Which season has the most episodes?",
        "Generate and save in current working directory 15 qrcodes that point to `www.amazon.com`"
    ]
    
    for query in queries:
        print(f"\nProcessing: {query}")
        result = router_agent.invoke({"input": query})
        print(f"Result: {result}")
        print("-" * 50)
    
    print("\nAll queries processed successfully!")


if __name__ == "__main__":
    main()