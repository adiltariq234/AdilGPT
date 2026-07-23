import math
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_tavily import TavilySearch


web_search=TavilySearch(
    max_results=5,
    topic="general",
    search_depth='advanced'
)

CURRENT_THREAD_ID='default'
def set_current_thread_id(thread_id : str):
    global CURRENT_THREAD_ID
    CURRENT_THREAD_ID=thread_id


@tool
def calculator(expression : str) -> str:
    """
    Useful for simple math calculation
    imput should bi valid math expression
    Example : 2+2 math sqrt(16),10*5
    """
    try:
        allowed={
            'math' : math,
            'abs' : abs,
            'round' : round,
            'min' : min,
            'max' : max,
            'sum' : sum
        }

        result=eval(expression,{__builtins__},allowed)
        return str(result)
    

    except Exception as e : 
        return f"Calculation Error  { str(e)}"