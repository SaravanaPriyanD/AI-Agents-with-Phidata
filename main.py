from phi.agent import Agent 
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv
from phi.tools.technical_analysis import TechnicalAnalysisTools
import logging
from typing import Optional


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='financial_ai_agents.log'
)
logger = logging.getLogger(__name__)

# web search agent
web_search_agent = Agent(
    name= "Web Search Agent",
    role= "Search the web for latest information",
    model = Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls= True,
    markdown= True,
)



# Financial agent
'''
finance_agent = Agent(
    name= "Financial Agent",
    role= "Gather financial data about companies",
    model = Groq(id="deepseek-r1-distill-llama-70b"),
     tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                      company_news=True),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls= True,
    markdown= True,
)
'''
finance_agent = Agent(
    name="Finance Analysis Agent",
    role="Analyze financial markets and provide detailed insights",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[
        YFinanceTools(
            stock_price=True, 
            stock_fundamentals=True,
            income_statement=True,
            balance_sheet=True,
            cash_flow=True,
            earnings=True
        )
    ],
    instructions=[
        "Provide detailed financial analysis",
        "Include key financial metrics and ratios",
        "Use tables to present numerical data",
        "Highlight significant trends"
    ],
    show_tool_calls=True,
    markdown=True
)

from phi.tools.twitter import TwitterTools  # For social media sentiment
from phi.tools.news import NewsTools       # For news sentiment

sentiment_agent = Agent(
    name="Sentiment Analysis Agent",
    role="Analyze market sentiment from various sources",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[
        TwitterTools(),
        NewsTools()
    ],
    instructions=[
        "Analyze sentiment from multiple sources",
        "Provide sentiment scores",
        "Highlight key trends and opinions",
        "Summarize overall market sentiment"
    ],
    show_tool_calls=True,
    markdown=True
)

technical_agent = Agent(
    name="Technical Analysis Agent",
    role="Perform technical analysis on stocks",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[
        TechnicalAnalysisTools(
            moving_averages=True,
            relative_strength_index=True,
            bollinger_bands=True,
            macd=True
        )
    ],
    instructions=[
        "Provide technical indicators analysis",
        "Identify key support and resistance levels",
        "Generate trading signals based on technical patterns",
        "Include visual charts when possible"
    ],
    show_tool_calls=True,
    markdown=True
)

enhanced_multi_agent = Agent(
    team=[web_search_agent, finance_agent, sentiment_agent, technical_agent],
    instructions=[
        "Provide comprehensive market analysis",
        "Include fundamental and technical analysis",
        "Consider market sentiment",
        "Always include sources",
        "Use tables and charts for data visualization",
        "Highlight key risks and opportunities"
    ],
    show_tool_calls=True,
    markdown=True
)

enhanced_multi_agent.print_response("Summarize analyst recommendation and share the latest news for NVDA", stream=True)

# Finance analysis agent

def analyze_stock_complete(ticker):
    """Perform a comprehensive analysis of a stock"""
    query = f"""Provide a comprehensive analysis of {ticker} including:
    1. Financial fundamentals
    2. Technical indicators
    3. Market sentiment
    4. Recent news
    5. Future outlook"""
    enhanced_multi_agent.print_response(query, stream=True)

def market_sentiment_analysis():
    """Analyze overall market sentiment"""
    sentiment_agent.print_response(
        "What is the current market sentiment based on social media and news? Focus on major indices and trending stocks.",
        stream=True
    )

def technical_signals_scan():
    """Scan for technical trading signals"""
    technical_agent.print_response(
        "Scan major tech stocks for technical trading signals. Look for breakout patterns and significant technical setups.",
        stream=True
    )

def safe_agent_query(agent: Agent, query: str) -> Optional[str]:
    """Safely execute agent queries with error handling"""
    try:
        return agent.print_response(query, stream=True)
    except Exception as e:
        logger.error(f"Error executing query '{query}': {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    # Analyze a specific stock
    analyze_stock_complete("AAPL")
    
    # Get market sentiment
    market_sentiment_analysis()
    
    # Scan for technical signals
    technical_signals_scan()


