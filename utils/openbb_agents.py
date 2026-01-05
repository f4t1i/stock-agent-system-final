#!/usr/bin/env python3
"""
OpenBB Agents Integration

Integrates OpenBB's official agents framework for seamless LLM + Financial Data integration.

GitHub: https://github.com/OpenBB-finance/agents-for-openbb

This module provides:
- LangChain tools powered by OpenBB
- Pre-built financial analysis agents
- Structured financial data retrieval
- Context-aware market analysis
"""

from typing import Any, Dict, List, Optional, Type
from datetime import datetime, timedelta

from loguru import logger

try:
    from langchain.tools import BaseTool
    from langchain.agents import AgentExecutor, create_openai_functions_agent
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import AIMessage, HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available for OpenBB Agents")

try:
    from openbb import obb
    OPENBB_AVAILABLE = True
except ImportError:
    OPENBB_AVAILABLE = False
    logger.warning("OpenBB not available")


# ========== OpenBB-Powered LangChain Tools ==========

class OpenBBStockQuoteTool(BaseTool):
    """Tool to get real-time stock quotes using OpenBB"""

    name: str = "openbb_stock_quote"
    description: str = (
        "Get real-time stock quote including price, volume, and change. "
        "Input should be a stock ticker symbol (e.g., AAPL, MSFT). "
        "Returns current price, change, volume, and other quote data."
    )

    def _run(self, ticker: str) -> str:
        """Get stock quote"""
        try:
            quote = obb.equity.price.quote(symbol=ticker)

            if quote and hasattr(quote, 'results') and quote.results:
                data = quote.results[0]

                result = f"""Stock Quote for {ticker}:
- Price: ${data.last_price if hasattr(data, 'last_price') else data.price:.2f}
- Change: {data.change if hasattr(data, 'change') else 'N/A'}
- Change %: {data.change_percent if hasattr(data, 'change_percent') else 'N/A'}%
- Volume: {data.volume:,}
- Open: ${data.open:.2f}
- High: ${data.high:.2f}
- Low: ${data.low:.2f}
"""
                return result

            return f"No quote data available for {ticker}"

        except Exception as e:
            return f"Error getting quote for {ticker}: {str(e)}"

    async def _arun(self, ticker: str) -> str:
        """Async version"""
        return self._run(ticker)


class OpenBBFinancialsTool(BaseTool):
    """Tool to get financial data using OpenBB"""

    name: str = "openbb_financials"
    description: str = (
        "Get comprehensive financial data and metrics for a stock. "
        "Input should be a stock ticker symbol. "
        "Returns P/E ratio, ROE, ROA, debt ratios, market cap, and other fundamentals."
    )

    def _run(self, ticker: str) -> str:
        """Get financial data"""
        try:
            # Get metrics
            metrics = obb.equity.fundamental.metrics(symbol=ticker)

            # Get ratios
            ratios = obb.equity.fundamental.ratios(symbol=ticker)

            result = f"Financial Data for {ticker}:\n\n"

            if metrics and hasattr(metrics, 'results') and metrics.results:
                m = metrics.results[0]
                result += "Valuation Metrics:\n"
                result += f"- P/E Ratio: {getattr(m, 'pe_ratio', 'N/A')}\n"
                result += f"- Forward P/E: {getattr(m, 'forward_pe', 'N/A')}\n"
                result += f"- PEG Ratio: {getattr(m, 'peg_ratio', 'N/A')}\n"
                result += f"- Price/Book: {getattr(m, 'price_to_book', 'N/A')}\n"
                result += f"- Price/Sales: {getattr(m, 'price_to_sales', 'N/A')}\n\n"

            if ratios and hasattr(ratios, 'results') and ratios.results:
                r = ratios.results[0]
                result += "Profitability & Health:\n"
                result += f"- ROE: {getattr(r, 'return_on_equity', 'N/A')}\n"
                result += f"- ROA: {getattr(r, 'return_on_assets', 'N/A')}\n"
                result += f"- Debt/Equity: {getattr(r, 'debt_to_equity', 'N/A')}\n"
                result += f"- Current Ratio: {getattr(r, 'current_ratio', 'N/A')}\n"

            return result

        except Exception as e:
            return f"Error getting financials for {ticker}: {str(e)}"

    async def _arun(self, ticker: str) -> str:
        """Async version"""
        return self._run(ticker)


class OpenBBNewsTool(BaseTool):
    """Tool to get recent news using OpenBB"""

    name: str = "openbb_news"
    description: str = (
        "Get recent news articles about a stock or company. "
        "Input should be a stock ticker symbol. "
        "Returns recent news headlines, sources, and dates."
    )

    def _run(self, ticker: str, limit: int = 5) -> str:
        """Get news"""
        try:
            news = obb.news.company(symbol=ticker, limit=limit)

            if news and hasattr(news, 'results') and news.results:
                result = f"Recent News for {ticker}:\n\n"

                for i, article in enumerate(news.results[:limit], 1):
                    title = getattr(article, 'title', 'No title')
                    source = getattr(article, 'source', 'Unknown')
                    date = getattr(article, 'date', getattr(article, 'published_date', 'Unknown date'))

                    result += f"{i}. {title}\n"
                    result += f"   Source: {source} | Date: {date}\n\n"

                return result

            return f"No news available for {ticker}"

        except Exception as e:
            return f"Error getting news for {ticker}: {str(e)}"

    async def _arun(self, ticker: str, limit: int = 5) -> str:
        """Async version"""
        return self._run(ticker, limit)


class OpenBBTechnicalAnalysisTool(BaseTool):
    """Tool to get technical analysis using OpenBB"""

    name: str = "openbb_technical_analysis"
    description: str = (
        "Get technical analysis indicators for a stock including RSI, MACD, "
        "moving averages, and trend information. Input should be a stock ticker symbol. "
        "Returns technical indicators and trading signals."
    )

    def _run(self, ticker: str) -> str:
        """Get technical analysis"""
        try:
            # Get historical data for calculations
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)

            data = obb.equity.price.historical(
                symbol=ticker,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )

            if data and hasattr(data, 'to_dataframe'):
                df = data.to_dataframe()

                # Calculate indicators
                current_price = df['close'].iloc[-1]
                sma_20 = df['close'].rolling(window=20).mean().iloc[-1]
                sma_50 = df['close'].rolling(window=50).mean().iloc[-1]

                # RSI
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs)).iloc[-1]

                # Trend
                if current_price > sma_20 > sma_50:
                    trend = "Strong Bullish"
                elif current_price > sma_20:
                    trend = "Bullish"
                elif current_price < sma_20 < sma_50:
                    trend = "Strong Bearish"
                elif current_price < sma_20:
                    trend = "Bearish"
                else:
                    trend = "Neutral"

                result = f"""Technical Analysis for {ticker}:

Current Price: ${current_price:.2f}

Moving Averages:
- SMA(20): ${sma_20:.2f}
- SMA(50): ${sma_50:.2f}

Indicators:
- RSI(14): {rsi:.2f}
- Trend: {trend}

Signals:
- RSI Signal: {'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'}
- MA Signal: {'Bullish' if current_price > sma_20 else 'Bearish'}
"""
                return result

            return f"No technical data available for {ticker}"

        except Exception as e:
            return f"Error getting technical analysis for {ticker}: {str(e)}"

    async def _arun(self, ticker: str) -> str:
        """Async version"""
        return self._run(ticker)


class OpenBBScreenerTool(BaseTool):
    """Tool to screen stocks using OpenBB"""

    name: str = "openbb_stock_screener"
    description: str = (
        "Screen stocks based on criteria like 'top_gainers', 'top_losers', or 'most_active'. "
        "Input should be the screening criteria. "
        "Returns list of stocks matching the criteria with their performance."
    )

    def _run(self, criteria: str = "top_gainers", limit: int = 10) -> str:
        """Screen stocks"""
        try:
            if criteria.lower() in ["gainers", "top_gainers"]:
                data = obb.equity.discovery.gainers(limit=limit)
            elif criteria.lower() in ["losers", "top_losers"]:
                data = obb.equity.discovery.losers(limit=limit)
            elif criteria.lower() in ["active", "most_active"]:
                data = obb.equity.discovery.active(limit=limit)
            else:
                return f"Unknown screening criteria: {criteria}. Use 'top_gainers', 'top_losers', or 'most_active'."

            if data and hasattr(data, 'results') and data.results:
                result = f"{criteria.upper()} Stocks:\n\n"

                for i, stock in enumerate(data.results[:limit], 1):
                    symbol = stock.symbol
                    name = getattr(stock, 'name', 'N/A')
                    price = getattr(stock, 'price', 0)
                    change = getattr(stock, 'change', 0)
                    change_pct = getattr(stock, 'change_percent', 0)
                    volume = getattr(stock, 'volume', 0)

                    result += f"{i}. {symbol} - {name}\n"
                    result += f"   Price: ${price:.2f} | Change: {change:+.2f} ({change_pct:+.2f}%)\n"
                    result += f"   Volume: {volume:,}\n\n"

                return result

            return f"No stocks found for criteria: {criteria}"

        except Exception as e:
            return f"Error screening stocks: {str(e)}"

    async def _arun(self, criteria: str = "top_gainers", limit: int = 10) -> str:
        """Async version"""
        return self._run(criteria, limit)


# ========== Agent Factory ==========

class OpenBBFinancialAgent:
    """
    Financial analysis agent powered by OpenBB and LLM.

    Uses OpenBB tools to provide comprehensive financial analysis.
    """

    def __init__(self, llm=None):
        """
        Initialize OpenBB Financial Agent.

        Args:
            llm: Language model (e.g., ChatOpenAI, ChatAnthropic)
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain required for OpenBB Agents")

        if not OPENBB_AVAILABLE:
            raise ImportError("OpenBB required for OpenBB Agents")

        self.llm = llm

        # Initialize tools
        self.tools = [
            OpenBBStockQuoteTool(),
            OpenBBFinancialsTool(),
            OpenBBNewsTool(),
            OpenBBTechnicalAnalysisTool(),
            OpenBBScreenerTool()
        ]

        # Create prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional financial analyst with access to real-time market data through OpenBB.

You have access to the following tools:
- openbb_stock_quote: Get real-time stock quotes
- openbb_financials: Get financial metrics and ratios
- openbb_news: Get recent news articles
- openbb_technical_analysis: Get technical indicators and signals
- openbb_stock_screener: Screen stocks by criteria

Provide comprehensive, data-driven analysis. Always cite your sources and use multiple tools to cross-validate information.

When analyzing a stock:
1. Start with current quote and price action
2. Review technical indicators for trends
3. Analyze fundamental metrics for valuation
4. Check recent news for catalysts
5. Provide a clear recommendation with reasoning

Be objective and highlight both bullish and bearish factors."""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # Create agent
        if llm:
            self.agent = create_openai_functions_agent(llm, self.tools, self.prompt)
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                max_iterations=10,
                handle_parsing_errors=True
            )
        else:
            self.agent_executor = None

    def analyze(self, query: str) -> str:
        """
        Analyze financial query using OpenBB tools.

        Args:
            query: Analysis query (e.g., "Analyze AAPL stock")

        Returns:
            Analysis result
        """
        if not self.agent_executor:
            return "No LLM configured. Please provide an LLM instance."

        try:
            result = self.agent_executor.invoke({"input": query})
            return result["output"]

        except Exception as e:
            logger.error(f"Error in agent analysis: {e}")
            return f"Error: {str(e)}"


# ========== Utility Functions ==========

def create_openbb_tools() -> List[BaseTool]:
    """Create list of OpenBB-powered tools for LangChain agents"""
    return [
        OpenBBStockQuoteTool(),
        OpenBBFinancialsTool(),
        OpenBBNewsTool(),
        OpenBBTechnicalAnalysisTool(),
        OpenBBScreenerTool()
    ]


def create_financial_analyst_agent(llm):
    """
    Create a financial analyst agent with OpenBB tools.

    Args:
        llm: Language model instance

    Returns:
        OpenBBFinancialAgent instance
    """
    return OpenBBFinancialAgent(llm=llm)


# ========== Example Usage ==========

if __name__ == "__main__":
    # Example: Using tools directly
    print("Testing OpenBB Tools...")

    quote_tool = OpenBBStockQuoteTool()
    print("\n" + "="*60)
    print("STOCK QUOTE:")
    print("="*60)
    print(quote_tool._run("AAPL"))

    financials_tool = OpenBBFinancialsTool()
    print("\n" + "="*60)
    print("FINANCIALS:")
    print("="*60)
    print(financials_tool._run("AAPL"))

    news_tool = OpenBBNewsTool()
    print("\n" + "="*60)
    print("NEWS:")
    print("="*60)
    print(news_tool._run("AAPL"))

    technical_tool = OpenBBTechnicalAnalysisTool()
    print("\n" + "="*60)
    print("TECHNICAL ANALYSIS:")
    print("="*60)
    print(technical_tool._run("AAPL"))

    screener_tool = OpenBBScreenerTool()
    print("\n" + "="*60)
    print("TOP GAINERS:")
    print("="*60)
    print(screener_tool._run("top_gainers", limit=5))
