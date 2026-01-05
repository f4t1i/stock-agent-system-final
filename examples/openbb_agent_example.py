#!/usr/bin/env python3
"""
OpenBB Agents Example

Demonstrates how to use OpenBB Agents with LangChain for financial analysis.

This example shows:
1. Using OpenBB tools directly
2. Creating an LLM-powered financial analyst agent
3. Integrating with existing multi-agent system
"""

import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.openbb_agents import (
    create_openbb_tools,
    create_financial_analyst_agent,
    OpenBBStockQuoteTool,
    OpenBBFinancialsTool,
    OpenBBNewsTool,
    OpenBBTechnicalAnalysisTool,
    OpenBBScreenerTool
)

# ========== Example 1: Using OpenBB Tools Directly ==========

def example_direct_tools():
    """Example: Using OpenBB tools directly"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Direct Tool Usage")
    print("="*70)

    # Create tools
    quote_tool = OpenBBStockQuoteTool()
    financials_tool = OpenBBFinancialsTool()
    news_tool = OpenBBNewsTool()
    technical_tool = OpenBBTechnicalAnalysisTool()
    screener_tool = OpenBBScreenerTool()

    symbol = "AAPL"

    # Get quote
    print(f"\n📊 Stock Quote for {symbol}:")
    print(quote_tool._run(symbol))

    # Get financials
    print(f"\n💰 Financials for {symbol}:")
    print(financials_tool._run(symbol))

    # Get news
    print(f"\n📰 Recent News for {symbol}:")
    print(news_tool._run(symbol, limit=3))

    # Get technical analysis
    print(f"\n📈 Technical Analysis for {symbol}:")
    print(technical_tool._run(symbol))

    # Screen stocks
    print("\n🔍 Top Gainers:")
    print(screener_tool._run("top_gainers", limit=5))


# ========== Example 2: LLM-Powered Financial Analyst ==========

def example_llm_agent():
    """Example: Using LLM-powered agent with OpenBB tools"""
    print("\n" + "="*70)
    print("EXAMPLE 2: LLM-Powered Financial Analyst Agent")
    print("="*70)

    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_openai import ChatOpenAI

        # Choose your LLM
        # Option 1: Anthropic Claude
        if os.getenv('ANTHROPIC_API_KEY'):
            llm = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                temperature=0
            )
            print("\n✓ Using Claude 3.5 Sonnet")

        # Option 2: OpenAI GPT-4
        elif os.getenv('OPENAI_API_KEY'):
            llm = ChatOpenAI(
                model="gpt-4-turbo-preview",
                temperature=0
            )
            print("\n✓ Using GPT-4")

        else:
            print("\n⚠️  No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY")
            return

        # Create agent
        agent = create_financial_analyst_agent(llm)

        # Example queries
        queries = [
            "Analyze AAPL stock. Should I buy, sell, or hold?",
            "Compare AAPL vs MSFT. Which is a better investment?",
            "Find the top 3 gainers today and tell me which one has the best fundamentals",
            "What are the main risks for TSLA right now based on recent news and technical indicators?"
        ]

        for query in queries[:1]:  # Start with one query
            print(f"\n{'='*70}")
            print(f"Query: {query}")
            print("="*70)

            response = agent.analyze(query)
            print(f"\nAgent Response:\n{response}")

    except ImportError as e:
        print(f"\n⚠️  LangChain not installed: {e}")
        print("Install with: pip install langchain langchain-anthropic langchain-openai")


# ========== Example 3: Integration with Existing System ==========

def example_system_integration():
    """Example: Integrate OpenBB agents with existing multi-agent system"""
    print("\n" + "="*70)
    print("EXAMPLE 3: System Integration")
    print("="*70)

    from orchestration.coordinator import SystemCoordinator

    # Create coordinator
    coordinator = SystemCoordinator()

    # Get OpenBB tools
    openbb_tools = create_openbb_tools()

    print(f"\n✓ Created {len(openbb_tools)} OpenBB tools:")
    for tool in openbb_tools:
        print(f"  - {tool.name}: {tool.description[:60]}...")

    # Example: Enhanced analysis workflow
    symbol = "AAPL"

    print(f"\n🔄 Running enhanced analysis for {symbol}...")

    # 1. Use OpenBB for data gathering
    quote_tool = OpenBBStockQuoteTool()
    technical_tool = OpenBBTechnicalAnalysisTool()
    financials_tool = OpenBBFinancialsTool()

    quote_data = quote_tool._run(symbol)
    technical_data = technical_tool._run(symbol)
    financial_data = financials_tool._run(symbol)

    # 2. Run existing multi-agent analysis
    try:
        result = coordinator.analyze_symbol(symbol)

        print(f"\n📊 Analysis Result:")
        print(f"  Recommendation: {result.get('recommendation', 'N/A')}")
        print(f"  Confidence: {result.get('confidence', 0):.2%}")
        print(f"  Position Size: {result.get('position_size', 0):.2%}")

        if 'reasoning' in result:
            print(f"\n  Reasoning: {result['reasoning'][:200]}...")

    except Exception as e:
        print(f"\n⚠️  Coordinator error: {e}")

    print("\n✓ Enhanced analysis complete (OpenBB + Multi-Agent System)")


# ========== Example 4: Custom Financial Research Agent ==========

def example_custom_research_agent():
    """Example: Build custom research agent with OpenBB tools"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Financial Research Agent")
    print("="*70)

    class CustomFinancialResearcher:
        """Custom financial researcher using OpenBB tools"""

        def __init__(self):
            self.quote_tool = OpenBBStockQuoteTool()
            self.financials_tool = OpenBBFinancialsTool()
            self.news_tool = OpenBBNewsTool()
            self.technical_tool = OpenBBTechnicalAnalysisTool()
            self.screener_tool = OpenBBScreenerTool()

        def research_stock(self, symbol: str) -> Dict:
            """Comprehensive stock research"""
            print(f"\n🔍 Researching {symbol}...")

            research = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }

            # Gather data
            print("  ├─ Fetching quote...")
            research['quote'] = self.quote_tool._run(symbol)

            print("  ├─ Analyzing financials...")
            research['financials'] = self.financials_tool._run(symbol)

            print("  ├─ Checking news...")
            research['news'] = self.news_tool._run(symbol, limit=5)

            print("  └─ Running technical analysis...")
            research['technical'] = self.technical_tool._run(symbol)

            return research

        def find_opportunities(self):
            """Find investment opportunities"""
            print("\n💡 Finding opportunities...")

            # Get gainers
            gainers = self.screener_tool._run("top_gainers", limit=10)

            # Get losers (potential value)
            losers = self.screener_tool._run("top_losers", limit=10)

            # Get most active
            active = self.screener_tool._run("most_active", limit=10)

            return {
                'gainers': gainers,
                'losers': losers,
                'most_active': active
            }

    # Use custom researcher
    from datetime import datetime

    researcher = CustomFinancialResearcher()

    # Research a stock
    research = researcher.research_stock("AAPL")
    print("\n✓ Research complete")

    # Find opportunities
    opportunities = researcher.find_opportunities()
    print("\n✓ Opportunities identified")


# ========== Main ==========

def main():
    """Run examples"""
    print("\n" + "="*70)
    print("OpenBB Agents Examples")
    print("="*70)

    # Run examples
    example_direct_tools()

    print("\n" + "="*70)
    print("Press Enter to continue to LLM Agent example...")
    print("="*70)
    input()

    example_llm_agent()

    print("\n" + "="*70)
    print("Press Enter to continue to System Integration example...")
    print("="*70)
    input()

    example_system_integration()

    print("\n" + "="*70)
    print("Press Enter to continue to Custom Research Agent example...")
    print("="*70)
    input()

    example_custom_research_agent()

    print("\n" + "="*70)
    print("✓ All examples complete!")
    print("="*70)


if __name__ == "__main__":
    main()
