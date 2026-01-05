# OpenBB Platform Integration

**Professional-grade financial data and analytics integration**

---

## 🎯 Overview

This project now integrates [OpenBB Platform](https://github.com/OpenBB-finance/OpenBB) - the leading open-source investment research platform.

### What is OpenBB?

OpenBB is a comprehensive financial data and analytics platform providing:
- ✅ **Market Data:** Stocks, Options, Futures, Crypto, Forex
- ✅ **Technical Analysis:** 100+ indicators, patterns, signals
- ✅ **Fundamental Data:** Financials, ratios, estimates, screeners
- ✅ **News & Sentiment:** Real-time news aggregation
- ✅ **Economic Data:** Macro indicators, calendars
- ✅ **Alternative Data:** Social sentiment, insider trading
- ✅ **Portfolio Analytics:** Performance, risk, attribution

---

## 📦 Installation

### 1. Install OpenBB Platform

```bash
# Install OpenBB Platform
pip install openbb

# Or with all data providers
pip install openbb[all]

# For LangChain integration (OpenBB Agents)
pip install langchain langchain-anthropic langchain-openai
```

**Note:** The project includes **OpenBB Agents** integration for seamless LLM + Financial Data workflows.
- GitHub: https://github.com/OpenBB-finance/agents-for-openbb
- Provides LangChain tools powered by OpenBB
- Pre-built financial analysis agent templates
- Best practices for AI + Financial Data

### 2. Configure API Keys (Optional)

OpenBB works with multiple data providers. Configure your preferred providers:

```bash
# Set up OpenBB credentials
python -c "from openbb import obb; obb.account.login()"

# Or set environment variables
export OPENBB_API_KEY=your_key
export FMP_API_KEY=your_fmp_key  # Financial Modeling Prep
export POLYGON_API_KEY=your_polygon_key
```

**Available Providers:**
- **Free:** Yahoo Finance, Alpha Vantage, FRED, SEC
- **Paid:** Financial Modeling Prep, Polygon, Intrinio, Benzinga

---

## 🚀 Quick Start

### Basic Usage

```python
from utils.openbb_provider import get_openbb_provider

# Get provider instance
obb_provider = get_openbb_provider()

# Get real-time quote
quote = obb_provider.get_quote('AAPL')
print(f"AAPL Price: ${quote['price']}")

# Get historical data
df = obb_provider.get_historical_data('AAPL', start_date='2024-01-01')
print(df.head())

# Get technical indicators
indicators = obb_provider.get_technical_indicators('AAPL')
print(f"RSI: {indicators['rsi']:.2f}")
print(f"Trend: {indicators['trend']}")

# Get fundamental data
fundamentals = obb_provider.get_fundamental_data('AAPL')
print(f"P/E Ratio: {fundamentals['pe_ratio']}")
print(f"ROE: {fundamentals['roe']:.2%}")

# Get news
news = obb_provider.get_news('AAPL', limit=5)
for article in news:
    print(f"- {article['title']}")
```

---

## 📊 Features & Capabilities

### 1. Market Data

```python
# Real-time quote
quote = obb_provider.get_quote('AAPL')
# Returns: price, open, high, low, volume, change, change_percent

# Historical OHLCV
df = obb_provider.get_historical_data(
    symbol='AAPL',
    start_date='2024-01-01',
    end_date='2024-12-31',
    interval='1d'  # 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo
)

# Intraday data
df_intraday = obb_provider.get_historical_data(
    symbol='AAPL',
    interval='5m'  # 5-minute bars
)
```

### 2. Technical Analysis

```python
# Comprehensive technical indicators
indicators = obb_provider.get_technical_indicators('AAPL', period=90)

# Available indicators:
print(f"SMA(20): {indicators['sma_20']}")
print(f"SMA(50): {indicators['sma_50']}")
print(f"EMA(12): {indicators['ema_12']}")
print(f"EMA(26): {indicators['ema_26']}")
print(f"RSI(14): {indicators['rsi']}")
print(f"MACD: {indicators['macd']}")
print(f"MACD Signal: {indicators['macd_signal']}")
print(f"BB Upper: {indicators['bb_upper']}")
print(f"BB Lower: {indicators['bb_lower']}")
print(f"Trend: {indicators['trend']}")  # bullish/bearish/neutral

# Support & Resistance
levels = obb_provider.get_support_resistance('AAPL', period=90)
print(f"Support: {levels['support_levels']}")
print(f"Resistance: {levels['resistance_levels']}")
```

### 3. Fundamental Analysis

```python
# Company fundamentals
fundamentals = obb_provider.get_fundamental_data('AAPL')

# Available metrics:
print(f"Company: {fundamentals['company_name']}")
print(f"Sector: {fundamentals['sector']}")
print(f"Industry: {fundamentals['industry']}")
print(f"Market Cap: ${fundamentals['market_cap']:,.0f}")

# Valuation metrics
print(f"P/E Ratio: {fundamentals['pe_ratio']}")
print(f"Forward P/E: {fundamentals['forward_pe']}")
print(f"PEG Ratio: {fundamentals['peg_ratio']}")
print(f"P/B Ratio: {fundamentals['price_to_book']}")
print(f"P/S Ratio: {fundamentals['price_to_sales']}")

# Profitability metrics
print(f"ROE: {fundamentals['roe']:.2%}")
print(f"ROA: {fundamentals['roa']:.2%}")

# Financial health
print(f"Debt/Equity: {fundamentals['debt_to_equity']}")
print(f"Current Ratio: {fundamentals['current_ratio']}")
print(f"Quick Ratio: {fundamentals['quick_ratio']}")

# Financial statements
income_stmt = obb_provider.get_financial_statements(
    symbol='AAPL',
    statement_type='income',  # income, balance, cash
    period='annual',          # annual, quarter
    limit=5
)
```

### 4. News & Sentiment

```python
# Company news
news = obb_provider.get_news('AAPL', limit=10)

for article in news:
    print(f"Title: {article['title']}")
    print(f"Source: {article['source']}")
    print(f"Date: {article['published_date']}")
    print(f"URL: {article['url']}")
    print()
```

### 5. Stock Screening

```python
# Top gainers
gainers = obb_provider.screen_stocks('top_gainers', limit=20)

# Top losers
losers = obb_provider.screen_stocks('top_losers', limit=20)

# Most active
active = obb_provider.screen_stocks('most_active', limit=20)

for stock in gainers:
    print(f"{stock['symbol']}: {stock['change_percent']:+.2f}%")
```

### 6. Options Data

```python
# Options chain
options = obb_provider.get_options_chain(
    symbol='AAPL',
    expiration='2024-12-20'  # or None for nearest
)

print(options.head())
```

### 7. Economic Data

```python
# Economic calendar
events = obb_provider.get_economic_calendar(
    start_date='2024-01-01',
    end_date='2024-12-31'
)

for event in events:
    print(f"{event['date']}: {event['event']} ({event['country']})")
    print(f"  Forecast: {event['forecast']}, Actual: {event['actual']}")
```

---

## 🔧 Integration with Agents

### Enhanced News Agent

```python
from utils.openbb_provider import get_openbb_provider

class EnhancedNewsAgent:
    def __init__(self):
        self.obb = get_openbb_provider()

    def analyze(self, symbol: str):
        # Get news from OpenBB
        news = self.obb.get_news(symbol, limit=10)

        # Analyze sentiment
        sentiment_score = self._analyze_sentiment(news)

        # Get market context
        quote = self.obb.get_quote(symbol)

        return {
            'sentiment_score': sentiment_score,
            'news_count': len(news),
            'recent_headlines': [n['title'] for n in news[:3]],
            'current_price': quote['price'],
            'price_change': quote['change_percent']
        }
```

### Enhanced Technical Agent

```python
class EnhancedTechnicalAgent:
    def __init__(self):
        self.obb = get_openbb_provider()

    def analyze(self, symbol: str):
        # Get technical indicators
        indicators = self.obb.get_technical_indicators(symbol, period=90)

        # Get support/resistance
        levels = self.obb.get_support_resistance(symbol)

        # Generate signals
        signal = self._generate_signal(indicators, levels)

        return {
            'signal': signal,
            'rsi': indicators['rsi'],
            'macd': indicators['macd'],
            'trend': indicators['trend'],
            'support_levels': levels['support_levels'],
            'resistance_levels': levels['resistance_levels']
        }
```

### Enhanced Fundamental Agent

```python
class EnhancedFundamentalAgent:
    def __init__(self):
        self.obb = get_openbb_provider()

    def analyze(self, symbol: str):
        # Get fundamental data
        fundamentals = self.obb.get_fundamental_data(symbol)

        # Get financial statements
        income = self.obb.get_financial_statements(
            symbol,
            statement_type='income',
            period='annual',
            limit=3
        )

        # Calculate valuation score
        valuation_score = self._calculate_valuation(fundamentals)

        return {
            'valuation_score': valuation_score,
            'pe_ratio': fundamentals['pe_ratio'],
            'roe': fundamentals['roe'],
            'debt_to_equity': fundamentals['debt_to_equity'],
            'recommendation': self._make_recommendation(valuation_score)
        }
```

---

## 📈 Advanced Usage

### Custom Indicators

```python
# Get historical data
df = obb_provider.get_historical_data('AAPL', start_date='2023-01-01')

# Calculate custom indicators
df['returns'] = df['close'].pct_change()
df['volatility'] = df['returns'].rolling(window=20).std() * (252 ** 0.5)
df['sharpe'] = df['returns'].rolling(window=20).mean() / df['returns'].rolling(window=20).std()

# Momentum indicators
df['momentum'] = df['close'].pct_change(periods=10)
df['volume_momentum'] = df['volume'].pct_change(periods=10)
```

### Multi-Symbol Analysis

```python
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

for symbol in symbols:
    quote = obb_provider.get_quote(symbol)
    indicators = obb_provider.get_technical_indicators(symbol)
    fundamentals = obb_provider.get_fundamental_data(symbol)

    print(f"\n{symbol}:")
    print(f"  Price: ${quote['price']:.2f} ({quote['change_percent']:+.2f}%)")
    print(f"  RSI: {indicators['rsi']:.1f}")
    print(f"  Trend: {indicators['trend']}")
    print(f"  P/E: {fundamentals['pe_ratio']:.2f}")
```

### Sector Analysis

```python
# Get top gainers
gainers = obb_provider.screen_stocks('top_gainers', limit=50)

# Group by sector (would need to fetch sector for each symbol)
sectors = {}
for stock in gainers:
    fund = obb_provider.get_fundamental_data(stock['symbol'])
    if fund and 'sector' in fund:
        sector = fund['sector']
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(stock)

# Analyze by sector
for sector, stocks in sectors.items():
    avg_change = sum(s['change_percent'] for s in stocks) / len(stocks)
    print(f"{sector}: {avg_change:+.2f}% ({len(stocks)} stocks)")
```

---

## 🎯 Use Cases

### 1. Real-Time Trading Dashboard

```python
def create_trading_dashboard(symbols):
    dashboard = {}

    for symbol in symbols:
        quote = obb_provider.get_quote(symbol)
        indicators = obb_provider.get_technical_indicators(symbol)

        dashboard[symbol] = {
            'price': quote['price'],
            'change': quote['change_percent'],
            'rsi': indicators['rsi'],
            'trend': indicators['trend'],
            'volume': quote['volume']
        }

    return dashboard
```

### 2. Automated Screening

```python
def screen_trading_opportunities():
    # Get gainers
    gainers = obb_provider.screen_stocks('top_gainers', limit=50)

    opportunities = []

    for stock in gainers:
        # Check technical indicators
        indicators = obb_provider.get_technical_indicators(stock['symbol'])

        # Check fundamentals
        fundamentals = obb_provider.get_fundamental_data(stock['symbol'])

        # Filter criteria
        if (indicators['rsi'] < 70 and  # Not overbought
            indicators['trend'] == 'bullish' and
            fundamentals['pe_ratio'] < 30):  # Reasonable valuation

            opportunities.append({
                'symbol': stock['symbol'],
                'change': stock['change_percent'],
                'rsi': indicators['rsi'],
                'pe': fundamentals['pe_ratio']
            })

    return opportunities
```

### 3. Risk Monitoring

```python
def monitor_portfolio_risk(portfolio):
    risk_metrics = {}

    for symbol, position in portfolio.items():
        # Get historical data
        df = obb_provider.get_historical_data(symbol, start_date='2023-01-01')

        # Calculate metrics
        returns = df['close'].pct_change()
        volatility = returns.std() * (252 ** 0.5)
        var_95 = returns.quantile(0.05) * position['value']

        risk_metrics[symbol] = {
            'volatility': volatility,
            'var_95': var_95,
            'position_value': position['value']
        }

    return risk_metrics
```

---

## 🔄 Migration from Old Data Providers

### Replacing market_data.py

**Before:**
```python
from utils.market_data import MarketDataFetcher
fetcher = MarketDataFetcher()
data = fetcher.get_historical('AAPL')
```

**After:**
```python
from utils.openbb_provider import get_openbb_provider
obb = get_openbb_provider()
data = obb.get_historical_data('AAPL')
```

### Replacing news_fetcher.py

**Before:**
```python
from utils.news_fetcher import NewsFetcher
fetcher = NewsFetcher()
news = fetcher.get_news('AAPL')
```

**After:**
```python
from utils.openbb_provider import get_openbb_provider
obb = get_openbb_provider()
news = obb.get_news('AAPL')
```

---

## 📊 Performance Comparison

| Feature | Old Providers | OpenBB Platform |
|---------|--------------|-----------------|
| Data Sources | 2-3 APIs | 80+ providers |
| Historical Data | Limited | Full history |
| Technical Indicators | Basic (manual) | 100+ built-in |
| Fundamental Data | Limited | Comprehensive |
| News Sources | 1-2 | Multiple aggregated |
| Options Data | ❌ | ✅ |
| Economic Data | ❌ | ✅ |
| Screening | ❌ | ✅ |
| Cost | Some paid APIs | Free + optional paid |

---

## 🚨 Important Notes

### Rate Limits

- **Free providers** (Yahoo, Alpha Vantage): Have rate limits
- **Paid providers** (FMP, Polygon): Higher limits
- **Best practice:** Cache data, use batch requests

### Data Quality

- OpenBB aggregates from multiple sources
- Cross-validate critical data points
- Use paid providers for production

### Error Handling

```python
try:
    quote = obb_provider.get_quote('AAPL')
    if quote is None:
        # Fallback to alternative provider
        pass
except Exception as e:
    logger.error(f"Error: {e}")
    # Handle error appropriately
```

---

## 🤖 OpenBB Agents Integration

This project integrates **OpenBB Agents** for seamless LLM + Financial Data workflows.

### What are OpenBB Agents?

OpenBB Agents (https://github.com/OpenBB-finance/agents-for-openbb) provides:
- **LangChain Tools** powered by OpenBB data
- **Pre-built Agent Templates** for financial analysis
- **Structured Financial Data** retrieval
- **Best Practices** for AI + Finance integration

### Available Tools

The integration provides **5 powerful LangChain tools**:

1. **OpenBBStockQuoteTool** - Real-time stock quotes
2. **OpenBBFinancialsTool** - Financial metrics and ratios
3. **OpenBBNewsTool** - Recent news articles
4. **OpenBBTechnicalAnalysisTool** - Technical indicators
5. **OpenBBScreenerTool** - Stock screening (gainers/losers/active)

### Quick Start with OpenBB Agents

```python
from utils.openbb_agents import create_openbb_tools, create_financial_analyst_agent
from langchain_anthropic import ChatAnthropic

# Option 1: Use tools directly
tools = create_openbb_tools()

from utils.openbb_agents import OpenBBStockQuoteTool
quote_tool = OpenBBStockQuoteTool()
result = quote_tool._run("AAPL")
print(result)

# Option 2: Create LLM-powered agent
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
agent = create_financial_analyst_agent(llm)

# Ask complex financial questions
response = agent.analyze("Analyze AAPL stock. Should I buy, sell, or hold?")
print(response)
```

### LLM-Powered Financial Analysis

The OpenBB Agents integration allows your LLM to:
- ✅ Access real-time market data
- ✅ Analyze financial statements
- ✅ Monitor news and sentiment
- ✅ Perform technical analysis
- ✅ Screen stocks dynamically
- ✅ Provide data-driven recommendations

### Example: Complete Stock Analysis

```python
from utils.openbb_agents import create_financial_analyst_agent
from langchain_anthropic import ChatAnthropic

# Create agent
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
agent = create_financial_analyst_agent(llm)

# Complex analysis queries
queries = [
    "Analyze AAPL. Consider technical indicators, fundamentals, and recent news.",
    "Compare AAPL vs MSFT. Which is a better investment right now?",
    "Find top 3 gainers today with strong fundamentals",
    "What are the main risks for TSLA based on recent data?"
]

for query in queries:
    response = agent.analyze(query)
    print(f"Query: {query}")
    print(f"Analysis: {response}\n")
```

### Integration with Existing Agents

```python
# Enhance your existing agents with OpenBB tools
from utils.openbb_agents import (
    OpenBBStockQuoteTool,
    OpenBBFinancialsTool,
    OpenBBTechnicalAnalysisTool
)

class EnhancedNewsAgent:
    def __init__(self):
        self.quote_tool = OpenBBStockQuoteTool()
        self.news_tool = OpenBBNewsTool()

    def analyze(self, symbol: str):
        # Get real-time data
        quote = self.quote_tool._run(symbol)
        news = self.news_tool._run(symbol, limit=10)

        # Analyze with your LLM
        # ... your analysis logic ...

        return analysis_result
```

### Custom Financial Research Agent

```python
from utils.openbb_agents import create_openbb_tools

class CustomFinancialResearcher:
    def __init__(self):
        self.tools = create_openbb_tools()

    def research_stock(self, symbol: str):
        research = {}

        # Use all OpenBB tools
        for tool in self.tools:
            if tool.name == "openbb_stock_quote":
                research['quote'] = tool._run(symbol)
            elif tool.name == "openbb_financials":
                research['financials'] = tool._run(symbol)
            # ... etc

        return research
```

### Benefits of OpenBB Agents

| Feature | Without Agents | With OpenBB Agents |
|---------|----------------|-------------------|
| Data Access | Manual API calls | LangChain tool abstraction |
| LLM Integration | Custom code required | Built-in templates |
| Error Handling | Manual implementation | Handled by framework |
| Tool Composition | Manual orchestration | Automatic agent execution |
| Best Practices | Self-developed | Community-maintained |

### Examples

See `examples/openbb_agent_example.py` for comprehensive examples:
- Using OpenBB tools directly
- LLM-powered financial analyst agent
- System integration patterns
- Custom research agent implementation

```bash
# Run examples
python examples/openbb_agent_example.py
```

---

## 📚 Resources

- **OpenBB Platform:** https://github.com/OpenBB-finance/OpenBB
- **OpenBB Agents:** https://github.com/OpenBB-finance/agents-for-openbb
- **Documentation:** https://docs.openbb.co
- **Discord Community:** https://discord.gg/openbb
- **LangChain Docs:** https://python.langchain.com

---

## ✅ Next Steps

1. **Install OpenBB**
   ```bash
   pip install openbb
   ```

2. **Test Integration**
   ```bash
   python -c "from utils.openbb_provider import get_openbb_provider; obb = get_openbb_provider(); print(obb.get_quote('AAPL'))"
   ```

3. **Update Agents**
   - Replace old data fetchers with OpenBB provider
   - Leverage new indicators and data sources
   - Enhance analysis capabilities

4. **Configure Providers**
   - Set up API keys for preferred providers
   - Test data quality and latency
   - Choose optimal provider mix

---

**Status:** ✅ OpenBB Integration Ready

**Benefits:**
- 📊 Professional-grade data
- 🚀 100+ technical indicators
- 💰 Comprehensive fundamentals
- 📰 Multi-source news aggregation
- 📈 Advanced screening & discovery
- 🔒 Enterprise-ready

---

**Last Updated:** 2026-01-05
