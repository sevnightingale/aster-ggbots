# Aster Vibe Trader - Autonomous AI Trading System

**Submission for Aster Vibe Trading Arena Competition**

An autonomous AI trading system that executes live trades on AsterDEX using multi-model AI reasoning, comprehensive market intelligence, and professional-grade execution engines.

**Live Demo**: [aster.ggbots.ai](https://aster.ggbots.ai)
**Platform**: [ggbots.ai](https://ggbots.ai) - Multi-user trading bot platform (see [PLATFORM.md](PLATFORM.md))

---

## What Makes This Different

### 1. True Autonomous AI Reasoning
Not just rule-based triggers - actual AI models analyzing market conditions and making trading decisions:
- **Multi-model support**: GPT-5, Claude Opus 4, Grok 4, DeepSeek R1
- **Real-time reasoning**: Live market analysis with explainable decision-making
- **Adaptive strategies**: AI learns from market conditions and adjusts approach

### 2. Three-Agent Architecture

```
Market Data → Extraction Agent → Decision Agent → Trading Agent → AsterDEX
     ↑              ↓               ↓              ↓             ↓
   Sources    Intelligence    AI Reasoning    Execution      Results
```

**Extraction Agent**: 21 technical indicators with pandas-ta integration
- RSI, MACD, Bollinger Bands, ATR, Stochastic, ADX, and 15 more
- Multi-timeframe analysis (5m, 15m, 30m, 1h, 4h, 1d)
- Real-time WebSocket price feeds from Binance

**Decision Agent**: AI-powered trading intelligence
- Template-based prompts for opportunity analysis and position management
- Confidence scoring and risk-aware position sizing
- Natural language strategy customization

**Trading Agent**: Professional execution on AsterDEX
- 33 supported trading pairs with up to 20x leverage
- Dynamic position sizing based on AI confidence
- Real-time monitoring with automatic TP/SL execution
- Agent override support for autonomous position control

### 3. Comprehensive Market Intelligence
32 data points across 7 categories:
- **Technical Analysis**: 21 pure Python indicators (no external dependencies)
- **On-Chain Analytics**: BTC TVL, whale activity (via Grok)
- **Derivatives & Leverage**: BTC/ETH funding rates
- **Sentiment & Social**: Twitter sentiment analysis
- **News & Regulatory**: Crypto news aggregation
- **Macro Economics**: VIX, DXY, CPI, NFP data

### 4. Professional Dashboard
Real-time activity timeline showing:
- Trade entries and exits with reasoning
- AI decision-making process visualization
- Performance metrics and P&L tracking
- Strategy configuration and market data selection

**Live at**: [aster.ggbots.ai](https://aster.ggbots.ai)

---

## AsterDEX Integration

### Supported Features
- **33 trading pairs**: All major crypto futures with USDT settlement
- **Up to 20x leverage**: Configurable leverage per trade
- **Dynamic position sizing**: AI-driven position size calculation
- **Web3 authentication**: ECDSA signature-based Pro API access
- **Real-time execution**: Sub-second order placement and monitoring

### Technical Implementation
Located in `trading/live/aster_service_v3.py`:
- Smart account integration with Aster Pro API
- Position size and leverage override support (agent can control these dynamically)
- Comprehensive error handling and logging
- Market order execution with automatic SL/TP placement

### Environment Setup
```bash
# Required in .env
ASTER_PRIVATE_KEY=<your_web3_private_key>
ASTER_SMART_ACCOUNT_ADDRESS=<your_aster_smart_account>
ASTER_API_URL=https://pro-api.aster.xyz
```

---

## Architecture Overview

### Core Components

**ggbot.py** - V2 Orchestrator
- FastAPI server with autonomous scheduling (APScheduler)
- Sequential execution: Extraction → Decision → Trading
- Real-time WebSocket updates for frontend
- Redis-based idempotency to prevent duplicate trades

**core/** - Shared Infrastructure
- Domain models with business logic (`position.py`, `decision.py`, `user_profile.py`)
- Symbol standardization across 141 trading pairs
- Database repositories for Supabase PostgreSQL
- LLM service with multi-provider support

**extraction/v2/** - Market Intelligence Engine
- 21 technical indicator preprocessors
- WebSocket-first architecture for real-time prices
- Universal data layer with catalog-driven routing
- Supabase database storage

**decision/** - AI Decision Engine
- Template-based prompt system (opportunity analysis, position management)
- Multi-LLM provider support (OpenAI, Anthropic, XAI, DeepSeek, Google)
- Confidence scoring and position sizing algorithms
- Real-time price integration

**trading/** - Execution Engines
- **Paper trading**: Risk-free testing with $10k virtual accounts
- **AsterDEX**: Live futures trading with Web3 auth
- Real-time position monitoring with 3-second P&L updates
- Automated TP/SL execution

**frontend/** - Next.js 15 Dashboard
- Activity Timeline Viewer (`/view/[config_id]`) - Competition submission page
- Forge Platform - Full multi-user trading bot configuration
- Real-time SSE streams for live updates
- Supabase auth with JWT token-based API access

**agent/** - Autonomous Agent System (Phase 4a)
- MCP (Model Context Protocol) server
- Tool-based trading execution
- Strategy definition and autonomous modes
- Redis-based message queuing

---

## Tech Stack

### Backend
- **Python 3.10** - Core backend language
- **FastAPI 0.115** - REST API framework
- **APScheduler 3.11** - Autonomous bot scheduling
- **pandas-ta 0.3.14** - Technical indicators
- **PostgreSQL** (Supabase) - Main database
- **Redis** - WebSocket cache, queues, idempotency

### AI/LLM
- **OpenAI** (GPT-4, GPT-5 Responses API)
- **Anthropic** (Claude Haiku 4.5, Sonnet 4.5, Opus 4)
- **XAI** (Grok 4 Agentic API)
- **Google** (Gemini models)
- **DeepSeek** (R1 reasoning model)

### Frontend
- **Next.js 15.3** - React framework with App Router
- **React 19** - UI library
- **TypeScript 5.x** - Type safety
- **Tailwind CSS** - Styling
- **Vercel** - Production deployment

### Trading & Data
- **AsterDEX Pro API** - Live futures execution
- **Binance WebSocket** - Real-time price feeds
- **CCXT 4.4** - Multi-exchange library

---

## Running the System

### Prerequisites
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install
```

### Environment Variables
Required in `.env`:
```bash
# Database
SUPABASE_URL=<your_supabase_url>
SUPABASE_SERVICE_KEY=<your_supabase_service_key>

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# AsterDEX (for live trading)
ASTER_PRIVATE_KEY=<your_web3_private_key>
ASTER_SMART_ACCOUNT_ADDRESS=<your_aster_smart_account>
ASTER_API_URL=https://pro-api.aster.xyz

# LLM Providers (at least one required)
OPENAI_API_KEY=<your_openai_key>
ANTHROPIC_API_KEY=<your_anthropic_key>
XAI_API_KEY=<your_xai_key>
```

### Start Services

**Backend (V2 Orchestrator)**:
```bash
python ggbot.py
# Runs on http://localhost:8000
```

**Frontend**:
```bash
cd frontend
npm run dev
# Runs on http://localhost:3000
```

---

## Competition Highlights

### What the Judges See at [aster.ggbots.ai](https://aster.ggbots.ai)

1. **Real-time Activity Timeline**
   - Visual scroll-through-time interface
   - Trade entries/exits with AI reasoning
   - Performance metrics and balance tracking
   - Zoom tiers (1h/4h/1d/1w/All)

2. **Transparent Decision-Making**
   - Every decision shows the AI's reasoning
   - Confidence scores and risk assessment
   - Market data that influenced each decision

3. **Live Trading Performance**
   - Real trades executed on AsterDEX
   - Actual P&L from live market conditions
   - Position management and risk controls

### Why This System Wins

- **Actual AI reasoning** (not hardcoded rules)
- **Multi-model support** (use the best model for the job)
- **Production-grade execution** (real money, real trades)
- **Comprehensive market intelligence** (32+ data points)
- **Professional dashboard** (transparency into AI decisions)
- **Extensible platform** (ggbots.ai lets anyone build their own vibe trader)

---

## Repository Structure

```
aster-ggbots/
├── README.md              # This file - Competition overview
├── PLATFORM.md            # ggbots platform documentation (bonus)
├── requirements.txt       # Python dependencies
├── ggbot.py              # V2 Orchestrator (main API server)
│
├── agent/                # Autonomous agent system (MCP server, tools, chat)
├── extraction/           # Market intelligence with 21 preprocessors
├── decision/             # AI reasoning engine with multi-LLM support
├── trading/              # Paper & live execution (Aster integration)
├── core/                 # Domain models, services, infrastructure
├── api/                  # REST endpoints (activities, positions, etc.)
└── frontend/             # Next.js dashboard and activity timeline
```

---

## Learn More

- **Live Demo**: [aster.ggbots.ai](https://aster.ggbots.ai)
- **Platform**: [ggbots.ai](https://ggbots.ai) - Build your own vibe trader
- **Platform Documentation**: See [PLATFORM.md](PLATFORM.md) for full system architecture

---

## Competition Submission

**Team**: ggbots
**Primary Contact**: Sev Nightingale sevnightingale@gmail.com
**Dashboard URL**: https://aster.ggbots.ai
**Platform URL**: https://ggbots.ai
**GitHub**: https://github.com/[sevnightingale]/aster-ggbots

Built with autonomous AI reasoning for the Aster Vibe Trading Arena.
Let's vibe. Let's trade. Let's build the future.
