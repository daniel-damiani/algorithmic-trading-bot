# QuantumSentiment - Core Directives & Project Context

## Project Snapshot
- **Project:** QuantumSentiment AI Trading System
- **Objective:** Build a fully functional, production-ready trading bot.
- **Broker:** Alpaca
- **Current Mode:** Paper Trading (Live trading is a future goal)
- **Budget Constraint:** Designed for a small capital base (€1,000)

## Prime Directive
Your prime directive is to execute the mandatory refactoring plan detailed in **`TODO.md`**. This document contains the full system analysis, identifies all critical flaws, and provides the step-by-step mandate for making the system functional. 

## Critical Safety Rules
1.  **Paper Trading First:** All development and testing must target paper trading mode.
2.  **No Hardcoded Keys:** All secrets must be loaded from `.env`.
3.  **Risk Management is Paramount:** Ensure all risk controls (drawdown, stop-loss, position limits) are functional and cannot be bypassed.

## Guiding Principles
- **Follow the Plan:** Execute the refactoring detailed in `TODO.md`.
- **Update `NOTES.md`:** After completing a significant task, log the changes, decisions, and rationale in `NOTES.md` in the required academic style.
- **Maintain Quality:** Adhere to the modular architecture. Write clean, documented code with full type hinting.