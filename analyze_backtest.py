import asyncio
from src.backtesting.backtest_runner import BacktestRunner

async def detailed_backtest():
    runner = BacktestRunner()
    results = await runner.run_backtest(
        symbols=['AAPL', 'TSLA', 'NVDA'], 
        start_date='2024-02-01', 
        end_date='2024-02-29', 
        initial_capital=10000
    )
    
    print('=== DETAILED PERFORMANCE REPORT ===')
    
    # Extract full report
    full_report = results.get('full_report', {})
    
    # Print all available metrics
    print('\nAVAILABLE METRICS:')
    for key, value in results.items():
        if key != 'full_report':
            print(f'{key}: {value}')
    
    # Print detailed report sections
    if full_report:
        print('\nFULL REPORT SECTIONS:')
        for section, data in full_report.items():
            print(f'\n{section.upper()}:')
            if isinstance(data, dict):
                for k, v in data.items():
                    print(f'  {k}: {v}')
            else:
                print(f'  {data}')
    
    return results

if __name__ == "__main__":
    asyncio.run(detailed_backtest())