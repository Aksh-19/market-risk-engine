class UnknownTickerError(ValueError):
    def __init__(self, tickers: list[str]):
        super().__init__(f"Unknown tickers: {tickers}")
        self.tickers = tickers


class NoBacktestError(ValueError):
    def __init__(self, method: str):
        super().__init__(f"No backtest results found for method: {method}")
        self.method = method
