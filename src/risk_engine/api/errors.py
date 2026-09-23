class UnknownTickerError(ValueError):
    def __init__(self, tickers: list[str]):
        super().__init__(f"Unknown tickers: {tickers}")
        self.tickers = tickers
