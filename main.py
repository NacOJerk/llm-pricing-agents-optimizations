from market_history import *


def main():
    test: PricedProduct = PricedProduct(2, 0.2)
    print(repr(test))

    test2: MarketIteration = MarketIteration([
        PricedProduct(1, 0.4),
        PricedProduct(5, 0.1),
        PricedProduct(42, 0.32),
    ])
    print(test2)

    past_prices = []
    for i in range(5):
        single_price: MarketIteration = MarketIteration([
            PricedProduct(1, 0.4 * i),
            PricedProduct(5, 0.1 * i + 2),
            PricedProduct(42, 0.32 * i + 0.1),
        ])
        past_prices.append(single_price)
    test3: MarketHistory = MarketHistory(past_prices)
    print(test3)

if __name__ == "__main__":
    main()