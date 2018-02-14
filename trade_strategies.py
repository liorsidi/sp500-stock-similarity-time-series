def simple_profit_evaluation(curr_price, predicted_price):
    """
    a profit evaluation that buys if stock goes up and sell of goes down
    :param curr_price:
    :param predicted_price:
    :return:
    """
    in_position = False
    profit = 0
    last_buy = 0
    profits = []
    for i in range(len(curr_price)):
        if predicted_price[i] > 0 and not in_position:
            in_position = True
            last_buy = curr_price[i]
        if predicted_price[i] < 0 and in_position:
            in_position = False
            profit = profit + (curr_price[i] - last_buy)
        if in_position:
            profits.append(profit + (curr_price[i] - last_buy))
        else:
            profits.append(profit)
    return profit, profits


def long_short_profit_evaluation(curr_price, predicted_price):
    """
    a profit evaluation that buys long if stock goes up and buys short of goes down
    :param curr_price:
    :param predicted_price:
    :return:
    """
    is_long = None
    profit = 0
    last_buy = 0
    profits = []
    position = 0
    for i in range(len(curr_price)):
        # go long
        if predicted_price[i] > 0:
            # first time
            if is_long is None:
                last_buy = curr_price[i]
                is_long = True
            # if short position - close it and go long
            elif not is_long:
                profit = profit + (last_buy - curr_price[i])
                position = profit
                last_buy = curr_price[i]
                is_long = True
            elif is_long:
                position = profit + (curr_price[i] - last_buy)

        # go short
        if predicted_price[i] < 0:
            # first time
            if is_long is None:
                last_buy = curr_price[i]
                is_long = False
            # if long position - close it and go short
            elif is_long:
                profit = profit + (curr_price[i] - last_buy)
                position = profit
                last_buy = curr_price[i]
                is_long = False
            elif not is_long:
                position = profit + (last_buy - curr_price[i])

        profits.append(position)

    return profit, profits

