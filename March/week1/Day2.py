#Best time to buy and sell Stock 121 easy

class Solution:
    def maxProfit(self,prices):
        min=float('inf')
        max=0

        for price in prices:
            if price<min:
                min=price

            profit=price-min

            if profit>max:
                max=profit

        return profit