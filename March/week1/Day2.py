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
    

#122 Best time to buy and sell stock II medium
class Solution:
    def maxProfit(self,prices):
        max=0

        for i in range(1,len(prices)):
            if prices[i]>prices[i-1]:
                max+=prices[i]-prices[i-1]

        return max