#Best time to buy and sell Stock 121 easy

from ast import List


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
    
#1536 Minimum swaps to arrange a binary grid medium
class Solution:
    def minSwaps(self, grid: List[List[int]]) -> int:
        n = len(grid)
        zeros = list(map(lambda r: (r[::-1] + [1]).index(1), grid))
        
        swaps = 0
        for i in range(n):
            j = (zeros + [n]).index(next(filter(lambda v: v >= n - 1 - i, zeros + [n])))
            if j == len(zeros): return -1
            swaps += j
            zeros.pop(j)
        return swaps


class Solution:
    def findKthBit(self, n: int, k: int) -> str:
        if n == 1:
            return '0'
        
        length = (1 << n) - 1
        mid = (length + 1) // 2
        
        if k == mid:
            return '1'
        if k < mid:
            return self.findKthBit(n - 1, k)
        
        c = self.findKthBit(n - 1, length - k + 1)
        return '1' if c == '0' else '0'