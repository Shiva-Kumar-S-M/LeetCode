class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        return ((x:=heapq.nlargest(2, nums))[0]-1)*(x[1]-1)