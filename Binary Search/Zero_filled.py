class Solution:
    def countZeroFilledSubarrays(self,nums):
        count=0
        res=0
        for num in nums:
            if num==0:
                count+=1
                res+=count
            else:
                count=0
        return res


class Solution:
    def canPartitionGrid(self, grid: List[List[int]]) -> bool:
        m, n = len(grid), len(grid[0])
        total = sum(sum(row) for row in grid)
        
        if total % 2:
            return False
        
        target = total // 2
        s = 0
        
        for i in range(m - 1):
            s += sum(grid[i])
            if s == target:
                return True
        
        s = 0
        
        for j in range(n - 1):
            for i in range(m):
                s += grid[i][j]
            if s == target:
                return True
        
        return False