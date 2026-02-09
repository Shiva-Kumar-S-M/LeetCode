#Single number 2> 137
from ast import List
from collections import defaultdict
from typing import Optional, Tuple


class Solution:
    def SingleNumber(self,nums):
        ones,twos=0,0
        for num in nums:
            ones=(ones^num) & ~twos
            twos=(twos^num) & ~ones
        return ones
    

#Single number 3.. 260

class Solution:
    def single(self,nums):
        xor=0
        for num in nums:
            xor^=num

        diff=xor & -xor
        res=[0,0]
        for num in nums:
            if num & diff:
                res[0]^=num

            else:
                res[1]^=num

        return res
    
#242 valid Anagram
class Solution:
    def isAnagram(self,s,t):
        return sorted(s)==sorted(t)
    

#Another approch:
class Solution:
    def isAnagram(self,s,t):
        if len(s)!=len(t):
            return False
        
        for i in set(s):
            if s.count(i)!=t.count(i):
                return False
        return True
    
#49 group Anagrams
class Solution:
    def groupAnagrams(self,strs):
        res=defaultdict(list)

        for i in strs:
            key="".join(sorted(i))
            res[key].append(i)
        return list(res.values())
        

#Trionic array  978
class Solution:
    def decompose(self, nums: List[int]) -> List[Tuple[int, int, int]]:
        n = len(nums)
        subarrays: List[Tuple[int, int, int]] = []

        l = 0
        s = nums[0]

        for i in range(1, n):
            # If we fail strict decreasing at boundary i-1 -> i, end the current subarray.
            if nums[i - 1] <= nums[i]:
                subarrays.append((l, i - 1, s))
                l = i
                s = 0
            s += nums[i]

        # last subarray
        subarrays.append((l, n - 1, s))
        return subarrays

    def maxSumTrionic(self, nums: List[int]) -> int:
        n = len(nums)

        maxEndingAt = [0] * n
        for i in range(n):
            maxEndingAt[i] = nums[i]
            if i > 0 and nums[i - 1] < nums[i]:
                if maxEndingAt[i - 1] > 0:
                    maxEndingAt[i] += maxEndingAt[i - 1]

        maxStartingAt = [0] * n
        for i in range(n - 1, -1, -1):
            maxStartingAt[i] = nums[i]
            if i < n - 1 and nums[i] < nums[i + 1]:
                if maxStartingAt[i + 1] > 0:
                    maxStartingAt[i] += maxStartingAt[i + 1]

        PQS = self.decompose(nums)
        ans = -10**30  
        for (p, q, s) in PQS:
            if (p > 0 and nums[p - 1] < nums[p] and
                q < n - 1 and nums[q] < nums[q + 1] and
                p < q):
                cand = maxEndingAt[p - 1] + s + maxStartingAt[q + 1]
                if cand > ans:
                    ans = cand

        return ans

class Solution:
    def constructTransformedArray(self, A: List[int]) -> List[int]:
        return [A[(i + v) % len(A)] for i, v in enumerate(A)]
    
print(Solution().constructTransformedArray([0,1,2,4,5,6,7,8,9]))

class Solution:
    def minRemoval(self, nums: List[int], k: int) -> int:
        nums.sort()
        i = 0
        max_len = 0
        
        for j in range(len(nums)):
            while nums[j] > nums[i] * k:
                i += 1
            max_len = max(max_len, j - i + 1)
            
        return len(nums) - max_len


class Solution:
    def minimumDeletions(self, s: str) -> int:
        res = len(s)
        a = 0
        b = 0

        for c in s:
            a += (c == 'a')

        for c in s:
            a -= (c == 'a')
            res = min(res, a + b)
            b += (c == 'b')

        return res

#py3
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool: # type: ignore
        def dfs(node):
            if not node:
                return 0
            
            left = dfs(node.left)
            if left == -1: return -1
            
            right = dfs(node.right)
            if right == -1: return -1
            
            if abs(left - right) > 1:
                return -1
            
            return 1 + max(left, right)
            
        return dfs(root) != -1
    

#121 best time to buy and sell stock:
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

        return max
    


#122 best time to buys and sell stock 2
class Solution:
    def maxProfit(self,prices):
        profit=0

        for i in range(1,len(prices)):
            if prices[i]>prices[i-1]:
                profit+=prices[i]-prices[i-1]
        return profit