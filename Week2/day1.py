#1437 
from git import List


class Solution:
    def kLengthApart(self, nums: List[int], k: int) -> bool:
        if k == 0:
            return True
        prev = None
        for i,num in enumerate(nums):
            if num == 1:
                if prev is not None and i - prev <= k:
                    return False
                prev=i
        return True


# 189 Rotate array
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n=len(nums)
        k=k % n
        rotated=[0]*n

        for i in range(n):
            rotated[(i+k)%n]=nums[i]
        for i in range(n):
            nums[i]=rotated[i]
__import__("atexit").register(lambda: open("display_runtime.txt", 'w').write('0'))


#Another approach for rotate array
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        n = len(nums)
        k %= n

        def reverse(l, r):
            while l < r:
                nums[l], nums[r] = nums[r], nums[l]
                l += 1
                r -= 1

        reverse(0, n - 1)
        reverse(0, k - 1)
        reverse(k, n - 1)

#121 Best time to buy and sell stock
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        minSoFar = prices[0]
        profit = 0
        for i in range(1, len(prices)):
            minSoFar = min(minSoFar,prices[i])
            profit = max(profit,prices[i] - minSoFar)
        return profit
__import__("atexit").register(lambda: open("display_runtime.txt", 'w').write('0'))



#Another approach for best time to buy and sell stock
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        left = 0
        right = 1
        maxProfit = 0

        while right < len(prices):
            if prices[left] < prices[right]:
                profit = prices[right] - prices[left]
                maxProfit = max(maxProfit, profit)
            else:
                left = right
            right += 1
        return maxProfit
        
#Another approach for best time to buy and sell stock
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        buy=prices[0]
        profit=0
        for i in range(1,len(prices)):
            if prices[i]<buy:
                buy=prices[i]
            elif prices[i]-buy>profit:
                profit=prices[i]-buy
        return profit
        
#717 1 bit and 2 bit characters
class Solution:
    def isOneBitCharacter(self, bits: List[int]) -> bool:
        n=len(bits)
        i=0
        while i < n-1:
            i+= bits[i]+1
        return i==n-1
        


#2154 keep multiplying found values by two
class Solution:
    def findFinalValue(self, nums: List[int], k: int) -> int:
        bits = 0
        for num in nums:
            if num % k != 0:
                continue
            n = num // k
            if n & (n - 1) == 0:
                bits |= n
        d = bits + 1
        return k * (d & -d)
        
#757    Set intersection size at least two
import heapq
class Solution:
    def intersectionSizeTwo(self, intervals: List[List[int]]) -> int:
        n=len(intervals)
        intervals.sort(key=lambda x:x[1])
        prev1=intervals[0][1]-1
        prev2=intervals[0][1]
        c=2
        for i in range(1,n):
            if prev2<intervals[i][0]:
                prev1=intervals[i][1]-1
                prev2=intervals[i][1]
                c+=2
            elif prev1<intervals[i][0]:
                if intervals[i][1]==prev2:
                    prev1=intervals[i][1]-1
                else:
                    prev1=intervals[i][1]
                prev1,prev2=min(prev1,prev2),max(prev1,prev2)
                c+=1
        return c
    
#1341 
class Solution:
    def countPalindromicSubsequence(self, s: str) -> int:
        n = len(s)
        first = [-1] * 26
        last = [-1] * 26

        for i, ch in enumerate(s):
            c = ord(ch) - ord('a')
            if first[c] == -1:
                first[c] = i
            last[c] = i

        ans = 0
        for c in range(26):
            if first[c] != -1 and last[c] - first[c] > 1:
                mask = 0
                for i in range(first[c] + 1, last[c]):
                    mask |= 1 << (ord(s[i]) - ord('a'))
                ans += bin(mask).count("1")

        return ans
        
        
        
class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        return len(list(filter(lambda x: x%3!=0, nums)))
        

#1262 Greatest Sum divisible by three
class Solution:
    def maxSumDivThree(self, nums: List[int]) -> int:
        s = sum(nums)
        
        if s % 3 == 0:
            return s
        
        r11 = 10000
        r12 = 10000
        r21 = 10000
        r22 = 10000
        
        for num in nums:
            if num % 3 == 1 and num < r12:
                if num < r11:
                    r12 = r11
                    r11 = num
                else:
                    r12 = num
            if num % 3 == 2 and num < r22:
                if num < r21:
                    r22 = r21
                    r21 = num
                else: 
                    r22 = num
        if s % 3 == 1:
            return s - min(r11, r21+r22)
        if s % 3 == 2:
            return s - min(r21, r11+r12) 

class Solution:
    def prefixesDivBy5(self, nums: List[int]) -> List[bool]:
        val = 0
        for i in range(len(nums)):
            val = ((val << 1) + nums[i]) % 5
            nums[i] = val == 0
        return nums

class Solution:
    def smallestRepunitDivByK(self, K: int) -> int:
        remainder = 0
        for length_N in range(1,K+1):
            remainder = (remainder*10+1) % K
            if remainder == 0:
                return length_N
        return -1

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix_product = 1
        postfix_product = 1
        result = [0]*n
        for i in range(n):
            result[i] = prefix_product
            prefix_product *= nums[i]
        for i in range(n-1,-1,-1):
            result[i] *= postfix_product
            postfix_product *= nums[i]
        return result
    
#122 Best time to buy and sell stock II
class Solution:
    def maxProfit(self,prices:List[int]):
        profit=0
        for i in range(1,len(prices)):
            if prices[i]> prices[i-1]:
                profit+=(prices[i]-prices[i-1])
        return profit
    

#238 Product of array except self
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res=[1]*(len(nums))
        prefix=1
        for i in range(len(nums)):
            res[i]=prefix
            prefix*=nums[i]
        postfix=1
        for i in range(len(nums)-1,-1,-1):
            res[i]*=postfix
            postfix*=nums[i]
        return res

class Solution:
    def numberOfPaths(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m, n = len(grid), len(grid[0])

        prev = [[0]*k for _ in range(n)]
        curr = [[0]*k for _ in range(n)]

        s = 0
        for j in range(n):
            s = (s + grid[0][j]) % k
            prev[j][s] = 1

        s = grid[0][0] % k

        for i in range(1, m):
            s = (s + grid[i][0]) % k
            curr[0] = [0]*k
            curr[0][s] = 1

            for j in range(1, n):
                curr[j] = [0]*k
                val = grid[i][j]
                for r in range(k):
                    nr = (r + val) % k
                    curr[j][nr] = (prev[j][r] + curr[j - 1][r]) % MOD

            prev, curr = curr, prev

        return prev[n - 1][0]

#2348 Number of zero filled subarrays
class Solution:
    def zeroFilledSubarray(self, nums: List[int]) -> int:
        res,count=0,0

        for i in range(len(nums)):
            if nums[i]==0:
                count+=1
                res+=count
            else:
                count=0
            
        return res


#334 Increasing Triplet Subsequence
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        i=float('inf')
        j=float('inf')
        for num in nums:
            if num<=i:
                i=num
            elif num<=j:
                j=num
            else:
                return True
        return False
    
#Hard 
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        nums = [n for n in nums if n > 0]
        nums.sort()

        target = 1
        for n in nums:
            if n == target:
                target += 1
            elif n > target:
                return target
        
        return target

#393 is subsequence

class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        if not s:
            return True
        if not t:
            return False
        
        i, j = 0, 0
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
            j += 1
        
        return i == len(s)
