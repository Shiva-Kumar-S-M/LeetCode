# 209. Minimum Size Subarray Sum medium
from typing import Dict, Tuple
from numpy import inf


class Solution:
    def minSubArrayLen(self,target,nums):
        left,sum=0,0
        minLength=float(inf)

        for right in range (len(nums)):
            sum+=nums[right]

            while sum>=target:
                minLength=min(minLength,right-left+1)
                sum-=nums[left]
                left+=1

        return 0 if minLength==float(inf) else minLength
    

# 1004. Max Consecutive Ones III (Medium)
class Solution:
    def longestOnes(self,nums,k):
        left,right,maxLen,zeros=0,0,0,0
        n=len(nums)

        while right<n:
            if nums[right]==0:
                zeros+=1

            while zeros>k:
                if nums[left]==0:
                    zeros-=1
                left+=1

            maxLen=max(maxLen,right-left+1)
            right+=1

        return maxLen

#53 Maximum Subarray (medium)
class Solution:
    def maxSubArray(self,nums):
        currSum=nums[0]
        maxSum=nums[0]

        for i in range(1,len(nums)):
            currSum=max(nums[i],currSum+nums[i])
            maxSum=max(currSum,maxSum)

        return maxSum
    
    #another approach
class Solution:
    def maxSubArray(self,nums):
        currSum=nums[0]
        maxSum=nums[0]

        for i in range(1,len(nums)):
            if currSum<0:
                currSum=nums[i]
            else:
                currSum+=nums[i]

            if maxSum>currSum:
                maxSum=currSum
        return maxSum
    
# 2414. Length of the Longest Alphabetical Continuous Substring

class Solution:
    def longestContinuousSubstring(self,s):
        maxLen=1
        currLen=1

        for i in range(1,len(s)):
            if ord(s[i])==ord(s[i-1])+1:
                currLen+=1
            else:
                currLen=1

            maxLen=max(maxLen,currLen)

        return maxLen
    

class Solution:
    def mono(self, s: str) -> int:
        if not s:
            return 0
        cnt = 1
        ans = 1
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                cnt += 1
            else:
                cnt = 1
            ans = max(ans, cnt)
        return ans

    def duo(self, s: str, c1: str, c2: str) -> int:
        pos: Dict[int, int] = {0: -1}
        ans = 0
        delta = 0
        for i, ch in enumerate(s):
            if ch != c1 and ch != c2:
                pos.clear()
                pos[0] = i
                delta = 0
                continue

            if ch == c1:
                delta += 1
            else:
                delta -= 1

            if delta in pos:
                ans = max(ans, i - pos[delta])
            else:
                pos[delta] = i

        return ans

    def trio(self, s: str) -> int:
        cnt0 = cnt1 = cnt2 = 0  
        pos: Dict[Tuple[int, int], int] = {(0, 0): -1}
        ans = 0
        for i, ch in enumerate(s):
            if ch == 'a':
                cnt0 += 1
            elif ch == 'b':
                cnt1 += 1
            else:  
                cnt2 += 1

            key = (cnt1 - cnt0, cnt2 - cnt0)

            if key in pos:
                ans = max(ans, i - pos[key])
            else:
                pos[key] = i

        return ans

    def longestBalanced(self, s: str) -> int:
        return max(
            self.mono(s),
            self.duo(s, 'a', 'b'),
            self.duo(s, 'a', 'c'),
            self.duo(s, 'b', 'c'),
            self.trio(s),
        )