# 209. Minimum Size Subarray Sum medium
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