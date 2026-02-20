#Maximum SubArray(medium) kadane's algorithm
class Solution:
    def maxSubArray(self,nums):
        maxSum=nums[0]
        currSum=nums[0]

        for i in range(1,len(nums)):
            if currSum<0:
                currSum=nums[i]

            else:
                currSum+=nums[i]

            if currSum>maxSum:
                maxSum=currSum

        return maxSum
    

#152 Maximum Product SubArray(medium) kadane's algorithm

class Solution:
    def maxProduct(self,nums):
        maxProd=nums[0]
        minProd=nums[0]
        res=maxProd

        for i in range(1,len(nums)):
            curr=nums[i]
            temp_max=max(curr,maxProd*curr,minProd*curr)
            minProd=min(curr,maxProd*curr,minProd*curr)
            maxProd=temp_max
            res=max(res,maxProd)

        return res
    
#918 Maximum Sum Circular SubArray(medium) kadane's algorithm
class Solution:
    def maxSubArray(self,nums):
        maxSum=float('-inf')
        curr_max=0
        minSum=float('inf')
        curr_min=0
        total=0

        for num in nums:
            total+=num
            curr_max=max(num,curr_max+num)
            maxSum=max(curr_max,maxSum)

            curr_min=min(num,curr_min+num)
            minSum=min(curr_min,minSum)

        if total<0:
            return maxSum
        
        return max(maxSum,total-minSum)
    
