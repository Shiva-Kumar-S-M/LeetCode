#Remove array from sorted lists
class Solution:
    def removeElement(self,nums):
        if len(nums)==0:
            return 0
        
        k=0
        for i in range(len(nums)):
            if nums[i]!=nums[k]:
                k+=1
                nums[k]=nums[i]
        return k+1
    

#Number of zero filled subarrays 
class Solution:
    def zeroFilledSubarray(self,nums):
        res,count=0,0

        for num in nums:
            if num==0:
                count+=1
                res+=count
            else:
                count=0

        return res