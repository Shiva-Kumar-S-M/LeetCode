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
    
    