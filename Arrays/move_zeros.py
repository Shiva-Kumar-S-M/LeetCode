class Solution:
    def moveZeros(self,nums):
        list=0
        for i in range(len(nums)):
            if nums[i]!=0:
                nums[list],nums[i]=nums[i],nums[list]
                list+=1

        return list