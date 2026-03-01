#Move Zeros (283)Easy

from git import List


class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        list=0
        for i in range(len(nums)):
            if nums[i]!=0:
                nums[list],nums[i]=nums[i],nums[list]
                list+=1

        return list
        