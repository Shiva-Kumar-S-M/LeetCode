class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        ans=0
        for i in range(len(nums)):
            if nums[i]!=0:
                nums[i],nums[ans]=nums[ans],nums[i]
                ans+=1
        return ans