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
        


#Rotate array(189)Medium

class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n=len(nums)
        k%=n

        def rev(l,r):
            while l<r:
                nums[l],nums[r]=nums[r],nums[l]
                l+=1
                r-=1

        rev(0,n-1)
        rev(0,k-1)
        rev(k,n-1)

        
#1689 parititioning into number of decimal binary number

class Solution:
    def minPartitions(self, n: str) -> int:
        return int(max(n))