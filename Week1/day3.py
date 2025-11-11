nums=[10,20,30,40,50,60,70,80,90,100,110]
print(len(nums)//2)


# 169 Given an array nums of size n, return the majority element.

# The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority element always exists in the array.

 

# Example 1:

# Input: nums = [3,2,3]
# Output: 3
# Example 2:

# Input: nums = [2,2,1,1,1,2,2]
# Output: 2

class Solution:
    def majorityElement(self,nums):
        nums.sort()
        return nums[len(nums)//2]


#Another solution using haspmap counting
class Solution:
    def majorityElement(self,nums):
        count={}
        majority=len(nums)//2
        for num in nums:
            if num in count:
                count[num]+=1
            else:
                count[num]=1
            if count[num]>majority:
                return num

        