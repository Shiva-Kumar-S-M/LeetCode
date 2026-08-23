class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        cand=nums[0]
        count=1
        n=len(nums)

        for i in range(1,n):
            if count==0:
                cand=nums[i]
                count=1
            elif nums[i]==cand:
                count+=1
            else:
                count-=1
        return cand
        