class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        cand=0
        count=0

        for num in nums:
            if count==0:
                cand=num
                count=1
            elif num==cand:
                count+=1
            else:
                count-=1

        return cand