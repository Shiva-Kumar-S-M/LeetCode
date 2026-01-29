#283 Move Zeros 
from typing import List
class Solution:
    def moveZeros(self,nums:List[int]):
        list=0
        for i in range (len(nums)):
            if nums[i]!=0:
                nums[list],nums[i]=nums[i],nums[list]
                list+=1
        return nums
    

#169 Majority element
class Solution:
    def majorityElements(self,nums:{List[int]}):
        nums.sort()
        return nums[len(nums)//2]
    

class Solution:
    def majorityElement(self,nums):
        candidate=nums[0]
        count=0
        for num in nums:
            if count==0:
                candidate=num
            count+=1 if num==candidate else -1
        return candidate

class Solution:
    def minimumCost(self, source: str, target: str, original: list[str], changed: list[str], cost: list[int]) -> int:
        inf = float('inf')
        dist = [[inf] * 26 for _ in range(26)]

        for i in range(26):
            dist[i][i] = 0

        for o, c, z in zip(original, changed, cost):
            u = ord(o) - 97
            v = ord(c) - 97
            dist[u][v] = min(dist[u][v], z)

        for k in range(26):
            for i in range(26):
                if dist[i][k] == inf:
                    continue
                for j in range(26):
                    if dist[k][j] != inf:
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

        total_cost = 0
        for s_char, t_char in zip(source, target):
            u = ord(s_char) - 97
            v = ord(t_char) - 97
            if u == v:
                continue
            if dist[u][v] == inf:
                return -1
            total_cost += dist[u][v]

        return total_cost