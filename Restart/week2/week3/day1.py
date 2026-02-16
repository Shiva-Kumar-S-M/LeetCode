#1047 remove all adjacent duplicates in string (easy)
class Solution:
    def removeDuplicates(self,s):
        list=[]
        for i in s:
            if list and list[-1]==i:
                list.pop()
            else:
                list.append(i)

        return ''.join(list)
    
#15 3 sum (medium)
class Solution:
    def threeSum(self,nums):
        res=[]
        nums.sort()

        for i in range(len(nums)-2):
            if i==0 or nums[i]!=nums[i-1]:
                left,right=i+1,len(nums)-1

                while left<right:
                    s=nums[i]+nums[left]+nums[right]

                    if s==0:
                        res.append([nums[i],nums[left],nums[right]])

                        while left<right and nums[left]==nums[left+1]:
                            left+=1

                        while left<right and nums[right]==nums[right-1]:
                            right-=1

                        left+=1
                        right-=1

                    elif s<0:
                        left+=1

                    else:
                        right-=1

        return res
    
#67 add binary(easy)

class Solution:
    def addBinary(self, a, b) -> str:
        x, y = int(a, 2), int(b, 2)
        while y:
            x, y = x ^ y, (x & y) << 1
        return bin(x)[2:]





               