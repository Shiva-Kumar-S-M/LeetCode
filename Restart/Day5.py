#Single number 2> 137
class Solution:
    def SingleNumber(self,nums):
        ones,twos=0,0
        for num in nums:
            ones=(ones^num) & ~twos
            twos=(twos^num) & ~ones
        return ones
    

#Single number 3.. 260

class Solution:
    def single(self,nums):
        xor=0
        for num in nums:
            xor^=num

        diff=xor & -xor
        res=[0,0]
        for num in nums:
            if num & diff:
                res[0]^=num

            else:
                res[1]^=num

        return res
    
#242 valid Anagram
class Solution:
    def isAnagram(self,s,t):
        return sorted(s)==sorted(t)
    

#Another approch:
class Solution:
    def isAnagram(self,s,t):
        if len(s)!=len(t):
            return False
        
        for i in set(s):
            if s.count(i)!=t.count(i):
                return False
        return True