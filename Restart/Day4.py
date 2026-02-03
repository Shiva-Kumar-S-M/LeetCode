#125 valid palindrome
class Solution:
    def isPalindrome(self,s):
        list=[]
        for i in s:
            if i.isalnum():
                list.append(i.lower())
        return list==list[::-1]


#136 Single number
class Solution:
    def singleNumber(self,nums):
        res=0
        for i in nums:
            res^=i
        return res
    


#191 Single number of 1 bits

class Solution:
    def Hamingweight(self,n):
        res=0
        for i in range(32):
            if (n>>i)&1:
                res+=1
        return res
    

#338 Counting bits
class Solution:
    def bitCount(self,n):
        res=[0]*(n+1)
        for i in range(n+1):
            res[i]=res[i>>1]+(i&1)
        return res
    

class solution:
    def bitCounts(self,n):
        res=[]
        for i in range(n+1):
            res.append(i.bit_count())
        return res
    
