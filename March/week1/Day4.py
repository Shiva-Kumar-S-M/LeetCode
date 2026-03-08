#Remove array from sorted lists
class Solution:
    def removeElement(self,nums):
        if len(nums)==0:
            return 0
        
        k=0
        for i in range(len(nums)):
            if nums[i]!=nums[k]:
                k+=1
                nums[k]=nums[i]
        return k+1
    

#Number of zero filled subarrays 
class Solution:
    def zeroFilledSubarray(self,nums):
        res,count=0,0

        for num in nums:
            if num==0:
                count+=1
                res+=count
            else:
                count=0

        return res
    
#Increasing Triplet Subsequence
class Solution:
    def increasingTriplet(self,nums):
        num1=float('inf')
        num2=float('inf')

        for num in nums:
            if num<=num1:
                num1=num
            elif num<=num2:
                num2=num
            else:
                return True
        return False
    
class Solution:
    def numSpecial(self, mat):
        m, n = len(mat), len(mat[0])
        row = [0] * m
        col = [0] * n

        for i in range(m):
            for j in range(n):
                if mat[i][j] == 1:
                    row[i] += 1
                    col[j] += 1

        return sum(
            mat[i][j] == 1 and row[i] == 1 and col[j] == 1
            for i in range(m)
            for j in range(n)
        )
    
class Solution:
    def minOperations(self, s: str) -> int:
        count, n = 0, len(s)
        for i in range(n):
            count += (ord(s[i]) ^ i) & 1
            
        return min(count, n - count)

class Solution:
    def checkOnesSegment(self, s: str) -> bool:
        return "01" not in s
        

class Solution:
    def minFlips(self, s: str) -> int:
        n = len(s)
        res = n
        op = [0, 0]

        for i in range(n):
            op[(ord(s[i]) ^ i) & 1] += 1

        for i in range(n):
            c = ord(s[i])
            op[(c ^ i) & 1] -= 1
            op[(c ^ (n + i)) & 1] += 1
            res = min(res, min(op))

        return res


class Solution:
    def findDifferentBinaryString(self, nums: List[str]) -> str:
        return "".join('1' if x[i]=='0' else '0' for i, x in enumerate(nums))