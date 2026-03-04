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