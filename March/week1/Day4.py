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

class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 1000000007
        maxN = zero + one
        
        fact = [0] * (maxN + 1)
        invFact = [0] * (maxN + 1)
        
        fact[0] = 1
        invFact[0] = 1
        for i in range(1, maxN + 1):
            fact[i] = (fact[i - 1] * i) % MOD
            
        invFact[maxN] = pow(fact[maxN], MOD - 2, MOD)
        for i in range(maxN - 1, 0, -1):
            invFact[i] = (invFact[i + 1] * (i + 1)) % MOD
            
        def C(n, k):
            if k < 0 or k > n:
                return 0
            return fact[n] * invFact[k] % MOD * invFact[n - k] % MOD

        def F(N, K, L):
            if K <= 0 or K > N:
                return 0
            ans = 0
            maxJ = (N - K) // L
            for j in range(maxJ + 1):
                term = C(K, j) * C(N - j * L - 1, K - 1) % MOD
                if j & 1:
                    ans = (ans - term + MOD) % MOD
                else:
                    ans = (ans + term) % MOD
            return ans

        maxK = min(zero, one + 1)
        fOne = [0] * (maxK + 2)
        for k in range(1, maxK + 2):
            fOne[k] = F(one, k, limit)
            
        ans = 0
        for k in range(1, maxK + 1):
            fz = F(zero, k, limit)
            if fz == 0:
                continue
            fo = (fOne[k - 1] + 2 * fOne[k] + fOne[k + 1]) % MOD
            ans = (ans + fz * fo) % MOD
            
        return ans