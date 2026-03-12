#Maximum SubArray(medium) kadane's algorithm
from ast import Dict, List


class Solution:
    def maxSubArray(self,nums):
        maxSum=nums[0]
        currSum=nums[0]

        for i in range(1,len(nums)):
            if currSum<0:
                currSum=nums[i]

            else:
                currSum+=nums[i]

            if currSum>maxSum:
                maxSum=currSum

        return maxSum
    

#152 Maximum Product SubArray(medium) kadane's algorithm

class Solution:
    def maxProduct(self,nums):
        maxProd=nums[0]
        minProd=nums[0]
        res=maxProd

        for i in range(1,len(nums)):
            curr=nums[i]
            temp_max=max(curr,maxProd*curr,minProd*curr)
            minProd=min(curr,maxProd*curr,minProd*curr)
            maxProd=temp_max
            res=max(res,maxProd)

        return res
    
#918 Maximum Sum Circular SubArray(medium) kadane's algorithm
class Solution:
    def maxSubArray(self,nums):
        maxSum=float('-inf')
        curr_max=0
        minSum=float('inf')
        curr_min=0
        total=0

        for num in nums:
            total+=num
            curr_max=max(num,curr_max+num)
            maxSum=max(curr_max,maxSum)

            curr_min=min(num,curr_min+num)
            minSum=min(curr_min,minSum)

        if total<0:
            return maxSum
        
        return max(maxSum,total-minSum)
    

class Solution:
    def makeLargestSpecial(self, s: str) -> str:
        if s == '':
            return ''
        ans = []
        cnt = 0
        i = j = 0
        while i < len(s):
            cnt += 1 if s[i] == '1' else -1
            if cnt == 0:
                ans.append('1' + self.makeLargestSpecial(s[j + 1 : i]) + '0')
                j = i + 1
            i += 1
        ans.sort(reverse=True)
        return ''.join(ans)


class Solution:
    def countPrimeSetBits(self, left: int, right: int) -> int:
        count = 0
        
        for i in range(left, right + 1):
            setBits = bin(i).count('1')  # Convert to binary and count '1's
            if self.isPrime(setBits):
                count += 1
        
        return count
    
    def isPrime(self, n: int) -> bool:
        if n <= 1:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True


class Solution:
    def binaryGap(self, n: int) -> int:
        n //= n & -n
        if n == 1: return 0

        max_gap = 0
        gap = 0

        while n:
            if n & 1:
                max_gap = max(max_gap, gap)
                gap = 0
            else:
                gap += 1
            n >>= 1

        return max_gap + 1
    
MOD = 1_000_000_007
MAXN = 1000
fact = [0] * (MAXN + 1)
invfact = [0] * (MAXN + 1)

def init():
    fact[0] = 1
    for i in range(1, MAXN + 1):
        fact[i] = (fact[i - 1] * i) % MOD
    invfact[MAXN] = pow(fact[MAXN], MOD - 2, MOD)
    for i in range(MAXN, 0, -1):
        invfact[i - 1] = (invfact[i] * i) % MOD

init()

class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        if zero > one:
            zero, one = one, zero

        if limit == 1:
            if zero == one: return 2
            if zero + 1 == one: return 1
            return 0

        def ncr(n: int, r: int) -> int:
            return fact[n] * invfact[r] * invfact[n - r]

        def ways(n: int, k: int) -> int:
            if n == k: return 1
            j, total, flag = 0, 0, True
            while j <= k <= n:
                term = ncr(k, j) * ncr(n - 1, k - 1)
                total = total + term if flag else total - term
                n -= limit
                j += 1
                flag = not flag
            return total

        result = 0
        start = (zero + limit - 1) // limit
        prv, cur, nxt = 0, ways(one, start), ways(one, start + 1)

        for k in range(start, zero + 1):
            result += (prv + 2 * cur + nxt) * ways(zero, k)
            prv, cur, nxt = cur, nxt, ways(one, k + 2)

        return result % MOD


class Solution:
    def bitwiseComplement(self, n: int) -> int:
        if n == 0: return 1
        mask = n
        for i in (1, 2, 4, 8, 16):
            mask |= mask >> i
        return ~n & mask

class Solution:
    def frequencySort(self, s: str) -> str:
        # Step 1: Count frequency of each character
        frequency_map: Dict[str, int] = {}
        for c in s:
            frequency_map[c] = frequency_map.get(c, 0) + 1

        # Step 2: Create buckets for frequencies
        max_frequency = len(s)
        buckets: List[List[str] | None] = [None] * (max_frequency + 1)
        for c, frequency in frequency_map.items():
            if buckets[frequency] is None:
                buckets[frequency] = []
            buckets[frequency].append(c)

        # Step 3: Build the result by traversing the buckets in reverse order
        sb: List[str] = []
        for i in range(max_frequency, 0, -1):
            if buckets[i] is not None:
                for c in buckets[i]:
                    for _ in range(i):
                        sb.append(c)

        return "".join(sb)

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def unite(self, a, b):
        pa = self.find(a)
        pb = self.find(b)

        if pa == pb:
            return False

        if self.rank[pa] < self.rank[pb]:
            pa, pb = pb, pa

        self.parent[pb] = pa

        if self.rank[pa] == self.rank[pb]:
            self.rank[pa] += 1

        self.components -= 1
        return True


class Solution:
    def canAchieve(self, n, edges, k, x):
        dsu = DSU(n)

        # Mandatory edges
        for u, v, s, must in edges:
            if must == 1:
                if s < x:
                    return False
                if not dsu.unite(u, v):
                    return False

        # Free optional edges
        for u, v, s, must in edges:
            if must == 0 and s >= x:
                dsu.unite(u, v)

        # Upgrade edges
        used_upgrades = 0

        for u, v, s, must in edges:
            if must == 0 and s < x and 2 * s >= x:
                if dsu.unite(u, v):
                    used_upgrades += 1
                    if used_upgrades > k:
                        return False

        return dsu.components == 1

    def maxStability(self, n, edges, k):
        # Check mandatory edges cycle
        dsu = DSU(n)
        for u, v, s, must in edges:
            if must == 1:
                if not dsu.unite(u, v):
                    return -1

        low, high = 1, 200000
        ans = -1

        while low <= high:
            mid = (low + high) // 2

            if self.canAchieve(n, edges, k, mid):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1

        return ans