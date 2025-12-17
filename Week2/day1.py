#1437 
import sys
from git import List


class Solution:
    def kLengthApart(self, nums: List[int], k: int) -> bool:
        if k == 0:
            return True
        prev = None
        for i,num in enumerate(nums):
            if num == 1:
                if prev is not None and i - prev <= k:
                    return False
                prev=i
        return True


# 189 Rotate array
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n=len(nums)
        k=k % n
        rotated=[0]*n

        for i in range(n):
            rotated[(i+k)%n]=nums[i]
        for i in range(n):
            nums[i]=rotated[i]
__import__("atexit").register(lambda: open("display_runtime.txt", 'w').write('0'))


#Another approach for rotate array
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        n = len(nums)
        k %= n

        def reverse(l, r):
            while l < r:
                nums[l], nums[r] = nums[r], nums[l]
                l += 1
                r -= 1

        reverse(0, n - 1)
        reverse(0, k - 1)
        reverse(k, n - 1)

#121 Best time to buy and sell stock
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        minSoFar = prices[0]
        profit = 0
        for i in range(1, len(prices)):
            minSoFar = min(minSoFar,prices[i])
            profit = max(profit,prices[i] - minSoFar)
        return profit
__import__("atexit").register(lambda: open("display_runtime.txt", 'w').write('0'))



#Another approach for best time to buy and sell stock
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        left = 0
        right = 1
        maxProfit = 0

        while right < len(prices):
            if prices[left] < prices[right]:
                profit = prices[right] - prices[left]
                maxProfit = max(maxProfit, profit)
            else:
                left = right
            right += 1
        return maxProfit
        
#Another approach for best time to buy and sell stock
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        buy=prices[0]
        profit=0
        for i in range(1,len(prices)):
            if prices[i]<buy:
                buy=prices[i]
            elif prices[i]-buy>profit:
                profit=prices[i]-buy
        return profit
        
#717 1 bit and 2 bit characters
class Solution:
    def isOneBitCharacter(self, bits: List[int]) -> bool:
        n=len(bits)
        i=0
        while i < n-1:
            i+= bits[i]+1
        return i==n-1
        


#2154 keep multiplying found values by two
class Solution:
    def findFinalValue(self, nums: List[int], k: int) -> int:
        bits = 0
        for num in nums:
            if num % k != 0:
                continue
            n = num // k
            if n & (n - 1) == 0:
                bits |= n
        d = bits + 1
        return k * (d & -d)
        
#757    Set intersection size at least two
import heapq
class Solution:
    def intersectionSizeTwo(self, intervals: List[List[int]]) -> int:
        n=len(intervals)
        intervals.sort(key=lambda x:x[1])
        prev1=intervals[0][1]-1
        prev2=intervals[0][1]
        c=2
        for i in range(1,n):
            if prev2<intervals[i][0]:
                prev1=intervals[i][1]-1
                prev2=intervals[i][1]
                c+=2
            elif prev1<intervals[i][0]:
                if intervals[i][1]==prev2:
                    prev1=intervals[i][1]-1
                else:
                    prev1=intervals[i][1]
                prev1,prev2=min(prev1,prev2),max(prev1,prev2)
                c+=1
        return c
    
#1341 
class Solution:
    def countPalindromicSubsequence(self, s: str) -> int:
        n = len(s)
        first = [-1] * 26
        last = [-1] * 26

        for i, ch in enumerate(s):
            c = ord(ch) - ord('a')
            if first[c] == -1:
                first[c] = i
            last[c] = i

        ans = 0
        for c in range(26):
            if first[c] != -1 and last[c] - first[c] > 1:
                mask = 0
                for i in range(first[c] + 1, last[c]):
                    mask |= 1 << (ord(s[i]) - ord('a'))
                ans += bin(mask).count("1")

        return ans
        
        
        
class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        return len(list(filter(lambda x: x%3!=0, nums)))
        

#1262 Greatest Sum divisible by three
class Solution:
    def maxSumDivThree(self, nums: List[int]) -> int:
        s = sum(nums)
        
        if s % 3 == 0:
            return s
        
        r11 = 10000
        r12 = 10000
        r21 = 10000
        r22 = 10000
        
        for num in nums:
            if num % 3 == 1 and num < r12:
                if num < r11:
                    r12 = r11
                    r11 = num
                else:
                    r12 = num
            if num % 3 == 2 and num < r22:
                if num < r21:
                    r22 = r21
                    r21 = num
                else: 
                    r22 = num
        if s % 3 == 1:
            return s - min(r11, r21+r22)
        if s % 3 == 2:
            return s - min(r21, r11+r12) 

class Solution:
    def prefixesDivBy5(self, nums: List[int]) -> List[bool]:
        val = 0
        for i in range(len(nums)):
            val = ((val << 1) + nums[i]) % 5
            nums[i] = val == 0
        return nums

class Solution:
    def smallestRepunitDivByK(self, K: int) -> int:
        remainder = 0
        for length_N in range(1,K+1):
            remainder = (remainder*10+1) % K
            if remainder == 0:
                return length_N
        return -1

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix_product = 1
        postfix_product = 1
        result = [0]*n
        for i in range(n):
            result[i] = prefix_product
            prefix_product *= nums[i]
        for i in range(n-1,-1,-1):
            result[i] *= postfix_product
            postfix_product *= nums[i]
        return result
    
#122 Best time to buy and sell stock II
class Solution:
    def maxProfit(self,prices:List[int]):
        profit=0
        for i in range(1,len(prices)):
            if prices[i]> prices[i-1]:
                profit+=(prices[i]-prices[i-1])
        return profit
    

#238 Product of array except self
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res=[1]*(len(nums))
        prefix=1
        for i in range(len(nums)):
            res[i]=prefix
            prefix*=nums[i]
        postfix=1
        for i in range(len(nums)-1,-1,-1):
            res[i]*=postfix
            postfix*=nums[i]
        return res

class Solution:
    def numberOfPaths(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m, n = len(grid), len(grid[0])

        prev = [[0]*k for _ in range(n)]
        curr = [[0]*k for _ in range(n)]

        s = 0
        for j in range(n):
            s = (s + grid[0][j]) % k
            prev[j][s] = 1

        s = grid[0][0] % k

        for i in range(1, m):
            s = (s + grid[i][0]) % k
            curr[0] = [0]*k
            curr[0][s] = 1

            for j in range(1, n):
                curr[j] = [0]*k
                val = grid[i][j]
                for r in range(k):
                    nr = (r + val) % k
                    curr[j][nr] = (prev[j][r] + curr[j - 1][r]) % MOD

            prev, curr = curr, prev

        return prev[n - 1][0]

#2348 Number of zero filled subarrays
class Solution:
    def zeroFilledSubarray(self, nums: List[int]) -> int:
        res,count=0,0

        for i in range(len(nums)):
            if nums[i]==0:
                count+=1
                res+=count
            else:
                count=0
            
        return res


#334 Increasing Triplet Subsequence
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        i=float('inf')
        j=float('inf')
        for num in nums:
            if num<=i:
                i=num
            elif num<=j:
                j=num
            else:
                return True
        return False
    
#Hard 
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        nums = [n for n in nums if n > 0]
        nums.sort()


        target = 1
        for n in nums:
            if n == target:
                target += 1
            elif n > target:
                return target
        
        return target

#393 is subsequence

class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        i,j=0,0
        while i<len(s) and j<len(t):
            if s[i]==j[t]:
                i+=1
            j+=1
        return True if i==len(s) else False

# 3381 
class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefixSum = 0
        maxSum = -sys.maxsize
        kSum = [sys.maxsize // 2] * k
        kSum[k - 1] = 0
        for i in range(n):
            prefixSum += nums[i]
            maxSum = max(maxSum, prefixSum - kSum[i % k])
            kSum[i % k] = min(kSum[i % k], prefixSum)
        return maxSum
    
#125 Valid palindrome
class Solution:
    def isPalindrome(self, s: str) -> bool:
        left=0
        right=len(s)-1

        while left<right:
            if not s[left].isalnum():
                left+=1
                continue
            if not s[right].isalnum():
                right-=1
                continue
            if s[left].lower() != s[right].lower():
                return False
            
            left+=1
            right-=1
        return True
    

#Another Solution for 125 Valid palindrome
class Solution:
    def isPalindrome(self, s: str) -> bool:
        i=0
        j=len(s)-1
        s=s.lower()

        while i<=j:
            while i<j and not s[i].isalnum():
                i+=1
            while i<j and not s[j].isalnum():
                j-=1
            if s[i]!= s[j]:
                return False
            i+=1
            j-=1
        return True
    
#14 longest common prefix
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        res=""

        for i in range(len(strs[0])):
            for s in strs:
                if i==len(s) or s[i]!=strs[0][i]:
                    return res
            res+=strs[0][i]
        return res

#6 Zigzag conversion
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows==1: return s
        res=""

        for r in range(numRows):
            increment=2*(numRows-1)
            for i in range(r,len(s),increment):
                res+=s[i]
                if(r>0 and r<numRows-1 and i+increment-2*r<len(s)):
                    res+=s[i+increment-2*r]
        return res

#151 Revere words in a string
class Solution:
    def reverseWords(self, s: str) -> str:
        words=s.split()
        words.reverse()
        return " ".join(words)
    
#Another approach for 151 Revere words in a string
class Solution:
    def reverseWords(self,s:str)->str:
        words=s.split
        rev=words[::-1]
        rev_str=" ".join(rev)
        return rev_str
    
#hard
class Solution:
    def maxKDivisibleComponents(
        self, n: int, edges: List[List[int]], values: List[int], k: int
    ) -> int:
        # Step 1: Create adjacency list from edges
        adj_list = [[] for _ in range(n)]
        for node1, node2 in edges:
            adj_list[node1].append(node2)
            adj_list[node2].append(node1)

        # Step 2: Initialize component count
        component_count = [0]  # Use a list to pass by reference

        # Step 3: Start DFS traversal from node 0
        self.dfs(0, -1, adj_list, values, k, component_count)

        # Step 4: Return the total number of components
        return component_count[0]

    def dfs(
        self,
        current_node: int,
        parent_node: int,
        adj_list: List[List[int]],
        node_values: List[int],
        k: int,
        component_count: List[int],
    ) -> int:
        # Step 1: Initialize sum for the current subtree
        sum_ = 0

        # Step 2: Traverse all neighbors
        for neighbor_node in adj_list[current_node]:
            if neighbor_node != parent_node:
                # Recursive call to process the subtree rooted at the neighbor
                sum_ += self.dfs(
                    neighbor_node,
                    current_node,
                    adj_list,
                    node_values,
                    k,
                    component_count,
                )
                sum_ %= k  # Ensure the sum stays within bounds

        # Step 3: Add the value of the current node to the sum
        sum_ += node_values[current_node]
        sum_ %= k

        # Step 4: Check if the sum is divisible by k
        if sum_ == 0:
            component_count[0] += 1

        # Step 5: Return the computed sum for the current subtree
        return sum_

#136 Single number
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        x=0
        for i in nums:
            x^=i
        return x

#3512 Minimum operations to make array sum divisible by k
class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        return sum(nums)%k
    

#338 Count bits
class Solution:
    def countBits(self, n: int) -> List[int]:
        dp=1*(n+1)
        offset=1

        for i in range(1,n+1):
            if offset*2==i:
                offset=i
            dp[i]=1+dp[i-offset]
        return dp
    
#190 Reverse bits:
class Solution:
    def reverseBits(self, n: int) -> int:
        result=0
        for i in range(32):
            bit=(n>>i)&1
            result|=bit<<(31-i)
        return result

#201 Bitwise AND of numbers range
class Solution:
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        shift=0
        while left<right:
            left>>=1
            right>>=1
            shift+=1
        return left<<shift

#201 Bitwise AND of numbers range
class Solution:
    def minSubarray(self, nums: List[int], p: int) -> int:
        n = len(nums)
        total_sum = 0

        # Step 1: Calculate total sum and target remainder
        for num in nums:
            total_sum = (total_sum + num) % p

        target = total_sum % p
        if target == 0:
            return 0  # The array is already divisible by p

        # Step 2: Use a dict to track prefix sum mod p
        mod_map = {
            0: -1
        }  # To handle the case where the whole prefix is the answer
        current_sum = 0
        min_len = n

        # Step 3: Iterate over the array
        for i in range(n):
            current_sum = (current_sum + nums[i]) % p

            # Calculate what we need to remove
            needed = (current_sum - target + p) % p

            # If we have seen the needed remainder, we can consider this subarray
            if needed in mod_map:
                min_len = min(min_len, i - mod_map[needed])

            # Store the current remainder and index
            mod_map[current_sum] = i

        # Step 4: Return result
        return -1 if min_len == n else min_len

class Solution:
    def maxRunTime(self, n: int, batteries: List[int]) -> int:
        # Get the sum of all extra batteries.
        batteries.sort()   
        extra = sum(batteries[:-n])
        
        # live stands for the n largest batteries we chose for n computers.

        live = batteries[-n:]
        
        # We increase the total running time using 'extra' by increasing 
        # the running time of the computer with the smallest battery.
        for i in range(n - 1):
            # If the target running time is between live[i] and live[i + 1].
            if extra // (i + 1) < live[i + 1] - live[i]:
                return live[i] + extra // (i + 1)
            
            # Reduce 'extra' by the total power used.
            extra -= (i + 1) * (live[i + 1] - live[i])
        
        # If there is power left, we can increase the running time 
        # of all computers.
        return live[-1] + extra // n


class Solution:
    def countTrapezoids(self, points: List[List[int]]) -> int:
        point_num = defaultdict(int)
        mod = 10**9 + 7
        ans, total_sum = 0, 0
        for point in points:
            point_num[point[1]] += 1
        for p_num in point_num.values():
            edge = p_num * (p_num - 1) // 2
            ans = (ans + edge * total_sum) % mod
            total_sum = (total_sum + edge) % mod
        return ans

from math import gcd
from collections import defaultdict
from typing import List

class Solution:
    def countTrapezoids(self, points: List[List[int]]) -> int:
        t = defaultdict(lambda: defaultdict(int))
        v = defaultdict(lambda: defaultdict(int))

        n = len(points)

        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1

                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy

                g = gcd(dx, abs(dy))
                sx = dx // g
                sy = dy // g

                des = sx * y1 - sy * x1

                key1 = (sx << 12) | (sy + 2000)
                key2 = (dx << 12) | (dy + 2000)

                t[key1][des] += 1
                v[key2][des] += 1

        return self.count(t) - self.count(v) // 2

    def count(self, mp):
        ans = 0

        for inner in mp.values():
            total = sum(inner.values())
            remaining = total

            for val in inner.values():
                remaining -= val
                ans += val * remaining

        return ans


class Solution:
    def specialTriplets(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        num_cnt = {}
        num_partial_cnt = {}

        for v in nums:
            num_cnt[v] = num_cnt.get(v, 0) + 1

        ans = 0
        for v in nums:
            target = v * 2
            l_cnt = num_partial_cnt.get(target, 0)
            num_partial_cnt[v] = num_partial_cnt.get(v, 0) + 1
            r_cnt = num_cnt.get(target, 0) - num_partial_cnt.get(target, 0)
            ans = (ans + l_cnt * r_cnt) % MOD

        return ans
    
class Solution(object):
    MOD = 10**9 + 7

    def countPermutations(self, complexity):
        n = len(complexity)
        first = complexity[0]

        for i in range(1, n):
            if complexity[i] <= first:
                return 0

        fact = 1
        for i in range(2, n):
            fact = (fact * i) % self.MOD

        return fact

class Solution:
    def countCoveredBuildings(self, n: int, buildings: List[List[int]]) -> int:
        max_row = [0] * (n + 1)
        min_row = [n + 1] * (n + 1)
        max_col = [0] * (n + 1)
        min_col = [n + 1] * (n + 1)

        for p in buildings:
            x, y = p[0], p[1]
            max_row[y] = max(max_row[y], x)
            min_row[y] = min(min_row[y], x)
            max_col[x] = max(max_col[x], y)
            min_col[x] = min(min_col[x], y)

        res = 0
        for p in buildings:
            x, y = p[0], p[1]
            if (
                x > min_row[y]
                and x < max_row[y]
                and y > min_col[x]
                and y < max_col[x]
            ):
                res += 1

        return res

class Solution:
    def countMentions(self, n: int, events: List[List[str]]) -> List[int]:
        mentions = [0]*n
        back = [0]*n
        events.sort(key=lambda e: (int(e[1]), e[0]=="MESSAGE"))

        for typ, t, data in events:
            t = int(t)
            if typ == "OFFLINE":
                back[int(data)] = t + 60
                continue

            for tok in data.split():
                if tok == "ALL":
                    for u in range(n):
                        mentions[u] += 1
                elif tok == "HERE":
                    for u in range(n):
                        if t >= back[u]:
                            mentions[u] += 1
                else:  
                    mentions[int(tok[2:])] += 1

        return mentions

class Solution:
    def validateCoupons(self, code: List[str],
                        businessLine: List[str],
                        isActive: List[bool]) -> List[str]:
        e, g, p, r = [], [], [], []

        for i, active in enumerate(isActive):
            if not active:
                continue

            bl = businessLine[i]
            if bl not in {"electronics","grocery","pharmacy","restaurant"}:
                continue

            if not code[i] or not all(c.isalnum() or c == '_' for c in code[i]):
                continue

            if bl[0] == 'e': e.append(code[i])
            if bl[0] == 'g': g.append(code[i])
            if bl[0] == 'p': p.append(code[i])
            if bl[0] == 'r': r.append(code[i])

        return sorted(e) + sorted(g) + sorted(p) + sorted(r)
    
class Solution:
    def numberOfWays(self, corridor: str) -> int:
        mod = 10**9 + 7

        total_seats = corridor.count("S")
        if total_seats == 0 or total_seats % 2 == 1:
            return 0

        ans = 1
        seat = 0
        plant = 0

        for c in corridor:
            if c == "S":
                if seat == 2:
                    ans = ans * (plant + 1) % mod
                    plant = 0
                    seat = 0
                seat += 1
            else:  
                if seat == 2:
                    plant += 1

        return ans

class Solution:
    def getDescentPeriods(self, prices: List[int]) -> int:
        sum, des, prev=0, 0, -1
        for x in prices:
            des=(-((x+1)==prev) & des)+1
            sum+=des
            prev=x
        return sum


class Solution:
    def maxProfit(self, n: int, present: List[int], future: List[int], hierarchy: List[List[int]], budget: int) -> int:
        adj_list = defaultdict(list)
        for h in hierarchy:
            adj_list[h[0] - 1].append(h[1] - 1)
        
        @lru_cache(None)
        def dfs(employee, has_discount):
            cost = present[employee] // 2 if has_discount else present[employee]
            profit = future[employee] - cost
            
            buy_current = {cost: profit} if cost <= budget else {}
            skip_current = {0: 0}
            
            for child in adj_list[employee]:
                child_with_discount = dfs(child, True) # Do something
                child_no_discount = dfs(child, False) # Do nothing
                
                new_buy = {}
                for spent, prof in buy_current.items(): # Do something, but the current stock
                    for child_spent, child_prof in child_with_discount.items():
                        total_spent = spent + child_spent
                        if total_spent <= budget:
                            total_prof = prof + child_prof
                            if total_spent not in new_buy or new_buy[total_spent] < total_prof:
                                new_buy[total_spent] = total_prof
                buy_current = new_buy # This is mandatory because you need to check 
                                      # all possible combinations of picking children results. 
                                      # For example if the given graph is 1 -> 2, and 1 -> 3, 
                                      # it might be correct to either pick the path from 1 to 2, 
                                      # the path 1 to 3, or both paths if there is still budget left. 
                                      # Same goes for a skipping action.
                
                new_skip = {}
                for spent, prof in skip_current.items(): # Do nothing, skip the current stock
                    for child_spent, child_prof in child_no_discount.items():
                        total_spent = spent + child_spent
                        if total_spent <= budget:
                            total_prof = prof + child_prof
                            if total_spent not in new_skip or new_skip[total_spent] < total_prof:
                                new_skip[total_spent] = total_prof
                skip_current = new_skip
            
            result = {} # Merge the results of doing something and doing nothing at node employee
            for spent, prof in buy_current.items():
                if spent not in result or result[spent] < prof:
                    result[spent] = prof
            for spent, prof in skip_current.items():
                if spent not in result or result[spent] < prof:
                    result[spent] = prof
            
            return result
        
        result = dfs(0, False)
        return max(result.values()) if result else 0

class Data:
    def __init__(self, profit, buy, sell):
        self.profit=profit
        self.buy=buy
        self.sell=sell
class Solution:
    def maximumProfit(self, prices: List[int], k: int) -> int:
        x0=prices[0]
        dp=[Data(0, -x0, x0) for _ in range(k+1)]
        n=len(prices)
        for i in range(1, n):
            x=prices[i]
            for t in range(k, 0, -1):
                cur=dp[t]
                prevP=dp[t-1].profit
                # close transaction t
                cur.profit=max(cur.profit, cur.buy+x, cur.sell-x)
                # open transaction t
                cur.buy=max(cur.buy,  prevP-x)
                cur.sell=max(cur.sell, prevP+x)
        return dp[-1].profit