from collections import deque
from typing import List


class Solution:
    def minEatingSpeed(self, piles: list[int], h: int) -> int:
        left = 1
        right = max(piles)

        while left < right:
            mid = left + (right - left) // 2

            # Check if Koko can finish at speed mid
            total_hours = 0
            for pile in piles:
                total_hours += (pile + mid - 1) // mid

            if total_hours <= h:
                # Feasible, but maybe a slower speed works too
                right = mid
            else:
                # Too slow, need a faster speed
                left = mid + 1

        return left

class Solution:
    N = 10**6 + 5
    prime = [True] * N
    prime[0] = prime[1] = False
    
    for i in range(2, 1001):
        if prime[i]:
            for j in range(i * i, N, i):
                prime[j] = False

    def minJumps(self, nums: List[int]) -> int:
        n = len(nums)
        limit = nums[0]
        for c in nums:
            limit = max(limit, c)

        head = [-1] * (limit + 1)
        nxt = [-1] * n
        for i in range(n):
            val = nums[i]
            nxt[i] = head[val]
            head[val] = i

        dp = [-1] * n
        dp[0] = 0
        queue = deque([0])
        seen = set()

        while queue:
            dq = queue.popleft()

            if dq == n - 1:
                return dp[dq]

            right = dq + 1
            if right < n and dp[right] == -1:
                dp[right] = dp[dq] + 1
                queue.append(right)

            left = dq - 1
            if left >= 0 and dp[left] == -1:
                dp[left] = dp[dq] + 1
                queue.append(left)

            val = nums[dq]
            if Solution.prime[val] and val not in seen:
                seen.add(val)
                for i in range(val, limit + 1, val):
                    j = head[i]
                    while j != -1:
                        if dp[j] == -1:
                            dp[j] = dp[dq] + 1
                            queue.append(j)
                        j = nxt[j]
                    head[i] = -1
        return -1

class Solution:
    def rotateGrid(self, grid: List[List[int]], k: int) -> List[List[int]]:
        T, L = 0, 0
        B, R = len(grid) - 1, len(grid[0]) - 1

        while T < B and L < R:
            ln, wid = B - T, R - L
            perimeter = 2 * ln + 2 * wid
            r = k % perimeter

            while r:
                tmp = grid[T][L]

                for i in range(L, R):
                    grid[T][i] = grid[T][i + 1]

                for i in range(T, B):
                    grid[i][R] = grid[i + 1][R]

                for i in range(R, L, -1):
                    grid[B][i] = grid[B][i - 1]

                for i in range(B, T, -1):
                    grid[i][L] = grid[i - 1][L]

                grid[T + 1][L] = tmp
                r -= 1

            T += 1
            L += 1
            B -= 1
            R -= 1

        return grid

class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)

        dp = [-1] * n

        # base case
        dp[0] = 0

        for i in range(n):

            # unreachable index
            if dp[i] == -1:
                continue

            for j in range(i + 1, n):

                diff = nums[j] - nums[i]

                if -target <= diff <= target:

                    dp[j] = max(dp[j], dp[i] + 1)

        return dp[-1]


class Solution:
    def separateDigits(self, nums: List[int]) -> List[int]:
        
        result = []

        for num in nums:

            s = str(num)

            for ch in s:

                result.append(int(ch))

        return result

class Solution:
    def minimumEffort(self, shop: List[List[int]]) -> int:
        shop.sort(key=lambda x: x[1] - x[0], reverse=True)
        
        start = shop[0][1]
        bal = shop[0][1] - shop[0][0]
        loan = 0

        for i in range(1, len(shop)):
            cost, thresh = shop[i]
            
            if bal < thresh:
                loan += thresh - bal
                bal = thresh
                
            bal -= cost

        return start + loan

class Solution:
    def minMoves(self, nums: List[int], limit: int) -> int:
        n = len(nums)
        delta = [0] * (2 * limit + 2)

        for i in range(n // 2):
            mini = min(nums[i], nums[-1 - i])
            maxi = max(nums[i], nums[-1 - i])

            delta[2] += 2
            delta[mini + 1] -= 1
            delta[mini + maxi] -= 1
            delta[mini + maxi + 1] += 1
            delta[maxi + limit + 1] += 1

        res = n
        moves = 0

        for targ in range(2, 2 * limit + 1):
            moves += delta[targ]
            res = min(res, moves)

        return res


class Solution:
    def isGood(self, nums: List[int]) -> bool:
        n = len(nums) - 1
        dup = 0

        for num in nums:
            val = abs(num)
            if val > n: return False

            if nums[val - 1] < 0:
                if val < n or dup: return False
                dup |= 1
                continue

            nums[val - 1] = -nums[val - 1]

        return True