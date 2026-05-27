class Solution:
    def searchInsert(self, nums, target):

        start = 0
        end = len(nums) - 1
        mid = 0

        while start <= end:

            mid = start + (end - start) // 2

            if nums[mid] == target:
                return mid

            elif nums[mid] < target:
                start = mid + 1

            else:
                end = mid - 1

        return mid + 1 if target > nums[mid] else mid

class Solution:
    def canReach(self, s: str, minJ: int, maxJ: int) -> bool:
        n = len(s)

        if int(s[-1]): return False

        dp = [False] * n
        dp[0] = True
        reach, maxR = 0, maxJ

        for i in range(minJ, n):
            if i > maxR: return False

            reach += dp[i - minJ]

            if i > maxJ:
                reach -= dp[i - maxJ - 1]

            if reach and not int(s[i]):
                dp[i] = True
                maxR = i + maxJ

        return reach > 0

class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lower = 0
        upper = 0

        for ch in word:
            if ch.islower():
                lower |= (1 << (ord(ch) - ord('a')))
            else:
                upper |= (1 << (ord(ch) - ord('A')))

        common = lower & upper

        # counting number of set bits
        return common.bit_count()

class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        A = [[False, False] for _ in range(27)]

        for ch in word:
            i = ord(ch) & 31
            c = ord(ch) >> 5 & 1
            A[i][c] = not (c and A[i][0])

        return sum(u and v for u, v in A)