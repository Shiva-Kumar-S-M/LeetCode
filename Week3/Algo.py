class Solution:
    from git import List


def largestSquareArea(self, bl: List[List[int]], tr: List[List[int]]) -> int:
        s = 0
        n = len(bl)

        for i in range(n):
            for j in range(i + 1, n):
                min_x = max(bl[i][0], bl[j][0])
                max_x = min(tr[i][0], tr[j][0])
                min_y = max(bl[i][1], bl[j][1])
                max_y = min(tr[i][1], tr[j][1])

                if min_x < max_x and min_y < max_y:
                    length = min(max_x - min_x, max_y - min_y)
                    s = max(s, length)

        return s * s


class Solution:
    def getSum(self, a: int, b: int) -> int:
        mask = 0xFFFFFFFF
        # Continue the loop until there are no carries left
        while b != 0:
            # Calculate the carry
            carry = (a & b) & mask
            # Calculate sum ignoring the carry
            a = (a ^ b) & mask
            # Update the carry, shifted left
            b = (carry << 1) & mask
        # Finally, a contains the sum
        # Convert to signed 32-bit
        return a if a <= 0x7FFFFFFF else a - 0x100000000
    

class MyHashMap:
    def __init__(self):
        self.data = [None] * 1000001
    def put(self, key: int, val: int) -> None:
        self.data[key] = val
    def get(self, key: int) -> int:
        val = self.data[key]
        return val if val != None else -1
    def remove(self, key: int) -> None:
        self.data[key] = None

class Solution:
    def largestMagicSquare(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        res = 1

        def isValid(i, j, k):
            s = None
            for x in range(i, i + k):
                row = sum(grid[x][j:j + k])
                if s is None: s = row
                elif s != row: return False

            for y in range(j, j + k):
                if sum(grid[x][y] for x in range(i, i + k)) != s:
                    return False

            if sum(grid[i + d][j + d] for d in range(k)) != s:
                return False

            if sum(grid[i + d][j + k - 1 - d] for d in range(k)) != s:
                return False

            return True

        for k in range(2, min(m, n) + 1):
            for i in range(m - k + 1):
                for j in range(n - k + 1):
                    if isValid(i, j, k):
                        res = k
        return res

class Solution:
    def maxNumberOfBalloons(self, text: str) -> int:
        # Create a frequency map for the input string
        freqMap = {}
        for c in text:
            freqMap[c] = freqMap.get(c, 0) + 1

        # Store the required frequencies of each character in 'balloon'
        balloon = "balloon"

        # Calculate the max number of "balloon" words
        maxBalloons = float('inf')
        balloonFreq = {}
        for c in balloon:
            balloonFreq[c] = balloonFreq.get(c, 0) + 1

        # Calculate the max possible number of "balloon" we can form
        for key, count in balloonFreq.items():
            maxBalloons = min(maxBalloons, freqMap.get(key, 0) // count)

        return 0 if maxBalloons == float('inf') else maxBalloons
    
class Solution:
    def numIdenticalPairs(self, nums):
        count = 0
        # Outer loop to fix the first element of the pair
        for i in range(len(nums)):
            # Inner loop to fix the second element of the pair
            for j in range(i + 1, len(nums)):
                # Check if we have a good pair
                if nums[i] == nums[j]:
                    count += 1
        return count