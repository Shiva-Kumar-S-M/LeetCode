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

class Solution:
    def isValid(self, pref, k, limit):
        n = len(pref)
        m = len(pref[0])

        for i in range(k - 1, n):
            for j in range(k - 1, m):
                x1 = i - k + 1
                y1 = j - k + 1

                total = pref[i][j]
                if x1 > 0:
                    total -= pref[x1 - 1][j]
                if y1 > 0:
                    total -= pref[i][y1 - 1]
                if x1 > 0 and y1 > 0:
                    total += pref[x1 - 1][y1 - 1]

                if total <= limit:
                    return True

        return False

    def maxSideLength(self, mat, threshold):
        n = len(mat)
        m = len(mat[0])

        pref = [row[:] for row in mat]

        # Row-wise prefix sum
        for i in range(n):
            for j in range(1, m):
                pref[i][j] += pref[i][j - 1]

        # Column-wise prefix sum
        for j in range(m):
            for i in range(1, n):
                pref[i][j] += pref[i - 1][j]

        low, high = 1, min(n, m)
        ans = 0

        while low <= high:
            mid = (low + high) // 2
            if self.isValid(pref, mid, threshold):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1

        return ans
    
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False

        mapST = {}
        mapTS = {}

        for cs, ct in zip(s, t):
            # Check mapping from s -> t
            if cs in mapST:
                if mapST[cs] != ct:
                    return False
            else:
                mapST[cs] = ct

            # Check mapping from t -> s
            if ct in mapTS:
                if mapTS[ct] != cs:
                    return False
            else:
                mapTS[ct] = cs

        return True
    
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        # Create a frequency array for the magazine's characters.
        magazineFreq = [0] * 26

        # Fill the frequency array with magazine characters.
        for ch in magazine:
            magazineFreq[ord(ch) - ord('a')] += 1

        # Check each character in ransomNote against the magazine frequency.
        for ch in ransomNote:
            # Decrement the count in magazine frequency array for each character in ransomNote.
            idx = ord(ch) - ord('a')
            if magazineFreq[idx] == 0:
                # If count is zero, we can't construct the ransom note.
                return False
            magazineFreq[idx] -= 1

        # All characters needed are available in the magazine.
        return True
class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        res = []
        for n in nums:
            if n & 1:
                res.append(n & ~(((n + 1) & ~n) >> 1))
            else:
                res.append(-1)
        return res
    

class Solution:
    def containsNearbyDuplicate(self, nums, k):
        # Loop through each element in the array as the first element of the pair
        for i in range(len(nums)):
            # Loop through each element after the i-th element
            j = i + 1
            while j <= i + k and j < len(nums):
                # If the same element is found within k distance, return true
                if nums[i] == nums[j]:
                    return True
                j += 1
        # If no such pair is found, return false
        return False
    
class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans = []
        for n in nums:
            if n != 2:
                ans.append(n - ((n + 1) & (-n - 1)) // 2)
            else:
                ans.append(-1)
        return ans

class Solution:
    def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
        # Map to store the list of anagrams
        anagram_map: dict[str, list[str]] = {}

        for word in strs:
            # Convert the word to a sorted-character key
            sorted_word = "".join(sorted(word))
            # If the sorted word is not in the map, add it with an empty list
            if sorted_word not in anagram_map:
                anagram_map[sorted_word] = []
            # Append the original word to the corresponding list
            anagram_map[sorted_word].append(word)

        # Return the grouped list of anagrams
        return list(anagram_map.values())

class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        def isSorted(nums, n) -> bool:
            for i in range(1,n):
                if nums[i] < nums[i - 1]: return False
            return True
        ans, n = 0, len(nums)
        while not isSorted(nums, n):
            ans += 1
            min_sum, pos = float('inf'), -1
            for i in range(1,n):
                sum = nums[i - 1] + nums[i]
                if sum < min_sum:
                    min_sum = sum
                    pos = i
            nums[pos - 1] = min_sum
            for i in range(pos, n-1): nums[i] = nums[i + 1]
            n -= 1
        return ans

class Solution:
    def reorganizeString(self, S: str) -> str:
        count = [0] * 26
        for c in S:
            count[ord(c) - ord('a')] += 1

        # Find the character with the maximum frequency
        max_count = 0
        max_char_idx = 0
        for i in range(26):
            if count[i] > max_count:
                max_count = count[i]
                max_char_idx = i
        max_char = chr(max_char_idx + ord('a'))

        # Check if reorganization is possible
        if max_count > (len(S) + 1) // 2:
            return ""

        result = [''] * len(S)
        index = 0

        # Place the highest frequency character at even positions
        while count[max_char_idx] > 0:
            result[index] = max_char
            index += 2
            count[max_char_idx] -= 1

        # Fill other characters
        for i in range(26):
            while count[i] > 0:
                if index >= len(S):
                    index = 1  # Switch to odd positions
                result[index] = chr(i + ord('a'))
                index += 2
                count[i] -= 1

        return "".join(result)
    
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if not nums:
            return 0

        nums.sort()

        longest_streak = 1
        current_streak = 1

        for i in range(1, len(nums)):
            # if the current element is identical to the previous,
            # just continue through the iteration
            if nums[i] == nums[i - 1]:
                continue
            # check for consecutive sequence
            if nums[i] == nums[i - 1] + 1:
                current_streak += 1
            else:
                longest_streak = max(longest_streak, current_streak)
                current_streak = 1

        return max(longest_streak, current_streak)
