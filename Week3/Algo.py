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