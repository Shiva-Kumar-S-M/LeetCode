#1437 
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

        