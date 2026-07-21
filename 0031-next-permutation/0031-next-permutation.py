class Solution:
    def nextPermutation(self, nums: list[int]) -> None:
        n = len(nums)

        # Step 1: Find the pivot (rightmost i where nums[i] < nums[i+1])
        i = n - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1

        # Step 2: If pivot exists, find the successor and swap
        if i >= 0:
            j = n - 1
            while nums[j] <= nums[i]:
                j -= 1
            # Step 3: Swap pivot with successor
            nums[i], nums[j] = nums[j], nums[i]

        # Step 4: Reverse the suffix starting at i+1
        left = i + 1
        right = n - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1