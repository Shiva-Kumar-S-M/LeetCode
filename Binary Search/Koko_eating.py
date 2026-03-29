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