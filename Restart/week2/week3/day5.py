import math

from pyparsing import Optional

from day1 import ListNode


class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None:
            return None
        
        # Initialize two pointers
        slow = head
        fast = head
        
        # Move the slow pointer by 1 and fast pointer by 2
        while fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next
            
            # If they meet, a cycle exists
            if slow == fast:
                start = head
                while start != slow:
                    slow = slow.next
                    start = start.next
                return start
        
        # No cycle present
        return None

class Solution:
    def isHappy(self, n: int) -> bool:
        slow = n
        fast = self.getNext(n)
        
        while fast != 1 and slow != fast:
            slow = self.getNext(slow)              # Move one step
            fast = self.getNext(self.getNext(fast))  # Move two steps
        
        return fast == 1

    def getNext(self, n: int) -> int:
        totalSum = 0
        while n > 0:
            digit = n % 10
            totalSum += digit * digit
            n //= 10
        return totalSum
    


class Solution:
    def minNumberOfSeconds(self, height: int, times: list[int]) -> int:
        lo, hi = 1, 10**16

        while lo < hi:
            mid = (lo + hi) >> 1
            tot = 0
            for t in times:
                tot += int(math.sqrt(mid / t * 2 + 0.25) - 0.5)
                if tot >= height: break
            if tot >= height:
                hi = mid
            else:
                lo = mid + 1

        return lo

class Solution:
    def getHappyString(self, n: int, k: int) -> str:

        total = 3 * (2 ** (n - 1))
        if k > total:
            return ""

        k -= 1
        result = []
        last = ""

        for pos in range(n):

            branch = 2 ** (n - pos - 1)
            choices = [c for c in "abc" if c != last]

            idx = k // branch
            result.append(choices[idx])

            last = choices[idx]
            k %= branch

        return "".join(result)

class Solution:
    def delNodes(self, root: Optional[TreeNode], to_delete: list[int]) -> list[TreeNode]:
        res: dict[int, TreeNode] = {root.val: root}
        to_delete: set[int] = set(to_delete)

        def recursion(parent: TreeNode | None, cur_node: TreeNode | None, isleft: bool) -> None:
            nonlocal res
            if cur_node is None:
                return

            recursion(cur_node, cur_node.left, True)
            recursion(cur_node, cur_node.right, False)

            if cur_node.val in to_delete:
                if cur_node.val in res:
                    del res[cur_node.val]

                if parent:
                    if isleft:
                        parent.left = None
                    else:
                        parent.right = None

                if cur_node.left:
                    res[cur_node.left.val] = cur_node.left
                if cur_node.right:
                    res[cur_node.right.val] = cur_node.right

        recursion(None, root, False)
        return res.values()