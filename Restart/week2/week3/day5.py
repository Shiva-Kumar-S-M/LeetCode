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