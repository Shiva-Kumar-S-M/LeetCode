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