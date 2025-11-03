#1578 Minimum tree to make rope colorful
from git import List


class Solution:
    def minCost(self, colors: str, neededTime: List[int]) -> int:
        total_cost = 0
        n = len(colors)
        i = 0
        
        while i < n:
            j = i + 1
            max_time = neededTime[i]
            sum_time = neededTime[i]
            
            while j < n and colors[j] == colors[i]:
                sum_time += neededTime[j]
                max_time = max(max_time, neededTime[j])
                j += 1
            
            total_cost += (sum_time - max_time)
            i = j
        
        return total_cost
    

#2 Add two  numbers
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next    
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy_head = ListNode(0)
        current = dummy_head
        carry = 0
        
        while l1 or l2 or carry:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0
            
            total = val1 + val2 + carry
            carry = total // 10
            current.next = ListNode(total % 10)
            current = current.next
            
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        
        return dummy_head.next