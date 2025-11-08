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


#3376
class Solution:
    def processQueries(self, c: int, connections: List[List[int]], queries: List[List[int]]) -> List[int]:
        parent = list(range(c + 1))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        # union connected stations
        for a, b in connections:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # link nodes in sorted order within each component
        next_node = [0] * (c + 1)
        comp_min = [0] * (c + 1)
        last = {}

        for i in range(1, c + 1):
            r = find(i)
            if comp_min[r] == 0:
                comp_min[r] = i
            else:
                next_node[last[r]] = i
            last[r] = i

        offline = [False] * (c + 1)
        res = []

        # process queries
        for t, x in queries:
            if t == 1: # maintenance check
                if not offline[x]:
                    res.append(x)
                else:
                    r = find(x)
                    res.append(comp_min[r] if comp_min[r] else -1)
            else: # t == 2 → turn off station
                if not offline[x]:
                    offline[x] = True
                    r = find(x)
                    if comp_min[r] == x:
                        y = next_node[x]
                        while y and offline[y]:
                            y = next_node[y]
                        comp_min[r] = y if y else 0

        return res


#2528 
class Solution(object):
    def maxPower(self, stations, r, k):
        """
        :type stations: List[int]
        :type r: int
        :type k: int
        :rtype: int
        """
        prefix = [0]
        size = len(stations)
        for i in stations:
            prefix.append(prefix[-1] + i)
        for i in range(size):
            stations[i] = prefix[min(i+r+1, size)] - prefix[max(0, i-r)]

        def check(min_power):
            diff = [0] * size
            cur_diff = 0
            cnt_station = 0
            for i, power in enumerate(stations):
                cur_diff += diff[i]
                power_diff = min_power - power - cur_diff

                if power_diff > 0: 
                    cnt_station += power_diff
                    if cnt_station > k:
                        return False
                    cur_diff += power_diff
                    if i + 2 * r + 1 < size: diff[i + 2 * r + 1] -= power_diff

            return True
        left = min(stations)
        right = left + k
        while left <= right:
            mid = left + (right - left) // 2
            if check(mid): left = mid + 1
            else: right = mid - 1

        return right

#1618
class Solution:
    def minimumOneBitOperations(self, n: int) -> int:
        res = 0
        while n:
            res ^= n
            n >>= 1
        return res