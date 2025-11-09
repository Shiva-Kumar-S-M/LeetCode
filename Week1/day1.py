''' 2619:
Input: num1 = 2, num2 = 3
Output: 3
Explanation: 
- Operation 1: num1 = 2, num2 = 3. Since num1 < num2, we subtract num1 from num2 and get num1 = 2, num2 = 3 - 2 = 1.
- Operation 2: num1 = 2, num2 = 1. Since num1 > num2, we subtract num2 from num1.
- Operation 3: num1 = 1, num2 = 1. Since num1 == num2, we subtract num2 from num1.
Now num1 = 0 and num2 = 1. Since num1 == 0, we do not need to perform any further operations.
So the total number of operations required is 3.'''


class Solution:
    def countOperations(self, x: int, y: int) -> int:
        return 0 if y==0 else x//y+self.countOperations(y,x%y)
    

'''1716:
Hercy wants to save money for his first car. He puts money in the Leetcode bank every day.

He starts by putting in $1 on Monday, the first day. Every day from Tuesday to Sunday, he will put in $1 more than the day before. On every subsequent Monday, he will put in $1 more than the previous Monday.

Given n, return the total amount of money he will have in the Leetcode bank at the end of the nth day.

 

Example 1:

Input: n = 4
Output: 10
Explanation: After the 4th day, the total is 1 + 2 + 3 + 4 = 10.
Example 2:

Input: n = 10
Output: 37
Explanation: After the 10th day, the total is (1 + 2 + 3 + 4 + 5 + 6 + 7) + (2 + 3 + 4) = 37. Notice that on the 2nd Monday, Hercy only puts in $2.'''

class Sloution:
    def totalMoney(self, n: int) -> int:
        weeks, days = divmod(n, 7)
        return weeks * (28 + 7 * (weeks - 1) // 2) + days * (weeks + 1) + days * (days - 1) // 2
    
''' class Solution:
    def totalMoney(self, n: int) -> int:
        return 28*(q:=n//7)+7*q*(q-1)//2+(2*q+(r:=n%7)+1)*r//2'''