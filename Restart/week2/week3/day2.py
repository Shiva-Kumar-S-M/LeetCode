#316 remove dulpicate letters (medium)

class Solution:
    def removeDuplicates(self,s):
        frq={}
        for char in s:
            frq[char]=frq.get(char,0)+1

        stack=[]
        in_stack=set()

        for i in s:
            frq[i]-=1

            if i in in_stack:
                continue
            while stack and stack[-1]>i and frq[stack[-1]]>0:
                rmv=stack.pop()
                in_stack.remove(rmv)
            stack.append(i)
            in_stack.add(i)
        return ''.join(stack)
    


    #2390 removing stars from a string (medium)

    class Solution:
        def removeStars(self,s):
            stack=[]
            for i in s:
                if i=='*':
                    stack.pop()

                else:
                    stack.append(i)

            return ''.join(stack)
        

#150 evaluate reverse polish notation (medium)
class Solution:
    def evalRPN(self,tokens):
        stack=[]
        for i in tokens:
            if i in "+-*/":
                b=stack.pop()
                a=stack.pop()

                if i=='+':
                    stack.append(a+b)
                elif i=='-':
                    stack.append(a-b)
                elif i=='*':
                    stack.append(a*b)
                else:
                    stack.append(int(a/b))
            else:
                stack.append(int(i))
                
        return stack[0]
    

#Daily temperatures (medium)
class Solution:
    def dailyTemeperatures(self,temp):
        stack=[]
        res=[0]*len(temp)

        for i in range(len(temp)):
            while stack and temp[i]>temp[stack[-1]]:
                idx=stack.pop()
                res[idx]=i-idx
            stack.append(i)

        return res