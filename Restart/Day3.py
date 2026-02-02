#2348 No of zeros filled subarray
def countZero(self,nums):
    count=0
    res=0
    for num in nums:
        if num==0:
            count+=1
            res+=count
        else:
            count=0
    return res


def countzeros(self,nums):
    res=0
    count=0
    for num in nums:
        if num==0:
            count+=1

        else:
            res+=count*(count+1)//2
            count=0
    res+=count*(count+1)//2
    return res


#334 increasing Triplet Sequence

def increasingSequence(self,nums):
    num1=float('inf')
    num2=float('inf')

    for num in nums:
        if num<=num1:
            num1=num
        elif num<=num2:
            num2=num
        else:
            return True
    return False



