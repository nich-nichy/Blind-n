# 1). Python program for prime factor

# 36 = 3 * 3 * 2 * 2

# def productPrimeFactor(n):
#   product = 1
#   for i in range(2, n+1):
#      if (n % i == 0): #n % i
#            isPrime = 1
#            for j in range(2, int(i/2 + 1)): #i / 2 + 1
#                 if(i % j == 0): # i % j
#                     isPrime = 0
#                 if(isPrime):
#                     product = product * i
#     return product
# n = 18  # n + 1 = 19
# print(productPrimeFactor(n))


# 2). Convert time from 12 hour to 24 hour

# def convert(str1):
#     if str1[-2:] == 'AM' and str1[:2] == '12':
#         return str[2:-2]
#     elif str1[-2:] == 'AM':
#         return str1[:-2]
#     elif str1[-2:] == "PM" and str1[:2] == "12":
#         return str[:-2]
#     else:
#         return str(int(str1[:2]) + 12) + str[2:8]
#
# print(convert("04:00:00 PM"))


# 3). Multiway selection in python

# def add(numb1, numb2):
#     return numb1 + numb2
# def substract(numb1, numb2):
#     return numb1 - numb2
# def multiply(numb1, numb2):
#     return numb1 * numb2
# def divide(numb1, numb2):
#     numb1 / numb2
# print("Select an operation vroo: - \n"
#       "1. Addition \n"
#       "2. Subraction \n"
#       "3. Multiplication \n"
#       "4. Division \n")
# select = int(input("Select operations on 1, 2, 3, 4: "))
#
# numb1 = int(input("Enter first num: "))
# numb2 = int(input("Enter second num: "))
#
# if select == 1:
#     print(numb1, "+", numb2, "=", add(numb1, numb2))
# elif select == 2:
#     print(numb1, "-", numb2, "=", substract(numb1, numb2))
# elif select == 3:
#     print(numb1, "*", numb2, "=", multiply(numb1, numb2))
# elif select == 4:
#     print(numb1, "/", numb2, "=", divide(numb1, numb2))



# 4). Calculate the string A and calculate the string B total

# word_count = 0
# char_count = 0
# usr_input = input("Enter the string: ")
# split_string = usr_input.split()
# word_count = len(split_string)
# for i in split_string:
#     char_count = len(i)
#     print("Total words: {} ".format(word_count))
#     print("Total chars is : {}".format(char_count))


# 5). Tower of Hanoi

# def towerOfHanoi(n, source, dest, aux):
#     if n == 1:
#         print("Move disk 1 from source", source, "to destination", dest)
#         return
#     towerOfHanoi(n-1, source, aux, dest)
#     print("Move disk 1 from source", source, "to destination", dest)
#     towerOfHanoi(n-1, aux, dest, source)
#
#
#
# n = 4
# towerOfHanoi(n, 'A', 'B', 'C')


# Unit 2
# 1). Tuple functions

# tuple1, tuple2 = ('apple', 'banana', 'strawberry'), ('berry', 'mango')
# list1 = ['hai', 'hello', 'we', 'are', 'best', 'friends']
# print("elements on tuple 1: ", max(tuple1))
# print("elements on tuple 1: ", min(tuple1))
# print("elements on tuple 2: ", max(tuple2))
# print("elements on tuple 2: ", max(tuple2))
# print("elements on tuple 1: ", len(tuple1))
# print("elements on tuple 2: ", len(tuple2))
# tuple3 = tuple(list1)
# print(tuple3)

# ii). min, max, sorted

# list = [(2,3), (4, 7), (8, 11), (3, 6)]
# print("List size is " + str(list))
# res1 = min(list)[0], max(list)[0]
# res2 = min(list)[1], max(list)[1]
# print(str(res1))
# print(str(res2))



# import calendar
# yy = int(input("Year"))
# mm = int(input("Month"))
# print(calendar.month(yy, mm))

# 2). Binary search

# def binarySearch(x, high, low, arr):
#     if high >= low:
#         mid = (high + low - 1) // 2
#         if arr[mid] == x:
#             return mid
#         elif arr[mid] > x:
#             return binarySearch(arr, low, mid - 1, x)
#         else:
#             return binarySearch(arr, high, mid + 1, x)
#     else:
#         return -1
#
# arr = [2, 3, 4, 10, 40]
# x = 10
# result = binarySearch(x, 0, len(arr) - 1, arr)
# if result != 1:
#     print(str(result))
# else:
#     print("Element is not present in array")



# 4). Count and sum the even and odd number

# maximum= int(input("Max"))
# even_total = 0
# odd_total = 0
#
# for i in range(1, maximum+1):
#     if (i % 2 == 0):
#         even_total = even_total + i
#     else:
#         odd_total = odd_total + i
#
# print("{0} = {1}". format(i, even_total))
# print("{0} = {1}". format(i, odd_total))



# 4). ii). linear search

# def linear_search(list1, key, n):
#     for i in range(0, n):
#         if (list1[i] == key):
#             return i
#         return -1
# list1 = [2, 4, 6, 8, 9]
# key = 7
# n = len(list1)
# result = linear_search(list1, key, n)
# if result == -1:
#     print("No")
# else:
#     print(result)



# Unit 3
# 1). Duplicate chars in string

# def removeDuplicate(str, n):
#     index = 0
#     for i in range(0, n):
#         for j in range(0, i+1):
#             if str[i] == str[j]:
#                 break
#         if j == i:
#             str[index] = str[i]
#             index += 1
#     return " ".join(str[:index])
# str="nichfornich"
# n = len(str)
# print(removeDuplicate(list(str), n))




#  2). Palindrome

# def isPalindrome(s):
#     return s == s[::-1]
# s = 'malayalam'
# ans = isPalindrome(s)
# if ans:
#     print("palindrome")
# else:
#     print("not a palindrome")



# from collections import Counter
# def winner(input):
#     votes = Counter(input)
#     dict = {}
#     for value in votes.values():
#         dict[value] = []
#     for (key,value) in votes.items():
#         dict[value].append(key)
# maxVote = sorted(dict.keys(),reverse=True)[0]
# if len(dict[maxVote])>1:
#     print (sorted(dict[maxVote])[0])
# else:
#     print (dict[maxVote][0])
# # Driver program
# if __name__ == "__main__":
#  input =['john','johnny','jackie','johnny',
#  'john','jackie','jamie','jamie',
#  'john','johnny','jamie','johnny',
#  'john']
#  winner(input)



#################################################################################
#################################################################################




# Data structures and algorithms practice
# Day 1
# Reversed linked list


# def reverseLinkedList(head):
#     prev, curr = None, head;
#
#     while curr:
#         nxt = curr.next
#         curr.next = prev
#         prev = curr
#         curr = nxt
#     return prev


# Pivot index
# def pivotIndex(nums):
#     total = sum(nums)
#     leftSum = 0
#     for i in range(len(nums)):
#         rightSum = total - nums[i] - leftSum
#         if leftSum == rightSum:
#             return i
#         leftSum += nums[i]
#     return -1
# nums = [1,7,3,6,5,6]
# print(pivotIndex(nums))


# Isomorphic string:
# def isomorphicStrings(s, t):
#     mapST, mapTS = {}, {}
#     for i in range(len(s)):
#         c1, c2 = s[i], t[i]
#         if (c1 in mapST and mapST[c1] != c2) or (c2 in mapTS and mapTS[c2] != c1):
#             return False
#         mapST[c1] = c2
#         mapTS[2] = c1
#     return True
#
# s = "foo"
# t = "bar"
# print(isomorphicStrings(s, t))



# Reverse linked
# def reverseLinked(head):
#     prev, curr = None, head
#     # prev -> curr -> next
#     while curr:
#         nxt = curr.next
#         curr.next = prev
#         prev = curr
#         curr = nxt
#     return prev



#  Subsequence
# def isSubsequence(s, t):
#     i, j = 0, 0
#     while i < len(s) and j < len(t):
#         if s[i] == t[j]:
#             i += 1
#         j += 1
#     return True if i == len(s) else False
#
# s = "abc"
# t = "abdc"
# print(isSubsequence(s, t))


# Sum of 1d array
# def runningSum(nums):
#     for i in range(1, len(nums)):
#         nums[i] = nums[i - 1]
#     return nums
# nums = [1,2,3,4,5]
# print(runningSum(nums))



# Pivot index
# def pivotIndex(num):
#     total = sum(num)
#
#     leftSum = 0
#
#     for i in range(len(num)):
#         rightSum = total - num[i] - leftSum
#         if leftSum == rightSum:
#             return i
#         leftSum += num[i]
#     return -1
#
# num = [1,7,3,6,5,6]
# print(pivotIndex(num))



# Merge two sorted list

# class ListNode:
#     def __init__(self, val = 0, next = None):
#         self.val = val
#         self.next = next
#
# class Solution:
#     def mergeList(self, l1 : ListNode, l2: ListNode) -> ListNode:
#         dummy = ListNode()
#         tail = dummy
#
#         while l1 and l2:
#             if l1.val < l2.val:
#                 tail.next = l1
#                 l1 = l1.next
#             else:
#                 tail.next = l2
#                 l2 = l2.next
#             tail = tail.next
#             if l1:
#                 tail.next = l1
#             elif l2:
#                 tail.next = l2
#             return dummy.next
# l1 = [1, 2, 3]
# l2 = [4, 5, 6]
# Solution.mergeList(l1, l2)



# Profit and loss in a stock
#     import collections
#     from typing import List

# class Solution:
#     def maxProfit(self, prices: List[int]) -> int:
#         l, r = 0, 1
#         maxP = 0
#         while r < len(prices):
#             if prices[l] < prices[r]:
#                 profit = prices[r] - prices[l]
#                 maxP = max(maxP, profit)
#             else:
#                 l = r
#             r += 1
#         return maxP
#
# # prices = [7,1,5,3,6,4]
# # print(profitStock(prices))
# print(Solution.maxProfit(prices=[7,1,5,3,6,4]))



# Longest Palindrome
# class Solution:
#     def longestPalindrome(s) -> int:
#         res = ""
#         resLen = 0
#         for i in range(len(s)):
#             #odd one
#             l, r = i, i
#             while l >= 0 and r < len(s) and s[l] == s[r]:
#                 if (r - l + 1) > resLen:
#                     res = s[l : r + 1]
#                     resLen = r - l + 1
#                 l -= 1
#                 r += 1
#             #even one
#             l, r = i, i + 1
#             while l >= 0 and r < len(s) and s[l] == s[r]:
#                 if (r - l + 1) > resLen:
#                     res = s[l : r + 1]
#                     resLen = r - l + 1
#                 l -= 1
#                 r += 1
#         return len(res)
# s = "abaddd"
# print(Solution.longestPalindrome(s))



# Tree preorder traversal

# def preOrderTraversal(s, root):
#     output = []
#     def dfs(node):
#         if not node:
#             return
#         output.append(node.val)
#         for i in node.children:
#             dfs(i)
#     dfs(root)
#     return output



# Binary tree level order traversal

# def binaryTreeTraversal(self, root: TreeNode) -> List[List[int]]:
#     res = []
#     q = collections.deque()
#     q.append(root)
#
#     while q:
#         qLen = len(q)
#         level= []
#         for i in range(qLen):
#             node = q.popleft()
#             if node:
#                 level.append(node.val)
#                 q.append(node.left)
#                 q.append(node.right)
#         if level:
#             res.append(level)
#     return res



# N ary tree using breath for search algorothm
# class Node:
#     def __init__(self, val=None, children=None):
#         self.val = val
#         self.children = children
# class Solution:
#     def preOrder(self, root: 'Node') -> List[int]:
#         output = []
#         def naryTree(node):
#             if not node:
#                 return
#             output.append(node.val)
#
#             for i in node.children:
#                 naryTree(i)
#             naryTree(root)
#             return output



# Binary tree order traversal
# import collections
# def binarySearch(root):
#     res = []
#     q = collections.deque
#     q.append(root)
#
#     while q:
#         qLen = len(q)
#         level = []
#         for i in range(qLen):
#             node = q.popleft()
#             if node:
#                 level.append(node.val)
#                 q.append(node.left)
#                 q.append(node.right)
#         if level:
#             res.appent(level)
#     return res
#



# Binary search

# -1, 0, 3, 5, 9, 12
#  L               R
# def binarySearch(nums, target):
#     l, r = 0, len(nums) - 1
#
#     while l <= r:
#         m = (l + r) // 2
#         if nums[m] > target:
#             r = m - 1
#         elif nums[m] < target:
#             r = m + 1
#         else:
#             return m
#     return -1
#
#
# nums = [-1, 0, 1, 2, 3]
# target = 2
# print(binarySearch(nums, target))



# Binary search
# def binarySearch(nums, target):
#     l, r = 0, len(nums) - 1
#
#     while l <= r:
#         m = (l + r) // 2
#         if nums[m] > target:
#             r = m - 1
#         elif nums[m] < target:
#             l = m + 1
#         else:
#             return m
#     return -1
# nums = [1, 2, 3, 4, 5 ]
# target = 2
# print(binarySearch(nums, target))



# Bad version

# def firstBadVersion(n, target):
#     low = 1
#     high = n
#     mid = 0
#     result = n
#     while (low <= high):
#         mid = (low + high) // 2
#         if mid:
#             result = mid
#             high = mid - 1
#         else:
#             low = mid - 1
#     return result
# n = 5
# target = 4
# print(firstBadVersion(n, target))



# def loop(x):
#     print(x*3)
#
# def map_simple(crazy, list):
#     for i in list:
#         crazy(i)
# list = ['biriyani', True, 3, '4', 5, 6 ]
# map_simple(crazy= [int], list = [1, 2, 3, 4, 5, 6 ])
# fruits = ['apple', 'orange', 'kiwi', 'pineapple']
# newFruit = [i for i in fruits if 'z' in i ]
# print(newFruit)
# print(list[1:-4])



# Binary Search Tree

# def binarySearchTree(node, left, right):
#     if not node:
#         return True
#      if not (node.val < right and node.val > left):
#         return False
#       return (binarySearchTree(node.left, left, node.val) and binarySearchTree(node.right, right, node.val))
#  return (binarySearchTree(root, float(-inf) and binarySearchTree(float(inf))



# Lowest common ancestor of Binary Search Tree

# class TreeNode:
#      def __init__(self, x):
#          self.val = x
#          self.left = None
#          self.right = None
# class Solution:
#     def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
#         curr = root
#
#         if (p.val > curr.val and q.val > curr.val):
#             curr = curr.right
#         elif (p.val < curr.val and q.val > curr.val):
#             curr = curr.left
#         else:
#             return curr



# class Human:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age
# h1 = Human('Nich', 21)
# print(h1.age)



# def square(x):
#     return x * x
#
# numbers = [1, 2, 3, 4, 5]
#
# listSquares = map(square, numbers)
# print(list(listSquares))




# def complainManagemant(complain):
#     if complain:
#         print("You have an complaint and it has been stored successfully!... :)")
#         print("We may take immediate action")
#         complain = save.append(complain)
#         print(complain)
#     else:
#         return -1
# complain = input("Enter your complaints here: ")
# save = []
# print(complainManagemant(complain))



# def generateIp(s):
#
#     if s == " ":
#         print(index(s) + ".")
#     else:
#         return -1
#
# # Main
# s = print(int(input()))
# print(generateIp(s))



# def characterReplacement(s, k):
#     count = {}
#     res = 0
#
#     l = 0
#     maxf = 0
#
#     for r in range(len(s)):
#         count[s[r]] = count.get(s[r] , 0)
#         maxf = max(maxf, count[s[r]])
#         while (r - l + 1) - maxf > k:
#             count[s[l]] -= 1
#             l += 1
#         res = max(res, r - 1 + 1)
#     return res

# def findAnagrams(s, p):
#     if len(p) > len(s): return []
#     pCount, sCount = {}, {}
#     for i in range(len(p)):
#         pCount[p[i]] = 1 + pCount.get(p[i], 0)
#         sCount[s[i]] = 1 + sCount.get(s[i], 0)
#     res = [0] if sCount == pCount else []
#     l = 0
#     for r in range(len(p), len(s)):
#         sCount[s[r]] = 1 + sCount.get(s[r], 0)
#         sCount[s[l]] -= 1
#
#         if sCount[s[l]] == 0:
#             sCount.pop(s[l])
#         l += 1
#         if sCount == pCount:
#             res.append(l)
#     return res



# class Solution:
#     def twoSum(nums, target):
#         prevMap = {}
#         for i, n in enumerate(nums):
#             diff = target - n
#             if diff in prevMap:
#                 return [prevMap[diff], i]
#             prevMap[n] = i
#         return





# def findTriplets(arr, n):
#     for i in arr:
#         if arr[i] == True:
#             res = sorted(arr)
#             # print(res)
#             for j in res:
#                 if (0 < res[j]) != (0 > res[j]):
#                     print("j:", j)
#
#         # m = (l + r) // 2
#
# n = 5
# arr = [0, -1, 2, -3, 1]
#
# print(findTriplets(arr, n))




#
# def findTriplets(arr, n):
#     found = False
#     for i in range(0, n - 2):
#         for j in range(i + 1, n - 1):
#             for k in range(j + 1, n):
#                 if (arr[i] + arr[j] + arr[k] == 0):
#                     print(1)
#                     found = True
#     if (found == False):
#         print(0)


# Driver code
# arr = [0, -1, 2, -3, 1]
# n = len(arr)
# findTriplets(arr, n)


# def learningArr(v):
#     for i in range(0, v - 2):
#         for j in range(i + 1, v - 1):
#             for k in range(j + 1, n):
#                 print(i, j, k)
# v = [0, -1, 2, -3, 1]
# print(learningArr(v))


# def backspaceCompare(s):
#     arr = []
#     for i in range(len(s)):
#         res = arr.append(s)
#         if "#" in res:
#             res = res.pop(i)
#             print(res)
# s = "ab#cd"
# print(backspaceCompare(s))

# def decodestring(s):
#     stack = []
#
#     for i in range(len(s)):
#         if s[i] != "]":
#             stack.append(s[i])
#         else:
#             substr = ""
#             while stack[-1] != "[":
#                 substr = stack.pop() + substr
#             stack.pop()
#             k = ""
#             while stack and stack[-1].isdigit():
#                 k = stack.pop() + k
#                 stack.append(int(k) * substr)
#     return "".join(stack)

# Is the number happy

# def isHappy(self, n):
#     #  Hash set
#     visit = set()
#
#     while n not in visit:
#         visit.add(n)
#         n = self.sumofSquares(n)
#
#         if n == 1:
#             return True
#     return False
#
# def sumofSquares(n):
#     output = 0
#
#     while n:
#         digit = n % 10
#         digit = digit ** 2
#         output += digit
#         n = n // 10
#     return output


# def reverseArray(arr):
#     arr1, arr2 = [], []
#
#     for i in range(len(arr)):
#         l, r = min(range(arr[i])), max(range(arr[i]))
#         m = (l + r) // 2
#         if m < arr[i]:
#             arr1.append(m)
#             print("i" ,arr1)
#         elif m > arr[i]:
#             arr2.append(m)
#             print("j", arr2)
#
# arr = [1, 2, 3, 4, 5]
# print(reverseArray(arr))


# def reverseInGroups(arr, N, K):
#     i = 0
#     while(i<N):
#         if (i+K<N):
#             arr[i:i+K]=reversed(arr[i:i+K])
#             i += K
#     else:
#         arr[i:] = reversed(arr[i:])
#         i+=K
# arr = [1, 2, 3, 4, 5]
# N = 5
# K = 3
# print("this", reverseInGroups(arr, N, K))


# Program for spiral order
# class Solution:
#     def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
#         res = []
#         left, right = 0, len(matrix[0])
#         top, bottom = 0, len(matrix)
#
#         while left < right and top < bottom:
#             for i in range(left, right):
#                 res.append(matrix[top][i])
#             top += 1
#
#             for i in range(top, bottom):
#                 res.append(matrix[i][right - 1])
#             right -= 1
#
#             if not (left < right and top < bottom):
#                 break
#
#             for i in range(right - 1, left - 1, -1):
#                 res.append(matrix[bottom - 1][i])
#             bottom -= 1
#
#             for i in range(bottom - 1, top - 1, -1):
#                 res.append(matrix[i][left])
#             left += 1
#         return res


# def findDups(nums):
#     i, j = 0, len(nums)
#     for i in range(nums[i]):
#         i += 1
#         print(i)
#         for j in range(nums[j]):
#             j -= 1
#             print(j)
#             if nums[i] == nums[j]:
#                 arr2 = []
#                 arr2.append(nums[i])
#                 print(arr2)
#                 return True
#             else:
#                 arr2 = []
#                 arr2.append(i)
#                 print(arr2)
#                 return False
# nums = [1, 2, 3, 1]
# print(findDups(nums))
#

# def sortAndFindRepitations(arr):
#     s1, s2 = 0, len(arr)
#     for i in arr:
#         res = arr.sort()
#         for j in range(arr):
#             j -= 1
#             if arr[i] == arr[j]:
#                 print(arr[j])
# arr = [1, 1, 3, 2]
# print(sortAndFindRepitations(arr))

# def containDuplicates(nums):
#     size = len(nums)
#     repeated = []
#     for i in range(size):
#         k = i + 1
#         for j in range(k, size):
#             if nums[i] == nums[j] and nums[i] not in repeated:
#                 repeated.append(nums[i])
#                 return True
#     return False
# nums = [1, 2, 3, 4]
# print(containDuplicates(nums))

# def dups(nums):
#     return not len(nums) == len(set(nums))
# nums = [1, 2, 3, 1]
# print(dups(nums))

# def sort012(arr, arr_size):
#     lo = 0
#     hi = arr_size - 1
#     mid = 0
#     while mid <= hi:
#         if arr[mid] == 0:
#             arr[lo], arr[mid] = arr[mid], arr[lo]
#             lo = lo + 1
#             mid = mid + 1
#         elif arr[mid] == 1:
#             mid = mid + 1
#         else:
#             arr[mid], arr[hi] = arr[hi], arr[mid]
#             hi = hi - 1
#     return arr
# def printArray(arr):
#     for k in arr:
#         print(k, end=' ')
# # Main
# arr = [0, 1, 0]
# arr_size = len(arr)
# arr = sort012(arr, arr_size)
# printArray(arr)

# def anagram(s, t):
#     if len(s) != len(t):
#         return False
#     countS, countT = {}, {}
#
#     for i in range(len(s)):
#         countS[s[i]] = 1 + countS.get(s[i], 0)
#         countT[t[i]] = 1 + countT.get(t[i], 0)
#     return countS == countT
# s = "rat"
# t = "car"
# print(anagram(s, t))

# def addTwoLinkedLists(l1, l2):
#     head = None
#     temp = None
#     carry = 0
#     while l1 is not None or l2 is not None:
#         sum_value = carry
#         if l1 is not None:
#             sum_value = l1.val
#             l1 = l1.next
#         if l2 is not None:
#             sum_value = l2.val
#             l2 = l2.next
#         node = ListNode(sum_value % 10)
#         carry = sum_value // 10
#
#         if temp is None:
#             temp = head = node
#         else:
#             temp.next = node
#             temp = temp.next
#     if carry > 0:
#         temp.next = ListNode(carry)
#     return head
#

# def inpMatrix():
#     rows = int(input())
#     column = int(input())
#     dummy = []
#     print("Please enter the entries: ")
#     for i in range(rows):
#         r = []
#         for j in range(column):
#             r.append(int(input()))
#         dummy.append(r)
#     for i in range(rows):
#         for j in range(column):
#             print(dummy[i][j], end =" ")
#         print()



# New codes 

def encodeString(arr):
    res = ""
    for i in arr:
        res += str(len(i)) + '#' + i
    return res

# 3#wed3#set3#ped

def decodeString(str):
    res, i = [], 0
    while i < len(str):
        j = i
        while str[j] != '#':
            j += 1
        length = int(str[i:j])
        i = j + 1
        j = i + length
        res.append(str[i:j])
        i = j
    return res

strArr = ['wed', 'set', 'ped']
encodedStr = encodeString(strArr)
print(encodedStr)
print(decodeString(encodedStr))


def isZero(arr, n, w):
    res = []
    for i in range(n - w + 1):
        window = arr[i:i + w]
        if 0 in window:
            res.append(i + window.index(0) + 1) # this makes two pair arrays for easy comparing
        else:
            res.append(-1)
    print(res)
# 2 2 -1 -1 6 6
arr = [1, 0, 6, 7, 4, 0, 9]
n = 7
w = 2
print(isZero(arr, n, w))


def characterReplacement(s):
    count = {}
    res = 0
    l = 0
    maxF = 0
    for r in range(len(s)):
        count[s[r]] = 1 + count.get(s[r], 0)
        check = r - l + 1
        print(check, "val")
        maxF = max(maxF, count[s[r]])
        while (r - l + 1) - maxF > k:
            count[s[l]] -= 1
            l += 1
        res = max(res, r - l + 1)
        print(res, "res")
    return res

s = "ABAB"
# s = "AABABBA"
k = 2
print(characterReplacement(s))

def passportNumbers(arr, n):
    res = []
    arr = arr.split(" ")
    for i in arr:
        if i not in res:
            res.append(i)
    return ' '.join(map(str, res))

n = 5
arr = "A23 B56 B56 C79 D16"
print(passportNumbers(arr, n))


class Solution:
    def isPalindrome(self, s: str) -> bool:
        l, r = 0, len(s) - 1
        while l < r:
            while l < r and not self.isAlphaNum(s[l]):
                l += 1
            while r > l and not self.isAlphaNum(s[r]):
                r -= 1
            if s[l].lower() != s[r].lower():
                return False
            l, r = l + 1, r - 1
        return True
    
    def isAlphaNum(self, word):
        return (ord('A') <= ord(word) <= ord('Z') or
        ord('a') <= ord(word) <= ord('z') or
        ord('0') <= ord(word) <= ord('9')
        )


def intToRoman(num):
    val_to_symbol = [
        (1000, 'M'),
        (900, 'CM'),
        (500, 'D'),
        (400, 'CD'),
        (100, 'C'),
        (90, 'XC'),
        (50, 'L'),
        (40, 'XL'),
        (10, 'X'),
        (9, 'IX'),
        (5, 'V'),
        (4, 'IV'),
        (1, 'I')
    ]
    result = ""
    for val, sym in val_to_symbol:
        while num >= val:
            result += sym
            num -= val
    return result

print(intToRoman(3749))
print(intToRoman(58))
print(intToRoman(1994))


def twoSumTwo(arr, target):
    l, r = 0, len(arr) - 1

    while l < r:
        diff = arr[l] + arr[r]
        if diff > target:
            r -= 1
        elif diff < target:
            l += 1
        else:
            return [l+1, r+1]
    return []

# numbers = [2,7,11,15]
target = 6
numbers = [2,3,4]
print(twoSumTwo(numbers, target))

def threeSum(nums):
    res = []
    nums.sort()

    for i, a in enumerate(nums):
        if i > 0 and a == nums[i - 1]:
            continue

        l, r = i + 1, len(nums) - 1

        while l < r:
            diff = a + nums[l] + nums[r]
            if diff > 0:
                r -= 1
            elif diff < 0:
                l += 1
            else:
                res.append([a, nums[l], nums[r]])
                l += 1
                while nums[l] == nums[i - 1] and l < r:
                    l += 1
    return res


nums = [-1,0,1,2,-1,-4]
print(threeSum(nums))

def maxArea(height):
    res = 0
    l, r = 0, len(height) - 1

    while l < r:
        area = (r - l) * min(height[l], height[r])
        res = max(res, area)
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1
    return res

height = [1,8,6,2,5,4,8,3,7]
print(maxArea(height))


def bestTimeToBuy(prices):
    l, r = 0, 1
    maxP = 0

    while r < len(prices):
        if prices[l] < prices[r]:
            profit = prices[r] - prices[l]
            maxP = max(maxP, profit)
        else:
            l = r
        r += 1
    return maxP
 
prices = [7,1,5,3,6,4]
# prices = [7,6,4,3,1]
print(bestTimeToBuy(prices))

def lengthOfLongestSubstring(s):
    l = 0
    res = 0
    charSet = set()
    for r in range(len(s)):
        while s[r] in charSet:
            charSet.remove(s[l])
            l += 1
        charSet.add(s[r])
        res = max(res, r - l + 1)
    return res

s1 = "abcabcbb"
s2 = "bbbbb"
s3 = "pwwkew"

print(lengthOfLongestSubstring(s1))
print(lengthOfLongestSubstring(s2))
print(lengthOfLongestSubstring(s3))

def minWindow(s, t):
        if t == "": return ""

        countT, window = {}, {}
        for c in t:
            countT[c] = 1 + countT.get(c, 0)

        have, need = 0, len(countT)
        res, resLen = [-1, -1], float("infinity")
        l = 0
        for r in range(len(s)):
            c = s[r]
            window[c] = 1 + window.get(c, 0)

            if c in countT and window[c] == countT[c]:
                have += 1

            while have == need:
                if (r - l + 1) < resLen:
                    res = [l, r]
                    resLen = (r - l + 1)
                window[s[l]] -= 1
                if s[l] in countT and window[s[l]] < countT[s[l]]:
                    have -= 1
                l += 1
        l, r = res
        return s[l : r + 1] if res != float("infinity") else ""

s = "ADOBECODEBANC"
t = "ABC"
print(minWindow(s, t))

def minWindow(s, t):
        if t == "": return ""
        countT, window = {}, {} # hash maps

        for c in t:
            countT[c] = 1 + countT.get(c, 0)

        have, seen = 0, len(countT)
        l = 0
        res, resLen = [-1, -1], float("infinity")

        for r in range(len(s)):
            c = s[r]
            window[c] = 1 + window.get(c, 0)

            if c in countT and countT[c] == window[c]:
                    have += 1

            while have == seen:
                if (r - l + 1) < resLen:
                    res = [l, r]
                    resLen = (r - l + 1)
                window[s[l]] -= 1
                if s[l] in countT and window[s[l]] < countT[s[l]]:
                    have -= 1
                l += 1
        l, r = res
        return s[l:r+1] if resLen != float("infinity") else ""

s = "ADOBECODEBANC"
t = "ABC"
print(minWindow(s, t))


class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        closeToOpen = { ")" : "(", "]" : "[", "}" : "{" }

        for c in s:
            if c in closeToOpen:
                if stack and stack[-1] == closeToOpen[c]:
                    stack.pop()
                else:
                    return False
            else:
                stack.append(c)
        
        return True if not stack else False

def findMin(arr):
    res = arr[0]
    l, r = 0, len(arr) - 1

    while l <= r:
        if arr[l] < arr[r]:
            res = min(res, arr[l])
            break

        m = (l + r) // 2
        res = min(res, arr[m])

        if arr[m] >= arr[l]:
            l = m + 1
        else:
            r = m - 1
    return res


# print(findMin([3,4,5,1,2])) # 1
print(findMin([4,5,6,7,0,1,2])) # 0
# print(findMin([11,13,15,17])) # 11



def search(arr, target):
    l, r = 0, len(arr) - 1

    while l <= r:
        mid = (l + r) // 2

        if arr[mid] == target:
            return mid

        if arr[mid] >= arr[l]:
            if target >= arr[mid] or target < arr[l]:
                l = mid + 1
            else:
                r = mid - 1
        else:
            if target >= arr[mid]:
                l = mid - 1
            else:
                r = mid + 1
    return -1


def reverseList(linkedL):
    prev, curr = None, linkedL

    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev

target = 2
print(search([4,5,6,7,0,1,2], target)) # 4

def reorderList(self, head: Optional[ListNode]) -> None:
        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        second = slow.next
        slow.next = None
        prev = None
        while second:
            temp = second.next
            second.next = prev
            prev = second
            second = temp
        first, second = head, prev
        while second:
            tmp1, tmp2 = first.next, second.next
            first.next = second
            second.next = tmp1
            first, second = tmp1, tmp2

def findElement(arr):
    n = len(arr)
    if n < 3:
        return -1

    left_max = [float('-inf')] * n
    for i in range(1, n):
        l_m = left_max[i - 1]
        a_l = arr[i - 1]
        print(l_m, a_l)
        mr_max = max(left_max[i - 1], arr[i - 1])
        print(mr_max)
        left_max[i] = max(left_max[i - 1], arr[i - 1])

    right_min = [float('inf')] * n
    for i in range(n - 2, -1, -1):
        l_m = right_min[i + 1]
        a_l = arr[i + 1]
        print(l_m, a_l)
        mr_min = min(right_min[i + 1], arr[i + 1])
        print(mr_min)
        right_min[i] = min(right_min[i + 1], arr[i + 1])

    for i in range(1, n-1):
        if left_max[i] < arr[i] < right_min[i]:
            return arr[i]
    return -1

print(findElement([4, 2, 5, 7]))
# print(findElement([98, 40, 65, 59, 27, 20, 45, 87, 34, 99]))

def check_alpha(word):
    return 'a' <= word <= 'z' or 'A' <= word <= 'Z'

def kPangram(s, k):
    no_of = len(s) - s.count(' ')
    if no_of < 26:
        return False
    if k > 25:
        return True
    alpha_set = set('abcdefghijklmnopqrstuvwxyz')
    current_char = set(c.lower() for c in s if check_alpha(c))
    missing = alpha_set - current_char
    return len(missing) <= k


def checkYear (self, n):
        if (n % 400 == 0) and (n % 100 == 0):
            return True
        elif (n % 4 == 0) and (n % 100 != 0):
            return True
        else:
            return False

modifyAndRearrangeArr
# Check two nums are equal if yes put in arr
# Check if not zero
# Add others
# Push zeros

def modifyAndRearrangeArr(arr):
    res = []
    i = 0
    n = len(arr)
    while i < n:
        if i < n - 1 and arr[i] == arr[i + 1]:
            res.append(arr[i] + arr[i + 1])
            i += 2
        elif arr[i] != 0:
            res.append(arr[i])
            i += 1
        else:
            i += 1
    while len(res) < n:
        res.append(0)
    return res

print(modifyAndRearrangeArr([2, 2, 0, 4, 0, 8]))
# [4, 4, 8, 0, 0, 0]
# print(modifyAndRearrangeArr([0, 2, 2, 2, 0, 6, 6, 0, 0, 8]))
# [4, 2, 12, 8, 0, 0, 0, 0, 0, 0]


# def modifyAndRearrangeArr(arr):
#     res = []
#     i = 0
#     n = len(arr)
#     while i < n:
#         if i < n - 1 and arr[i] == arr[i + 1] and arr[i] != 0:
#             res.append(arr[i] + arr[i + 1])
#             i += 2
#         elif arr[i] != 0:
#             res.append(arr[i])
#             i += 1
#         else:
#             i += 1
#     res.extend([0] * (n - len(res)))
#     return res
#
# def modArr(arr):
#     res = []
#     i = 0
#     n = len(arr)
#     while i < n:
#         if i < n - 1 and arr[i] == arr[i + 1] and arr[i] != 0:
#             res.append(arr[i] + arr[i + 1])
#             i += 2
#         elif arr[i] != 0:
#             res.append(arr[i])
#             i += 1
#         else:
#             i += 1
#     res.extend([0] * (n - len(res)))
#     return res

    # i = 0
    # n = len(arr)
    # res = []
    # while i < n:
    #     if i < n - 1 and arr[i] == arr[i + 1] and arr[i] != 0:
    #         res.append(arr[i] + arr[i + 1])
    #         i += 2
    #     elif arr[i] != 0:
    #         res.append(arr[i])
    #         i += 1
    #     else:
    #         i += 1
    # res.extend([0] * (n - len(res)))
    # return res

from numpy.ma.core import max_val, min_val


# print(modifyAndRearrangeArr([0, 2, 2, 2, 0, 6, 6, 0, 0, 8]))
# print(modArr([0, 2, 2, 2, 0, 6, 6, 0, 0, 8]))
# print(modifyAndRearrangeArr([2, 2, 0, 4, 0, 8]))
# print(modifyAndRearrangeArr([2, 2, 0, 4, 0, 8]))
# print(modArr([2, 2, 0, 4, 0, 8]))
# print(modifyAndRearrangeArr([1, 3, 5]))
# print(modifyAndRearrangeArr([]))
# print(modifyAndRearrangeArr([0, 0, 0]))

# def buyAndSell(arr):
#     profit = 0
#     max_val, min_val = 0, float('inf')
#     for i in arr:
#         min_val = min(min_val, i)
#         print(min_val)
#         profit = i - min_val
#         print(profit)
#         max_val = max(max_val, profit)
#         print(max_val)
#     return max_val
#
# print(buyAndSell([7, 10, 1, 3, 6, 9, 2]))
# print(buyAndSell([7, 6, 4, 3, 1]))
# print(buyAndSell([1, 3, 6, 9, 11]))

# Program for top k elements ( bucket sort )
# O(n)
# def topKElements(arr, k):
#     count = {}
#     freq = [[] for i in range(len(arr))]
#     for i in arr:
#         count[i] = 1 + count.get(i, 0)
#     for n, c in count.items():
#         freq[c].append(n)
#     res = []
#     for n in range(len(arr) - 1, 0, -1):
#         print(n, "n", freq[n])
#         for m in freq[n]:
#             res.append(m)
#             if len(res) == k:
#                 return res
        

# arr = [1, 1, 1, 2, 2, 100]
# k = 2
# print(topKElements(arr, k))

# Longest consecutive sequence
# def longestConsecutive(nums):
#     numSet = set(nums)
#     longest = 0
#     for n in nums:
#         if (n - 1) not in numSet:
#             length = 0
#             while (n + length) in numSet:
#                 length += 1
#             longest = max(length, longest)
#     return longest
            
        

# arr = [10, 200, 1, 3, 2]
# print(longestConsecutive(arr))


# def twoSum(arr, target):
#     hashM = {}
#     for i, n in enumerate(arr):
#         diff = target - n
#         if diff in hashM:
#             return [i, arr[i]]
#         hashM[i] = n
#     return -1
# arr = [1, 2, 4, 7, 8]
# target = 5
# print(twoSum(arr, target))

def twoSum(arr, target):
    hashM = {}
    for i, n in enumerate(arr):
        diff = target - n
        if diff in hashM:
            return [hashM[diff], i]
        hashM[n] = i
    return -1
arr = [1, 2, 4, 7, 8]
target = 5
print(twoSum(arr, target))


def maximumProfit(prices):
    res = 0
    l, r = 0, 1
    while r < len(prices):
        if prices[l] <= prices[r]:
            profit = prices[r] - prices[l]
            res = max(res, profit)
        else:
            l = r
        r += 1
    return res
arr = [1, 2, 3, 4, 5, 2]
print(maximumProfit(arr))

def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()
        for i, a in enumerate(nums):
            if a > 0:
                break
            if i > 0 and a == nums[i - 1]:
                continue
            l, r = i + 1, len(nums) - 1
            while l < r:
                threeSum = a + nums[l] + nums[r]
                if threeSum > 0:
                    r -= 1
                elif threeSum < 0:
                    l += 1
                else:
                    res.append([a, nums[l], nums[r]])
                    l += 1
                    r -= 1
                    while nums[l] == nums[l - 1] and l < r:
                        l += 1
        return res

result store original arr sort
check if index is > 0 and the number is not as same as the left one if yes skip it continue
then start two sum two keeping pointers l should be i + 1 and r should be arr of last element
find the current sum by adding arrl, r, and the current for loop val example: a
if the sum is greater reduce r if the sum is lesser update the l then if both alighns what we ant store it in res and
there is a special case
if the number repetitive so again check the while loop with arr[l] arr[l - 1] equal then check whether
it goes out of bounce then return the result


def lengthOfLongestSubstring(self, s: str) -> int:
    charSet = set()
    l = 0
    res = 0
    for r in range(len(s)):
        while s[r] in charSet:
            charSet.remove(s[l])
            l += 1
        charSet.append(s[r])
        res = max(res, l - r + 1)
    return res


def lengthOfLongestSubstring(self, s: str) -> int:
        consec = set()
        l = 0
        res = 0
        for r in range(len(s)):
            while s[r] in consec:
                consec.remove(s[l])
                l += 1
            consec.add(s[r])
            res = max(res, r - l + 1)
        return res


# def checkAnagram(w):
#     res = ""
#     for i in w:
#         r = removeSpecialChars((i))
#         if r:
#             res += i
#     return res == res[::-1]
# def removeSpecialChars(word):
#     return (
#         ord('a') <= ord(word) <= ord('z') or
#         ord('A') <=  ord(word) <=ord('Z') or
#         ord('0') <= ord(word) <= ord('9')
#     )
# words = "A man, nam "
# print(checkAnagram(words))

# def stockBuySell(arr):
# #     right is < left l+= 1
# #     right += 1 right > left check profit
#     maxP = 0
#     l, r = 0, 1
#     while r < len(arr):
#         if arr[l] < arr[r]:
#             profit = arr[r] - arr[l]
#             maxP = max(profit, maxP)
#         else:
#             l = r
#         r += 1
#     return maxP
# arr = [7, 1, 4, 3, 6, 4]
# print(stockBuySell(arr))

# def threeSum(nums, target):
#     res = []
#     nums.sort()
#     for i, t in enumerate(nums):
#         if i > 0 and t == nums[i - 1]:
#             continue
#         l, r = i + 1, len(nums) - 1
#         while l < r:
#             diff = t + nums[l] + nums[r]
#             if diff > 0:
#                 r -= 1
#             elif diff < 0:
#                 l += 1
#             else:
#                 res.append([t, nums[l], nums[r]])
#                 l += 1
#                 while nums[i] == nums[i - 1] and l < r:
#                     l += 1
#     return res
#
# arrE = [-1, 0, 1, 2, -1, -4]
# target = 0
# print(threeSum(arrE, target))

# result store original arr sort
# check if index is > 0 and the number is not as same as the left one if yes skip it continue
# then start two sum two keeping pointers l should be i + 1 and r should be arr of last element
# find the current sum by adding arrl, r, and the current for loop val example: a
# if the sum is greater reduce r if the sum is lesser update the l then if both alighns what we ant store it in res and
# there is a special case
# if the number repetitive so again check the while loop with arr[l] arr[l - 1] equal then check whether
# it goes out of bounce then return the result

# def maxArea(arr):
#     res = 0
#     l, r = 0, len(arr) - 1
#     while l < r:
#         area = (r - l) * min(arr[l], arr[r])
#         res = max(res, area)
#         if arr[l] < arr[r]:
#             l += 1
#         else:
#             r -= 1
#     return res
# arr = [1, 8, 6, 2, 5, 4, 8, 3, 7]
# print(maxArea(arr))

# def twoSum(arr, target):
#     hashM = {}
#     for i, n in enumerate(arr):
#         diff = target - n
#         if diff in hashM:
#             return [i, arr[i]]
#         hashM[i] = n
#     return -1
# arr = [1, 2, 4, 7, 8]
# target = 5
# print(twoSum(arr, target))

# def twoSum(arr, target):
#     hashM = {}
#     for i, n in enumerate(arr):
#         diff = target - n
#         if diff in hashM:
#             return [hashM[diff], i]
#         hashM[n] = i
#     return -1
# arr = [1, 2, 4, 7, 8]
# target = 5
# print(twoSum(arr, target))

# TODO
# Suppose you have a sorted list of 128 names, and you’re searching through it using binary search. What’s the maximum number
#                of steps it would take?
# Suppose you have a sorted list of 128 names, and you’re searching through it using binary search. What’s the maximum number
#                of steps it would take?
# You have a name, and you want to find the person’s phone number in the phone book.
# You have a phone number, and you want to find the person’s name in the phone book. (Hint: You’ll have to search through the
#                whole book!)
# You want to read the numbers of every person in the phone book.
# You want to read the numbers of just the As. (This is a tricky one! It involves concepts that are covered more in chapter 4. Read the answer—you may be surprised!)





# def checkAnagram(w):
#     res = ""
#     for i in w:
#         r = removeSpecialChars((i))
#         if r:
#             res += i
#     return res == res[::-1]
# def removeSpecialChars(word):
#     return (
#         ord('a') <= ord(word) <= ord('z') or
#         ord('A') <=  ord(word) <=ord('Z') or
#         ord('0') <= ord(word) <= ord('9')
#     )
# words = "A man, nam "
# print(checkAnagram(words))

# def stockBuySell(arr):
# #     right is < left l+= 1
# #     right += 1 right > left check profit
#     maxP = 0
#     l, r = 0, 1
#     while r < len(arr):
#         if arr[l] < arr[r]:
#             profit = arr[r] - arr[l]
#             maxP = max(profit, maxP)
#         else:
#             l = r
#         r += 1
#     return maxP
# arr = [7, 1, 4, 3, 6, 4]
# print(stockBuySell(arr))

# def threeSum(nums, target):
#     res = []
#     nums.sort()
#     for i, t in enumerate(nums):
#         if i > 0 and t == nums[i - 1]:
#             continue
#         l, r = i + 1, len(nums) - 1
#         while l < r:
#             diff = t + nums[l] + nums[r]
#             if diff > 0:
#                 r -= 1
#             elif diff < 0:
#                 l += 1
#             else:
#                 res.append([t, nums[l], nums[r]])
#                 l += 1
#                 while nums[i] == nums[i - 1] and l < r:
#                     l += 1
#     return res
#
# arrE = [-1, 0, 1, 2, -1, -4]
# target = 0
# print(threeSum(arrE, target))

# result store original arr sort
# check if index is > 0 and the number is not as same as the left one if yes skip it continue
# then start two sum two keeping pointers l should be i + 1 and r should be arr of last element
# find the current sum by adding arrl, r, and the current for loop val example: a
# if the sum is greater reduce r if the sum is lesser update the l then if both alighns what we ant store it in res and
# there is a special case
# if the number repetitive so again check the while loop with arr[l] arr[l - 1] equal then check whether
# it goes out of bounce then return the result

# def maxArea(arr):
#     res = 0
#     l, r = 0, len(arr) - 1
#     while l < r:
#         area = (r - l) * min(arr[l], arr[r])
#         res = max(res, area)
#         if arr[l] < arr[r]:
#             l += 1
#         else:
#             r -= 1
#     return res
# arr = [1, 8, 6, 2, 5, 4, 8, 3, 7]
# print(maxArea(arr))

# def twoSum(arr, target):
#     hashM = {}
#     for i, n in enumerate(arr):
#         diff = target - n
#         if diff in hashM:
#             return [i, arr[i]]
#         hashM[i] = n
#     return -1
# arr = [1, 2, 4, 7, 8]
# target = 5
# print(twoSum(arr, target))

# def twoSum(arr, target):
#     hashM = {}
#     for i, n in enumerate(arr):
#         diff = target - n
#         if diff in hashM:
#             return [hashM[diff], i]
#         hashM[n] = i
#     return -1
# arr = [1, 2, 4, 7, 8]
# target = 5
# print(twoSum(arr, target))

# TODO
# Suppose you have a sorted list of 128 names, and you’re searching through it using binary search. What’s the maximum number
#                of steps it would take?
# Suppose you have a sorted list of 128 names, and you’re searching through it using binary search. What’s the maximum number
#                of steps it would take?
# You have a name, and you want to find the person’s phone number in the phone book.
# You have a phone number, and you want to find the person’s name in the phone book. (Hint: You’ll have to search through the
#                whole book!)
# You want to read the numbers of every person in the phone book.
# You want to read the numbers of just the As. (This is a tricky one! It involves concepts that are covered more in chapter 4. Read the answer—you may be surprised!)


class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        count = {}
        res = 0
        l = 0
        maxF = 0
        for r in range(len(s)):
            count[s[r]] = 1 + count.get(s[r], 0)
            maxF = max(maxF, count[s[r]])
            while (r - l + 1) - maxF > k:
                count[s[l]] -= 1
                l += 1
            res = max(res, r - l + 1)
        return res

def isValid(s):
        stack = []
        checkClose = {')' : '(', '}' : '{', ']' : '['}
        for c in s:
            if c in checkClose:
                if stack and stack[-1] == checkClose[c]:
                    stack.pop()
                else:
                    return False
            else:
                stack.append(c)
        return True if not stack else False


def minWindow(self, s: str, t: str) -> str:
        if t == "":
            return ""
        window, countT = {}, {}
        for c in t:
            countT[c] = 1 + countT.get(c, 0)
        
        have, seen = 0, len(countT)
        res, resLen = [-1, -1], float("infinity")
        l = 0
        for r in range(len(s)):
            c = s[r]
            window[c] = 1 + window.get(c, 0)

            if c in countT and window[c] == countT[c]:
                have += 1

            while have == seen:
                if (r - l + 1) < resLen:
                    res = [l, r]
                    resLen = r - l + 1
                window[s[l]] -= 1

                if s[l] in countT and window[s[l]] < countT[s[l]]:
                    have -= 1
                l += 1
        l, r = res
        return s[l : r + 1] if resLen != float("infinity") else ""
