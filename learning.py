def topKElements(arr, k):
    count = {}
    freq = [[] for i in range(len(arr))]
    for i in arr:
        count[i] = 1 + count.get(i, 0)
    for n, c in count.items():
        freq[c].append(n)
    res = []
    for n in range(len(arr) - 1, 0, -1):
        print(n, "n", freq[n])
        for k in freq[n]:
            res.append(k)
            


arr = [1, 1, 1, 2, 2, 100]
k = 2
print(topKElements(arr, k))