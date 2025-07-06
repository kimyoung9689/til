

---

## 📋 목차
1. [시간복잡도 & 공간복잡도](#1-시간복잡도--공간복잡도)
2. [기본 자료구조](#2-기본-자료구조)
3. [고급 자료구조](#3-고급-자료구조)
4. [정렬 알고리즘](#4-정렬-알고리즘)
5. [검색 알고리즘](#5-검색-알고리즘)
6. [그래프 알고리즘](#6-그래프-알고리즘)
7. [동적 프로그래밍](#7-동적-프로그래밍)
8. [문자열 알고리즘](#8-문자열-알고리즘)
9. [실무 핵심 팁](#9-실무-핵심-팁)

---

## 1. 시간복잡도 & 공간복잡도

### 빅오 표기법 (Big-O Notation)
```
O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2^n) < O(n!)
```

### 실무에서 자주 보는 복잡도들
- **O(1)**: 해시 테이블 접근, 배열 인덱스 접근
- **O(log n)**: 이진 탐색, 균형 트리 탐색
- **O(n)**: 선형 탐색, 배열 순회
- **O(n log n)**: 효율적인 정렬 (퀵정렬, 머지정렬)
- **O(n²)**: 버블정렬, 선택정렬, 중첩 루프

### 팁
> **실무에서는 O(n²) 이상은 웬만하면 피해라.** 데이터가 조금만 커져도 시스템이 터진다.

---

## 2. 기본 자료구조

### 2.1 배열 (Array)
```python
# 장점: O(1) 접근, 캐시 효율성 좋음
# 단점: 크기 고정, 중간 삽입/삭제 비용 높음
arr = [1, 2, 3, 4, 5]
print(arr[2])  # O(1)
```

**실무 사용 예시:**
- 고정된 크기의 데이터 저장
- 인덱스 기반 빠른 접근이 필요한 경우
- 수학적 계산, 이미지 처리

### 2.2 연결 리스트 (Linked List)
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# 장점: 동적 크기, 삽입/삭제 O(1)
# 단점: 임의 접근 O(n), 메모리 오버헤드
```

**실무 사용 예시:**
- 크기가 자주 변하는 데이터
- 삽입/삭제가 빈번한 경우
- 메모리 효율성이 중요한 경우

### 2.3 스택 (Stack) - LIFO
```python
stack = []
stack.append(1)    # push - O(1)
stack.append(2)    # push - O(1)
top = stack.pop()  # pop - O(1)
```

**실무 사용 예시:**
- 함수 호출 관리 (Call Stack)
- 백트래킹 알고리즘
- 괄호 검사, 수식 계산
- 브라우저 뒤로가기 기능

### 2.4 큐 (Queue) - FIFO
```python
from collections import deque
queue = deque()
queue.append(1)      # enqueue - O(1)
queue.append(2)      # enqueue - O(1)
front = queue.popleft()  # dequeue - O(1)
```

**실무 사용 예시:**
- 작업 스케줄링
- BFS 탐색
- 프린터 대기열
- 네트워크 패킷 처리

### 2.5 해시 테이블 (Hash Table)
```python
# Python의 dict는 해시 테이블 구현
hash_table = {}
hash_table['key1'] = 'value1'  # O(1) 평균
value = hash_table['key1']     # O(1) 평균
```

**실무 사용 예시:**
- 데이터베이스 인덱싱
- 캐싱 시스템
- 중복 검사
- 빠른 조회가 필요한 모든 경우

### 팁
> **해시 테이블은 실무에서 가장 자주 쓰는 자료구조다.** 언어별로 구현 방식을 정확히 알아두자.

---

## 3. 고급 자료구조

### 3.1 이진 탐색 트리 (Binary Search Tree)
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 장점: 정렬된 상태 유지, 탐색/삽입/삭제 O(log n)
# 단점: 불균형 시 O(n)으로 퇴화
```

**실무 사용 예시:**
- 데이터베이스 인덱싱
- 파일 시스템
- 표현식 파싱

### 3.2 균형 트리 (AVL Tree, Red-Black Tree)
```python
# 자동으로 균형을 맞춰 항상 O(log n) 보장
# Python의 bisect 모듈 내부적으로 사용
```

**실무 사용 예시:**
- 데이터베이스 B-Tree 인덱스
- 언어 내장 정렬 컨테이너
- 실시간 순위 시스템

### 3.3 힙 (Heap)
```python
import heapq

# 최소 힙 (Python 기본)
min_heap = []
heapq.heappush(min_heap, 3)  # O(log n)
heapq.heappush(min_heap, 1)  # O(log n)
min_val = heapq.heappop(min_heap)  # O(log n)

# 최대 힙 (음수 활용)
max_heap = []
heapq.heappush(max_heap, -3)
max_val = -heapq.heappop(max_heap)
```

**실무 사용 예시:**
- 우선순위 큐
- 작업 스케줄링
- 다익스트라 알고리즘
- 실시간 데이터 처리

### 3.4 트라이 (Trie)
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
```

**실무 사용 예시:**
- 자동완성 기능
- 사전 검색
- IP 라우팅 테이블
- 문자열 매칭

###  팁
> **힙은 실무에서 정말 유용하다.** 특히 실시간 시스템에서 우선순위 처리할 때 필수다.

---

## 4. 정렬 알고리즘

### 4.1 버블 정렬 (Bubble Sort) - O(n²)
```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
```
**실무 평가**: 절대 쓰지 마라. 면접용으로만 알아두자.

### 4.2 퀵 정렬 (Quick Sort) - O(n log n) 평균
```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)
```
**실무 평가**: ⭐⭐⭐ 많이 사용됨. 평균적으로 빠름.

### 4.3 머지 정렬 (Merge Sort) - O(n log n) 보장
```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```
**실무 평가**: ⭐⭐⭐⭐ 안정적이고 예측 가능한 성능.

### 4.4 힙 정렬 (Heap Sort) - O(n log n)
```python
def heap_sort(arr):
    import heapq
    heapq.heapify(arr)  # O(n)
    return [heapq.heappop(arr) for _ in range(len(arr))]  # O(n log n)
```
**실무 평가**: ⭐⭐⭐ 메모리 효율적이지만 캐시 성능이 아쉬움.

###  팁
> **실무에서는 언어 내장 정렬을 써라.** Python의 `sorted()`, Java의 `Arrays.sort()` 등은 이미 최적화되어 있다.

---

## 5. 검색 알고리즘

### 5.1 선형 탐색 (Linear Search) - O(n)
```python
def linear_search(arr, target):
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1
```
**실무 사용**: 작은 데이터셋, 정렬되지 않은 데이터

### 5.2 이진 탐색 (Binary Search) - O(log n)
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Python 내장 모듈 사용
import bisect
index = bisect.bisect_left(arr, target)
```
**실무 사용**: 정렬된 대용량 데이터, 데이터베이스 인덱싱

### 5.3 해시 기반 검색 - O(1) 평균
```python
# 해시 테이블 활용
data = {'apple': 100, 'banana': 200, 'cherry': 300}
price = data.get('apple', 0)  # O(1) 평균
```

###  팁
> **이진 탐색은 반드시 정렬된 배열에서만 사용해야 한다.** 실무에서 이걸 놓치는 경우가 많다.

---

## 6. 그래프 알고리즘

### 6.1 그래프 표현 방법
```python
# 인접 리스트 (추천)
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# 인접 행렬 (밀집 그래프에서 사용)
adj_matrix = [
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
]
```

### 6.2 깊이 우선 탐색 (DFS)
```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    
    visited.add(start)
    print(start)
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    
    return visited

# 반복 버전 (스택 사용)
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            print(node)
            stack.extend(graph[node])
    
    return visited
```

### 6.3 너비 우선 탐색 (BFS)
```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        node = queue.popleft()
        print(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return visited
```

### 6.4 다익스트라 알고리즘 (최단 경로)
```python
import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances
```

### 6.5 위상 정렬 (Topological Sort)
```python
from collections import deque, defaultdict

def topological_sort(graph):
    in_degree = defaultdict(int)
    
    # 진입 차수 계산
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1
    
    # 진입 차수가 0인 노드들로 시작
    queue = deque([node for node in graph if in_degree[node] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result
```

### 팁
> **그래프 알고리즘은 실무에서 정말 많이 쓰인다.** 특히 소셜 네트워크, 추천 시스템, 의존성 해결 등에서 필수다.

---

## 7. 동적 프로그래밍

### 7.1 피보나치 수열 (기본 예제)
```python
# 재귀 (비효율적) - O(2^n)
def fib_recursive(n):
    if n <= 1:
        return n
    return fib_recursive(n-1) + fib_recursive(n-2)

# 메모이제이션 (Top-down) - O(n)
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# 타뷸레이션 (Bottom-up) - O(n)
def fib_tabulation(n):
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

# 공간 최적화 - O(1) 공간
def fib_optimized(n):
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b
```

### 7.2 배낭 문제 (Knapsack Problem)
```python
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # 현재 아이템을 포함하지 않는 경우
            dp[i][w] = dp[i-1][w]
            
            # 현재 아이템을 포함하는 경우
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], 
                              dp[i-1][w-weights[i-1]] + values[i-1])
    
    return dp[n][capacity]
```

### 7.3 최장 공통 부분 수열 (LCS)
```python
def lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

### 팁
> **DP는 최적화 문제에서 핵심이다.** 특히 비즈니스 로직에서 최적해를 구할 때 자주 사용한다.

---

## 8. 문자열 알고리즘

### 8.1 KMP 알고리즘 (문자열 매칭)
```python
def kmp_search(text, pattern):
    def compute_lps(pattern):
        lps = [0] * len(pattern)
        length = 0
        i = 1
        
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps
    
    lps = compute_lps(pattern)
    i = j = 0
    matches = []
    
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
        
        if j == len(pattern):
            matches.append(i - j)
            j = lps[j - 1]
        elif i < len(text) and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return matches
```

### 8.2 라빈-카프 알고리즘 (해시 기반 매칭)
```python
def rabin_karp(text, pattern):
    BASE = 256
    MOD = 101
    
    m = len(pattern)
    n = len(text)
    
    pattern_hash = 0
    text_hash = 0
    h = 1
    
    # h = BASE^(m-1) % MOD
    for i in range(m - 1):
        h = (h * BASE) % MOD
    
    # 패턴과 첫 번째 윈도우의 해시 계산
    for i in range(m):
        pattern_hash = (BASE * pattern_hash + ord(pattern[i])) % MOD
        text_hash = (BASE * text_hash + ord(text[i])) % MOD
    
    matches = []
    
    for i in range(n - m + 1):
        if pattern_hash == text_hash:
            # 해시가 같으면 문자열 비교
            if text[i:i+m] == pattern:
                matches.append(i)
        
        # 다음 윈도우의 해시 계산
        if i < n - m:
            text_hash = (BASE * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % MOD
            if text_hash < 0:
                text_hash += MOD
    
    return matches
```

#### 팁
> **문자열 알고리즘은 검색 엔진, 로그 분석, 보안 등에서 필수다.** 특히 대용량 텍스트 처리에서 성능 차이가 크다.

---

## 9. 실무 핵심 팁

### 9.1 언어별 최적화 팁

#### Python
```python
# 리스트 컴프리헨션 사용
squares = [x**2 for x in range(10)]  # 빠름
squares = []
for x in range(10):
    squares.append(x**2)  # 느림

# collections 모듈 활용
from collections import defaultdict, Counter, deque
counter = Counter([1, 2, 2, 3, 3, 3])
```

#### Java
```java
// StringBuilder 사용
StringBuilder sb = new StringBuilder();
sb.append("Hello").append(" World");  // 빠름

String str = "";
str += "Hello";
str += " World";  // 느림 (새로운 객체 생성)

// ArrayList vs LinkedList
List<Integer> list = new ArrayList<>();  // 인덱스 접근 빠름
List<Integer> list = new LinkedList<>();  // 삽입/삭제 빠름
```

### 9.2 성능 최적화 체크리스트

#### 1. 시간 복잡도 최적화
- [ ] 중첩 루프 최소화
- [ ] 적절한 자료구조 선택
- [ ] 불필요한 연산 제거

#### 2. 공간 복잡도 최적화
- [ ] 메모리 재사용
- [ ] 불필요한 데이터 저장 방지
- [ ] 제자리 알고리즘 사용 검토

#### 3. 캐시 효율성
- [ ] 지역성 활용
- [ ] 연속된 메모리 접근
- [ ] 적절한 데이터 레이아웃

### 9.3 실무 문제 해결 패턴

#### 1. 두 포인터 기법
```python
def two_sum_sorted(arr, target):
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []
```

#### 2. 슬라이딩 윈도우
```python
def max_sum_subarray(arr, k):
    if len(arr) < k:
        return None
    
    # 첫 번째 윈도우의 합
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # 윈도우를 슬라이딩하며 최대값 찾기
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i-k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum
```

#### 3. 분할 정복
```python
def max_subarray_sum(arr):
    def divide_conquer(arr, left, right):
        if left == right:
            return arr[left]
        
        mid = (left + right) // 2
        
        left_sum = divide_conquer(arr, left, mid)
        right_sum = divide_conquer(arr, mid + 1, right)
        
        # 교차 부분의 최대 합
        left_cross = float('-inf')
        temp_sum = 0
        for i in range(mid, left - 1, -1):
            temp_sum += arr[i]
            left_cross = max(left_cross, temp_sum)
        
        right_cross = float('-inf')
        temp_sum = 0
        for i in range(mid + 1, right + 1):
            temp_sum += arr[i]
            right_cross = max(right_cross, temp_sum)
        
        cross_sum = left_cross + right_cross
        
        return max(left_sum, right_sum, cross_sum)
    
    return divide_conquer(arr, 0, len(arr) - 1)
```

### 9.4 디버깅 및 테스트 전략

#### 1. 단위 테스트 작성
```python
def test_binary_search():
    # 정상 케이스
    assert binary_search([1, 2, 3, 4, 5], 3) == 2
    
    # 경계 케이스
    assert binary_search([1], 1) == 0
    assert binary_search([], 1) == -1
    
    # 예외 케이스
    assert binary_search([1, 2, 3], 4) == -1
```

#### 2. 성능 측정
```python
import time
import cProfile

def measure_time(func, *args):
    start = time.time()
    result = func(*args)
    end = time.time()
    print(f"실행 시간: {end - start:.4f}초")
    return result

# 프로파일링
cProfile.run('your_function()')
```

### 9.5 실무에서 자주 쓰는 라이브러리

#### Python
```python
# 기본 라이브러리
import heapq          # 힙 구현
import bisect         # 이진 탐색
import collections    # 고급 자료구조
import itertools      # 순열, 조합
import functools      # 메모이제이션 (@lru_cache)

# 수치 계산
import numpy as np    # 배열 연산
import pandas as pd   # 데이터 분석

# 그래프 알고리즘
import networkx as nx # 그래프 라이브러리
```

#### Java
```java
// 유틸리티 클래스
import java.util.*;
import java.util.stream.*;

// 고성능 컬렉션
import java.util.concurrent.*;
import gnu.trove.*;  // 원시 타입 컬렉션
```

### 🔥 최종 전문가 조언

#### 1. 실무에서 중요한 순서
1. **정확성** - 먼저 작동하는 코드를 작성
2. **가독성** - 동료가 이해할 수 있는 코드
3. **유지보수성** - 나중에 수정하기 쉬운 코드
4. **성능** - 필요할 때만 최적화

#### 2. 알고리즘 선택 가이드
```python
# 데이터 크기별 알고리즘 선택
def choose_algorithm(data_size):
    if data_size < 100:
        return "간단한 알고리즘도 OK (O(n²))"
    elif data_size < 10000:
        return "효율적인 알고리즘 필요 (O(n log n))"
    else:
        return "고도로 최적화된 알고리즘 필수 (O(n), O(log n))"
```

#### 3. 자료구조 선택 기준
| 상황 | 추천 자료구조 | 이유 |
|------|---------------|------|
| 빠른 검색 | Hash Table | O(1) 평균 검색 |
| 순서 유지 + 빠른 검색 | Balanced Tree | O(log n) 검색 |
| 우선순위 처리 | Heap | O(log n) 삽입/삭제 |
| 최근 데이터 접근 | Stack/Queue | LIFO/FIFO |
| 범위 검색 | B-Tree | 범위 쿼리 효율적 |

#### 4. 실무 면접 대비 핵심 문제들
```python
# 1. 배열에서 두 수의 합 찾기
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# 2. 연결 리스트 사이클 검출
def has_cycle(head):
    if not head or not head.next:
        return False
    
    slow = head
    fast = head.next
    
    while fast and fast.next:
        if slow == fast:
            return True
        slow = slow.next
        fast = fast.next.next
    
    return False

# 3. 유효한 괄호 검사
def is_valid_parentheses(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    
    return not stack

# 4. 최대 서브어레이 합 (카데인 알고리즘)
def max_subarray(nums):
    max_sum = current_sum = nums[0]
    
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# 5. 이진 트리 레벨 순회
def level_order(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
```

#### 5. 성능 최적화 실전 팁
```python
# 1. 조기 종료 활용
def find_first_duplicate(arr):
    seen = set()
    for num in arr:
        if num in seen:
            return num  # 찾자마자 즉시 반환
        seen.add(num)
    return None

# 2. 메모이제이션 활용
from functools import lru_cache

@lru_cache(maxsize=None)
def expensive_function(n):
    if n <= 1:
        return n
    return expensive_function(n-1) + expensive_function(n-2)

# 3. 제너레이터 사용 (메모리 효율성)
def fibonacci_generator():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# 4. 비트 연산 활용
def count_set_bits(n):
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count

# 더 효율적인 방법
def count_set_bits_optimized(n):
    count = 0
    while n:
        n &= n - 1  # 가장 오른쪽 1비트 제거
        count += 1
    return count
```

#### 6. 실무 코딩 스타일 가이드
```python
# 나쁜 예
def f(l):
    r = []
    for i in l:
        if i % 2 == 0:
            r.append(i * 2)
    return r

# 좋은 예
def double_even_numbers(numbers):
    """주어진 숫자 리스트에서 짝수만 골라 2배로 만들어 반환"""
    doubled_evens = []
    for number in numbers:
        if number % 2 == 0:
            doubled_evens.append(number * 2)
    return doubled_evens

# 더 좋은 예 (함수형 스타일)
def double_even_numbers_functional(numbers):
    """주어진 숫자 리스트에서 짝수만 골라 2배로 만들어 반환"""
    return [num * 2 for num in numbers if num % 2 == 0]
```

#### 7. 디버깅 전략
```python
# 1. 단계별 로깅
def binary_search_with_logging(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        print(f"탐색 범위: [{left}, {right}], 중간값: {arr[mid]}")
        
        if arr[mid] == target:
            print(f"발견: 인덱스 {mid}")
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    print("찾지 못함")
    return -1

# 2. 불변 조건 검사
def merge_sort_with_invariant(arr):
    """정렬 과정에서 불변 조건 검사"""
    def is_sorted(arr):
        return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))
    
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort_with_invariant(arr[:mid])
    right = merge_sort_with_invariant(arr[mid:])
    
    # 불변 조건: 각 부분이 정렬되어 있어야 함
    assert is_sorted(left), "왼쪽 부분이 정렬되지 않음"
    assert is_sorted(right), "오른쪽 부분이 정렬되지 않음"
    
    result = merge(left, right)
    
    # 불변 조건: 결과가 정렬되어 있어야 함
    assert is_sorted(result), "최종 결과가 정렬되지 않음"
    
    return result
```

#### 8. 대용량 데이터 처리 패턴
```python
# 1. 청크 단위 처리
def process_large_file(filename, chunk_size=1024):
    with open(filename, 'r') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            # 청크 단위로 처리
            process_chunk(chunk)

# 2. 스트리밍 처리
def process_stream(data_stream):
    for item in data_stream:
        # 메모리에 모든 데이터를 로드하지 않고 처리
        result = process_item(item)
        yield result

# 3. 배치 처리
def batch_process(items, batch_size=100):
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        process_batch(batch)
```

#### 9. 병렬 처리 기초
```python
# 1. 멀티스레딩 (I/O 바운드 작업)
import concurrent.futures
import requests

def fetch_url(url):
    response = requests.get(url)
    return response.status_code

urls = ['http://example.com', 'http://google.com', 'http://github.com']

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(fetch_url, urls))

# 2. 멀티프로세싱 (CPU 바운드 작업)
import multiprocessing

def cpu_bound_task(n):
    return sum(i * i for i in range(n))

if __name__ == '__main__':
    with multiprocessing.Pool() as pool:
        results = pool.map(cpu_bound_task, [1000000, 2000000, 3000000])
```

#### 10. 실무 최적화 체크리스트

##### 코드 리뷰 시 확인사항
- [ ] 시간 복잡도가 적절한가?
- [ ] 공간 복잡도가 효율적인가?
- [ ] 엣지 케이스 처리가 되어있는가?
- [ ] 코드가 읽기 쉬운가?
- [ ] 테스트 케이스가 충분한가?

##### 성능 개선 우선순위
1. **알고리즘 개선** - 가장 큰 성능 향상
2. **자료구조 최적화** - 메모리 및 접근 성능
3. **코드 최적화** - 세부 구현 개선
4. **하드웨어 업그레이드** - 마지막 수단

---




**알고리즘과 자료구조는 프로그래머의 기본 소양이다. 하지만 이것들은 수단이지 목적이 아니다. 진짜 실력은 이 도구들을 언제, 어떻게 사용할지 아는 것이다.


