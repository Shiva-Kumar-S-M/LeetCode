# LeetCode

A collection of my LeetCode solutions in Python, organized by problem topic instead of by submission order.

## Purpose

This repository is where I keep my LeetCode practice as I work through data structures and algorithms. Grouping solutions by topic makes it easier to review a specific pattern (for example, all sliding window problems) instead of scrolling through solutions in the order they were solved.

## Topics Covered

- Array
- Binary Search
- Bit Manipulation
- Dynamic Programming
- Enumeration
- Graph
- Greedy
- Hash Table
- Heap
- Linked List
- Math
- Prefix Sum
- Segment Tree
- Sliding Window
- Sorting
- Stack
- String
- Trees
- Trie
- Two Pointers

## Repository Structure

```
LeetCode/
├── array/
├── binary_search/
├── bit_manipulation/
├── dynamic_programming/
├── enumeration/
├── graph/
├── greedy/
├── hash_table/
├── heap/
├── linked_list/
├── math/
├── prefix_sum/
├── segment_tree/
├── sliding_window/
├── sorting/
├── stack/
├── string/
├── trees/
├── trie/
├── two_pointers/
├── .gitignore
├── LICENSE
└── README.md
```

Each topic folder only exists if there is at least one solved problem for it.

## How Solutions Are Organized

- Every problem has its own folder, named as `NNNN-problem-slug`, where `NNNN` is the LeetCode problem number.
- Inside each problem folder:
  - `NNNN-problem-slug.py` — the solution.
  - `README.md` — the original problem statement, as provided by LeetCode.
- A problem folder lives under the topic folder that best matches the technique used in its solution (for example, a two-pointer solution goes under `two_pointers/`, a solution using a segment tree goes under `segment_tree/`).
- A few problems have more than one solution with a different approach (for example, a brute-force version alongside an optimized one). In those cases, both files are kept in the same problem folder with a suffix describing the approach, such as `-brute-force` or `-sorting`.

## Python Version

The repository does not declare a specific Python version (no `requirements.txt` or version file). Based on the type hints used across the solutions (`typing.List`, and in some files the newer built-in generics like `list[int]`), the code targets Python 3.9 or newer.

## Progress

- Problems solved: 100
- Topics covered: 20

<!---LeetCode Topics Start-->
# LeetCode Topics
## Hash Table
|  |
| ------- |
| [1096-brace-expansion-ii](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1096-brace-expansion-ii) |
| [1807-evaluate-the-bracket-pairs-of-a-string](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1807-evaluate-the-bracket-pairs-of-a-string) |
| [2897-apply-operations-on-array-to-maximize-sum-of-squares](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2897-apply-operations-on-array-to-maximize-sum-of-squares) |
| [2913-subarrays-distinct-element-sum-of-squares-i](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2913-subarrays-distinct-element-sum-of-squares-i) |
| [2916-subarrays-distinct-element-sum-of-squares-ii](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2916-subarrays-distinct-element-sum-of-squares-ii) |
| [3551-minimum-swaps-to-sort-by-digit-sum](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/3551-minimum-swaps-to-sort-by-digit-sum) |
## String
|  |
| ------- |
| [0020-valid-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0020-valid-parentheses) |
| [0022-generate-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0022-generate-parentheses) |
| [0032-longest-valid-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0032-longest-valid-parentheses) |
| [0301-remove-invalid-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0301-remove-invalid-parentheses) |
| [0678-valid-parenthesis-string](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0678-valid-parenthesis-string) |
| [0856-score-of-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0856-score-of-parentheses) |
| [0921-minimum-add-to-make-parentheses-valid](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0921-minimum-add-to-make-parentheses-valid) |
| [1021-remove-outermost-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1021-remove-outermost-parentheses) |
| [1096-brace-expansion-ii](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1096-brace-expansion-ii) |
| [1111-maximum-nesting-depth-of-two-valid-parentheses-strings](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1111-maximum-nesting-depth-of-two-valid-parentheses-strings) |
| [1190-reverse-substrings-between-each-pair-of-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1190-reverse-substrings-between-each-pair-of-parentheses) |
| [1541-minimum-insertions-to-balance-a-parentheses-string](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1541-minimum-insertions-to-balance-a-parentheses-string) |
| [1614-maximum-nesting-depth-of-the-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1614-maximum-nesting-depth-of-the-parentheses) |
| [1807-evaluate-the-bracket-pairs-of-a-string](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1807-evaluate-the-bracket-pairs-of-a-string) |
| [2409-count-days-spent-together](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2409-count-days-spent-together) |
| [3863-minimum-operations-to-sort-a-string](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/3863-minimum-operations-to-sort-a-string) |
## Backtracking
|  |
| ------- |
| [0022-generate-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0022-generate-parentheses) |
| [0301-remove-invalid-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0301-remove-invalid-parentheses) |
| [1096-brace-expansion-ii](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1096-brace-expansion-ii) |
## Stack
|  |
| ------- |
| [0020-valid-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0020-valid-parentheses) |
| [0032-longest-valid-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0032-longest-valid-parentheses) |
| [0678-valid-parenthesis-string](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0678-valid-parenthesis-string) |
| [0856-score-of-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0856-score-of-parentheses) |
| [0921-minimum-add-to-make-parentheses-valid](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0921-minimum-add-to-make-parentheses-valid) |
| [1021-remove-outermost-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1021-remove-outermost-parentheses) |
| [1096-brace-expansion-ii](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1096-brace-expansion-ii) |
| [1111-maximum-nesting-depth-of-two-valid-parentheses-strings](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1111-maximum-nesting-depth-of-two-valid-parentheses-strings) |
| [1190-reverse-substrings-between-each-pair-of-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1190-reverse-substrings-between-each-pair-of-parentheses) |
| [1541-minimum-insertions-to-balance-a-parentheses-string](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1541-minimum-insertions-to-balance-a-parentheses-string) |
| [1614-maximum-nesting-depth-of-the-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1614-maximum-nesting-depth-of-the-parentheses) |
## Breadth-First Search
|  |
| ------- |
| [0301-remove-invalid-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0301-remove-invalid-parentheses) |
| [1096-brace-expansion-ii](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1096-brace-expansion-ii) |
## Sorting
|  |
| ------- |
| [1096-brace-expansion-ii](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1096-brace-expansion-ii) |
| [2089-find-target-indices-after-sorting-array](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2089-find-target-indices-after-sorting-array) |
| [2333-minimum-sum-of-squared-difference](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2333-minimum-sum-of-squared-difference) |
| [3551-minimum-swaps-to-sort-by-digit-sum](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/3551-minimum-swaps-to-sort-by-digit-sum) |
| [3727-maximum-alternating-sum-of-squares](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/3727-maximum-alternating-sum-of-squares) |
| [3886-sum-of-sortable-integers](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/3886-sum-of-sortable-integers) |
## Array
|  |
| ------- |
| [1569-number-of-ways-to-reorder-array-to-get-same-bst](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1569-number-of-ways-to-reorder-array-to-get-same-bst) |
| [1807-evaluate-the-bracket-pairs-of-a-string](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1807-evaluate-the-bracket-pairs-of-a-string) |
| [1929-concatenation-of-array](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1929-concatenation-of-array) |
| [2089-find-target-indices-after-sorting-array](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2089-find-target-indices-after-sorting-array) |
| [2267-check-if-there-is-a-valid-parentheses-string-path](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2267-check-if-there-is-a-valid-parentheses-string-path) |
| [2333-minimum-sum-of-squared-difference](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2333-minimum-sum-of-squared-difference) |
| [2778-sum-of-squares-of-special-elements](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2778-sum-of-squares-of-special-elements) |
| [2897-apply-operations-on-array-to-maximize-sum-of-squares](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2897-apply-operations-on-array-to-maximize-sum-of-squares) |
| [2913-subarrays-distinct-element-sum-of-squares-i](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2913-subarrays-distinct-element-sum-of-squares-i) |
| [2916-subarrays-distinct-element-sum-of-squares-ii](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2916-subarrays-distinct-element-sum-of-squares-ii) |
| [3551-minimum-swaps-to-sort-by-digit-sum](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/3551-minimum-swaps-to-sort-by-digit-sum) |
| [3644-maximum-k-to-sort-a-permutation](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/3644-maximum-k-to-sort-a-permutation) |
| [3727-maximum-alternating-sum-of-squares](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/3727-maximum-alternating-sum-of-squares) |
| [3886-sum-of-sortable-integers](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/3886-sum-of-sortable-integers) |
| [3942-minimum-operations-to-sort-a-permutation](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/3942-minimum-operations-to-sort-a-permutation) |
## Bracket Sequences
|  |
| ------- |
| [0020-valid-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0020-valid-parentheses) |
| [0022-generate-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0022-generate-parentheses) |
| [0032-longest-valid-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0032-longest-valid-parentheses) |
| [0678-valid-parenthesis-string](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0678-valid-parenthesis-string) |
| [0856-score-of-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0856-score-of-parentheses) |
| [0921-minimum-add-to-make-parentheses-valid](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0921-minimum-add-to-make-parentheses-valid) |
| [1021-remove-outermost-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1021-remove-outermost-parentheses) |
| [1111-maximum-nesting-depth-of-two-valid-parentheses-strings](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1111-maximum-nesting-depth-of-two-valid-parentheses-strings) |
| [1190-reverse-substrings-between-each-pair-of-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1190-reverse-substrings-between-each-pair-of-parentheses) |
| [1541-minimum-insertions-to-balance-a-parentheses-string](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1541-minimum-insertions-to-balance-a-parentheses-string) |
| [1614-maximum-nesting-depth-of-the-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1614-maximum-nesting-depth-of-the-parentheses) |
| [2267-check-if-there-is-a-valid-parentheses-string-path](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2267-check-if-there-is-a-valid-parentheses-string-path) |
## Dynamic Programming
|  |
| ------- |
| [0022-generate-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0022-generate-parentheses) |
| [0032-longest-valid-parentheses](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0032-longest-valid-parentheses) |
| [0678-valid-parenthesis-string](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0678-valid-parenthesis-string) |
| [1569-number-of-ways-to-reorder-array-to-get-same-bst](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1569-number-of-ways-to-reorder-array-to-get-same-bst) |
| [2267-check-if-there-is-a-valid-parentheses-string-path](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2267-check-if-there-is-a-valid-parentheses-string-path) |
## Matrix
|  |
| ------- |
| [2267-check-if-there-is-a-valid-parentheses-string-path](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2267-check-if-there-is-a-valid-parentheses-string-path) |
## Greedy
|  |
| ------- |
| [0678-valid-parenthesis-string](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0678-valid-parenthesis-string) |
| [0921-minimum-add-to-make-parentheses-valid](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/0921-minimum-add-to-make-parentheses-valid) |
| [1541-minimum-insertions-to-balance-a-parentheses-string](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1541-minimum-insertions-to-balance-a-parentheses-string) |
| [2333-minimum-sum-of-squared-difference](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2333-minimum-sum-of-squared-difference) |
| [2897-apply-operations-on-array-to-maximize-sum-of-squares](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2897-apply-operations-on-array-to-maximize-sum-of-squares) |
| [3723-maximize-sum-of-squares-of-digits](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/3723-maximize-sum-of-squares-of-digits) |
| [3727-maximum-alternating-sum-of-squares](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/3727-maximum-alternating-sum-of-squares) |
## Simulation
|  |
| ------- |
| [1929-concatenation-of-array](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1929-concatenation-of-array) |
## Binary Search
|  |
| ------- |
| [2089-find-target-indices-after-sorting-array](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2089-find-target-indices-after-sorting-array) |
| [2333-minimum-sum-of-squared-difference](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2333-minimum-sum-of-squared-difference) |
## Heap (Priority Queue)
|  |
| ------- |
| [2333-minimum-sum-of-squared-difference](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2333-minimum-sum-of-squared-difference) |
## Enumeration
|  |
| ------- |
| [2778-sum-of-squares-of-special-elements](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2778-sum-of-squares-of-special-elements) |
| [3886-sum-of-sortable-integers](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/3886-sum-of-sortable-integers) |
## Bit Manipulation
|  |
| ------- |
| [2897-apply-operations-on-array-to-maximize-sum-of-squares](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2897-apply-operations-on-array-to-maximize-sum-of-squares) |
| [3644-maximum-k-to-sort-a-permutation](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/3644-maximum-k-to-sort-a-permutation) |
## Math
|  |
| ------- |
| [1569-number-of-ways-to-reorder-array-to-get-same-bst](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1569-number-of-ways-to-reorder-array-to-get-same-bst) |
| [2409-count-days-spent-together](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2409-count-days-spent-together) |
| [3723-maximize-sum-of-squares-of-digits](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/3723-maximize-sum-of-squares-of-digits) |
| [3886-sum-of-sortable-integers](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/3886-sum-of-sortable-integers) |
## Binary Indexed Tree
|  |
| ------- |
| [2913-subarrays-distinct-element-sum-of-squares-i](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2913-subarrays-distinct-element-sum-of-squares-i) |
| [2916-subarrays-distinct-element-sum-of-squares-ii](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2916-subarrays-distinct-element-sum-of-squares-ii) |
## Segment Tree
|  |
| ------- |
| [2913-subarrays-distinct-element-sum-of-squares-i](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2913-subarrays-distinct-element-sum-of-squares-i) |
| [2916-subarrays-distinct-element-sum-of-squares-ii](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/2916-subarrays-distinct-element-sum-of-squares-ii) |
## Divide and Conquer
|  |
| ------- |
| [1569-number-of-ways-to-reorder-array-to-get-same-bst](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1569-number-of-ways-to-reorder-array-to-get-same-bst) |
## Tree
|  |
| ------- |
| [1569-number-of-ways-to-reorder-array-to-get-same-bst](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1569-number-of-ways-to-reorder-array-to-get-same-bst) |
## Union-Find
|  |
| ------- |
| [1569-number-of-ways-to-reorder-array-to-get-same-bst](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1569-number-of-ways-to-reorder-array-to-get-same-bst) |
## Binary Search Tree
|  |
| ------- |
| [1569-number-of-ways-to-reorder-array-to-get-same-bst](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1569-number-of-ways-to-reorder-array-to-get-same-bst) |
## Memoization
|  |
| ------- |
| [1569-number-of-ways-to-reorder-array-to-get-same-bst](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1569-number-of-ways-to-reorder-array-to-get-same-bst) |
## Combinatorics
|  |
| ------- |
| [1569-number-of-ways-to-reorder-array-to-get-same-bst](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1569-number-of-ways-to-reorder-array-to-get-same-bst) |
## Binary Tree
|  |
| ------- |
| [1569-number-of-ways-to-reorder-array-to-get-same-bst](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1569-number-of-ways-to-reorder-array-to-get-same-bst) |
## Fermat's Little Theorem
|  |
| ------- |
| [1569-number-of-ways-to-reorder-array-to-get-same-bst](https://github.com/Shiva-Kumar-S-M/LeetCode/tree/master/1569-number-of-ways-to-reorder-array-to-get-same-bst) |
<!---LeetCode Topics End-->