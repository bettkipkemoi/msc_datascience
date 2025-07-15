#Pattern Matching Algorithms
def naive_string_match(text, pattern):
    """
    Naive string matching algorithm.
    Returns list of starting indices where pattern matches in text.
    """
    matches = []
    n, m = len(text), len(pattern)
    
    for i in range(n - m + 1):
        j = 0
        while j < m and text[i + j] == pattern[j]:
            j += 1
        if j == m:
            matches.append(i)
    
    return matches

def kmp_string_match(text, pattern):
    """
    Knuth-Morris-Pratt (KMP) string matching algorithm.
    Returns list of starting indices where pattern matches in text.
    """
    def compute_lps(pattern):
        """Compute the Longest Proper Prefix which is also Suffix array."""
        m = len(pattern)
        lps = [0] * m
        length = 0
        i = 1
        
        while i < m:
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
    
    matches = []
    n, m = len(text), len(pattern)
    lps = compute_lps(pattern)
    i = j = 0
    
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1
        if j == m:
            matches.append(i - j)
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return matches

def boyer_moore_string_match(text, pattern):
    """
    Boyer-Moore string matching algorithm using bad character heuristic.
    Returns list of starting indices where pattern matches in text.
    """
    def bad_character_heuristic(pattern):
        """Create bad character table."""
        bad_char = {}
        for i in range(len(pattern)):
            bad_char[pattern[i]] = i
        return bad_char
    
    matches = []
    n, m = len(text), len(pattern)
    bad_char = bad_character_heuristic(pattern)
    
    s = 0
    while s <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[s + j]:
            j -= 1
        if j < 0:
            matches.append(s)
            s += 1
        else:
            s += max(1, j - bad_char.get(text[s + j], -1))
    
    return matches

def main():
    # Test inputs
    text = "ABAAABABAA"
    pattern = "ABA"
    
    print(f"Text: {text}")
    print(f"Pattern: {pattern}")
    
    # Test Naive algorithm
    naive_matches = naive_string_match(text, pattern)
    print(f"Naive Algorithm Matches: {naive_matches}")
    
    # Test KMP algorithm
    kmp_matches = kmp_string_match(text, pattern)
    print(f"KMP Algorithm Matches: {kmp_matches}")
    
    # Test Boyer-Moore algorithm
    bm_matches = boyer_moore_string_match(text, pattern)
    print(f"Boyer-Moore Algorithm Matches: {bm_matches}")

if __name__ == "__main__":
    main()