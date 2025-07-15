def is_anagram(s1, s2):
    """
    Determine if two strings are anagrams, ignoring case and non-alphabetic characters.
    
    Args:
        s1 (str): First string
        s2 (str): Second string
    
    Returns:
        bool: True if strings are anagrams, False otherwise
    """
    # Handle empty strings
    if not s1 and not s2:
        return True
    
    # Convert to lowercase and keep only alphabetic characters
    s1 = ''.join(c.lower() for c in s1 if c.isalpha())
    s2 = ''.join(c.lower() for c in s2 if c.isalpha())
    
    # Check lengths after preprocessing
    if len(s1) != len(s2):
        return False
    
    # Create frequency dictionary for s1
    freq_s1 = {}
    for c in s1:
        freq_s1[c] = freq_s1.get(c, 0) + 1
    
    # Subtract frequencies for s2
    for c in s2:
        if c not in freq_s1:
            return False
        freq_s1[c] -= 1
        if freq_s1[c] == 0:
            del freq_s1[c]
    
    # If dictionary is empty, strings are anagrams
    return len(freq_s1) == 0

# Test cases
test_cases = [
    ("LISTEN", "SILENT"),
    ("NOW", "WON"),
    ("SELL", "LESS"),
    ("Hello, World!", "World, Hello!"),
    ("", ""),
    ("abc", "abcd"),
    ("Tea", "Eat")
]

for s1, s2 in test_cases:
    result = is_anagram(s1, s2)
    print(f"s1='{s1}', s2='{s2}' -> Anagrams: {result}")