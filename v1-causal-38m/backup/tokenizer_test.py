# dataset/tokenizer_test.py

# Step 1: Create some sample training text for Agastya
training_text = "hello agastya! who created you? i am an open source ai assistant."

# Step 2: Extract every unique character to build our Vocabulary
# Sorting them gives us a consistent numerical order
chars = sorted(list(set(training_text)))
vocab_size = len(chars)

print("--- Agastya's Vocabulary ---")
print(f"Total Unique Characters: {vocab_size}")
print(f"Characters: {chars}\n")

# Step 3: Create look-up dictionaries (The Mapping Rules)
# char_to_int: converts a character to a number ID (e.g., 'a' -> 1)
# int_to_char: converts a number ID back to a character (e.g., 1 -> 'a')
char_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_char = { i:ch for i,ch in enumerate(chars) }

# Step 4: Define encoder and decoder functions
def encode(string):
    # Take text string -> return a list of integer IDs
    return [char_to_int[c] for c in string]

def decode(integer_list):
    # Take a list of integer IDs -> return the text string
    return ''.join([int_to_char[i] for i in integer_list])

# Step 5: Test the Tokenizer live!
sample_phrase = "hello agastya"
encoded_sequence = encode(sample_phrase)
decoded_sequence = decode(encoded_sequence)

print("--- Tokenizer Test Run ---")
print(f"Original Text:  '{sample_phrase}'")
print(f"Encoded IDs:    {encoded_sequence}")
print(f"Decoded Back:   '{decoded_sequence}'")