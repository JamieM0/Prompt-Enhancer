from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import nltk

# Download required NLTK data
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

def get_wordnet_pos(treebank_tag):
    """
    Convert Penn Treebank POS tags to WordNet POS tags

    Args:
        treebank_tag (str): POS tag from Penn Treebank
    Returns:
        wordnet POS tag or None if no match found
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def get_enhanced_word(word, pos):
    """
    Find the most specific synset for a word based on its part of speech

    Args:
        word (str): The word to enhance
        pos (str): WordNet POS tag
    Returns:
        enhanced version of the word with its definition
    """
    synsets = wordnet.synsets(word, pos=pos)
    if not synsets:
        return word

    # Get the first synset (most common usage)
    synset = synsets[0]

    # Create an enhanced version with the specific term and its definition
    enhanced = f"{word} (specifically: {synset.name().split('.')[0]}, meaning: {synset.definition()})"
    return enhanced

def enhance_prompt(prompt):
    """
    Enhance a prompt by adding specific WordNet definitions for important words

    Args:
        prompt (str): Original prompt text
    Returns:
        enhanced prompt with specific word meanings and definitions
    """
    # Tokenize and POS tag the input prompt
    tokens = word_tokenize(prompt)
    tagged = pos_tag(tokens)

    # Process each word
    enhanced_words = []
    for word, tag in tagged:
        # Convert POS tag to WordNet format
        wordnet_pos = get_wordnet_pos(tag)

        # Only enhance content words (nouns, verbs, adjectives, adverbs)
        if wordnet_pos:
            enhanced_word = get_enhanced_word(word, wordnet_pos)
            enhanced_words.append(enhanced_word)
        else:
            enhanced_words.append(word)

    # Reconstruct the prompt
    enhanced_prompt = ' '.join(enhanced_words)
    return enhanced_prompt

# Example usage
if __name__ == "__main__":
    sample_prompt = "Imagine a world where cars can drive themselves. Describe the benefits of self-driving cars."
    enhanced = enhance_prompt(sample_prompt)
    print("Original prompt:", sample_prompt)
    print("\nEnhanced prompt:", enhanced)