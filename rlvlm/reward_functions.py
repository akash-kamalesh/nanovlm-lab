# Reward functions
import re
from typing import Optional, List
def format_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Reward function that checks if the completion has the correct format.
    
    Expects format: <think>...</think> followed by <answer>...</answer>
    Flexible with whitespace and newlines.
    
    Args:
        completions: List of completion strings
        **kwargs: Additional arguments (ignored)
    
    Returns:
        List of rewards (1.0 for correct format, 0.0 otherwise)
    """
    # Pattern allows flexible whitespace between tags
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    rewards = []
    
    for content in completions:
        if re.search(pattern, content, re.DOTALL):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    return rewards


def accuracy_reward(
    completions: List[str],
    answer: Optional[str] = None,
    partial_credit: bool = True,
    partial_credit_threshold: float = 0.5,
    **kwargs
) -> List[float]:
    """
    Reward function for answer accuracy using regex matching.
    
    Scoring:
    - 1.0: Answer extracted from <answer> tags matches expected answer exactly
    - 0.5: Expected answer is found within the extracted answer (partial credit)
    - 0.0: No match or answer not found
    
    Args:
        completions: List of completion strings
        answer: Expected answer (from dataset) - can be just the letter (e.g., "C") or full text
        partial_credit: Whether to give partial credit if answer is found within response (default: True)
        partial_credit_threshold: Reward value for partial matches (default: 0.5)
        **kwargs: Additional arguments (ignored)
    
    Returns:
        List of rewards (0.0, 0.5, or 1.0)
    """
    if completions is None or answer is None:
        return [0.0] * len(completions) if completions else []
    
    rewards = []
    answer_str = str(answer).strip().lower()
    
    # Extract just the letter/answer from the expected answer
    # Handle cases like "The answer is C" or "Answer: C" or full solution text ending with answer
    # Look for the pattern "answer:" or "answer is" followed by a letter
    answer_match_expected = re.search(r"(?:answer\s*(?:is|:)?\s*)([a-d])", answer_str)
    if answer_match_expected:
        answer_clean = answer_match_expected.group(1).lower()
    else:
        # If no "answer:" pattern found, look for the last letter [a-d] in the string
        letters = re.findall(r"[a-d]", answer_str)
        if letters:
            answer_clean = letters[-1].lower()  # Take the last occurrence
        else:
            answer_clean = answer_str
    
    for completion in completions:
        reward = 0.0
        
        # Extract answer from <answer>...</answer> tags
        answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        
        if answer_match:
            extracted_answer = answer_match.group(1).strip().lower()
            
            # Check for exact match (single letter or full answer)
            if extracted_answer == answer_clean:
                reward = 1.0
            # Check for partial match (answer found within extracted answer)
            elif partial_credit and answer_clean in extracted_answer:
                reward = partial_credit_threshold
        
        rewards.append(reward)
    
    return rewards