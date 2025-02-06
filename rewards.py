
def compute_rewards_batched(output_ids: torch.Tensor, target_ids: torch.Tensor,
                          tokenizer: AutoTokenizer) -> torch.Tensor:
    """
    Compute rewards for answer accuracy using simple binary rewards.
    
    Args:
        output_ids: Tensor of shape [batch_size, num_groups, seq_len] containing token ids
        target_ids: Tensor of shape [batch_size, target_len] containing target token ids
        tokenizer: Tokenizer for decoding ids to text
    
    Returns:
        Tensor of shape [batch_size, num_groups] containing reward values
    """
    device = output_ids.device
    batch_size, num_groups, seq_len = output_ids.shape
    rewards = torch.zeros(batch_size, num_groups, device=device)
    
    # Reshape output_ids to process each group
    flat_outputs = output_ids.view(batch_size * num_groups, seq_len)
    
    for b in range(batch_size):
        # Get target for this batch
        target = tokenizer.decode(target_ids[b], skip_special_tokens=True)
        
        for g in range(num_groups):
            # Calculate flat index
            idx = b * num_groups + g
            output = tokenizer.decode(flat_outputs[idx], skip_special_tokens=True)
            
            # Find content between circle tags
            start_tag = "circle>"
            end_tag = "</circle"
            end_think = "</think>"
            output = output[output.find(end_think):]
            
            start_pos = output.find(start_tag)
            if start_pos != -1:
                start_pos += len(start_tag)
                end_pos = output.find(end_tag, start_pos)
                
                if end_pos != -1:
                    content = output[start_pos:end_pos].strip()
                    
                    # Binary reward: 1.0 for exact match, 0.0 otherwise
                    if content == target:
                        rewards[b, g] = 5.0
                    # Check for character overlap
                    elif len(content) > 0 and len(target) == len(content):
                        common_chars = sum(1 for c in content if c in target)
                        rewards[b, g] = common_chars / len(target)
    
    return rewards

def compute_format_rewards_batched(output_ids: torch.Tensor,
                                 tokenizer: AutoTokenizer) -> torch.Tensor:
    """
    Compute rewards for think tag completion using simple format checks.
    Required format: <think>content</think><circle>answer</circle>
    Circle tags must come after think tags, not inside them.

    Args:
        output_ids: Tensor of shape [batch_size, num_groups, seq_len] containing token ids
        tokenizer: Tokenizer for decoding tokens to text

    Returns:
        Tensor of shape [batch_size, num_groups] containing reward values
    """
    batch_size, num_groups, seq_len = output_ids.shape
    device = output_ids.device

    # Define tags
    think_tags = ['</think>', ' </think>', '\n</think>']
    think_start = '<think>'
    circle_start = '<circle>'
    circle_end = '</circle>'
    
    # Initialize rewards
    rewards = torch.zeros((batch_size, num_groups), dtype=torch.float, device=device)
    
    # Reshape output_ids to process each group
    flat_outputs = output_ids.view(batch_size * num_groups, seq_len)
    
    for b in range(batch_size):
        for g in range(num_groups):
            # Calculate flat index
            idx = b * num_groups + g
            sequence = tokenizer.decode(flat_outputs[idx], skip_special_tokens=True)
            
            # Check for presence of all required tags
            think_start_pos = sequence.find(think_start)
            think_end_pos = min((sequence.find(tag) for tag in think_tags if tag in sequence), default=-1)
            circle_start_pos = sequence.find(circle_start)
            circle_end_pos = sequence.find(circle_end)
            
            # Only give reward if all format requirements are met:
            if (think_start_pos != -1 and think_end_pos != -1 and
                circle_start_pos != -1 and circle_end_pos != -1):
                
                # Check that only one of each tag exists
                if (sequence.count(think_start) == 1 and 
                    sum(sequence.count(tag) for tag in think_tags) == 1 and
                    sequence.count(circle_start) == 1 and 
                    sequence.count(circle_end) == 1):
                    
                    # Check proper tag ordering:
                    # 1. Nothing before think start
                    # 2. Think tags must be paired
                    # 3. Circle tags must be paired
                    # 4. Circle tags must come after think tags
                    # 5. Nothing after circle end
                    before_think = sequence[:think_start_pos].strip()
                    after_circle = sequence[circle_end_pos + len(circle_end):].strip()
                    
                    if (not before_think and 
                        not after_circle and
                        think_start_pos < think_end_pos and
                        circle_start_pos < circle_end_pos and
                        circle_start_pos > think_end_pos):  # Circle must start after think ends
                        rewards[b, g] = 1.0
    
    return rewards
