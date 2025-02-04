from transformers import AutoTokenizer
import logging
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import torch as T
from dataset import ReasoningHashDataset
from torch.utils.data import DataLoader

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=T.bfloat16
)

# Create a small test dataset
dataset = ReasoningHashDataset(
    tokenizer=tokenizer,
    num_samples=10000,  # Small number for testing
    hash_length=4,  # Shorter hashes for testing
    chains=[2, 3, 4],  # Simpler chain lengths
    vary_hash=True,
    num_chains=3,
    device=model.device
)

# Create reference model (copy of initial weights)
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)


def compute_rewards_batched(output_ids: torch.Tensor, target_ids: torch.Tensor,
                            tokenizer: AutoTokenizer) -> torch.Tensor:
    """
    Compute rewards for circle tag content matching in a batched manner.

    Args:
        output_ids: Tensor of shape [batch_size, seq_len] containing token ids
        target_ids: Tensor of shape [target_len] containing target token ids
        tokenizer: Tokenizer for encoding tags

    Returns:
        Tensor of shape [batch_size] containing reward values
    """
    batch_size, seq_len = output_ids.shape
    device = output_ids.device

    # Encode tags once
    start_tag = torch.tensor(tokenizer.encode("circle>", add_special_tokens=False),
                             device=device)
    end_tag = torch.tensor(tokenizer.encode("</circle", add_special_tokens=False),
                           device=device)

    start_len = start_tag.size(0)
    end_len = end_tag.size(0)
    target_len = target_ids.size(0)

    # Initialize rewards
    rewards = torch.zeros(batch_size, device=device)

    # Create sliding windows for tag detection
    # [batch_size, seq_len - window_size + 1, window_size]
    start_windows = F.unfold(output_ids.unsqueeze(1).float(),
                             (1, start_len)).view(batch_size, -1, start_len)
    end_windows = F.unfold(output_ids.unsqueeze(1).float(),
                           (1, end_len)).view(batch_size, -1, end_len)

    # Find tag positions
    # [batch_size, seq_len - start_len + 1]
    start_matches = torch.all(start_windows == start_tag, dim=-1)
    end_matches = torch.all(end_windows == end_tag, dim=-1)

    # Get first occurrence of tags
    start_pos = torch.argmax(start_matches.float(), dim=-1)
    end_pos = torch.argmax(end_matches.float(), dim=-1)

    # Validate tag positions and extract content
    valid_tags = (start_pos < end_pos) & (start_pos > 0) & (end_pos < seq_len)

    if valid_tags.any():
        # Handle valid tag cases
        valid_batch = torch.where(valid_tags)[0]

        for idx in valid_batch:
            s_pos = start_pos[idx] + start_len
            e_pos = end_pos[idx]

            # Extract content between tags
            content = output_ids[idx, s_pos:e_pos]

            # Check content match
            if content.size(0) == target_len and torch.equal(content, target_ids):
                rewards[idx] = 5.0  # Full reward
            else:
                rewards[idx] = 1.0  # Partial reward

    # Handle start tag only cases
    start_only = start_matches.any(dim=-1) & ~valid_tags
    rewards[start_only] = 0.5

    return rewards


def compute_format_rewards_batched(output_ids: torch.Tensor,
                                   tokenizer: AutoTokenizer) -> torch.Tensor:
    """
    Compute rewards for think tag completion in a batched manner.

    Args:
        output_ids: Tensor of shape [batch_size, seq_len] containing token ids
        tokenizer: Tokenizer for encoding tags

    Returns:
        Tensor of shape [batch_size] containing reward values
    """
    batch_size, seq_len = output_ids.shape
    device = output_ids.device

    # Define closing tags
    closing_tags = ['</think>', ' </think>', '\n</think>']
    tag_tensors = [torch.tensor(tokenizer.encode(tag, add_special_tokens=False),
                                device=device) for tag in closing_tags]

    # Initialize results
    min_positions = torch.full((batch_size,), float('inf'), device=device)

    # Find earliest closing tag for each sequence in batch
    for tag_tensor in tag_tensors:
        tag_len = tag_tensor.size(0)

        # Create sliding windows
        # [batch_size, seq_len - tag_len + 1, tag_len]
        windows = F.unfold(output_ids.unsqueeze(1).float(),
                           (1, tag_len)).view(batch_size, -1, tag_len)

        # Find matches
        # [batch_size, seq_len - tag_len + 1]
        matches = torch.all(windows == tag_tensor, dim=-1)

        # Get positions of first match in each sequence
        positions = torch.where(
            matches.any(dim=-1),
            torch.argmax(matches.float(), dim=-1),
            torch.full((batch_size,), seq_len, device=device)
        )

        # Update minimum positions
        min_positions = torch.minimum(min_positions, positions)

    # Calculate rewards
    no_tag = min_positions == float('inf')
    zero_pos = min_positions == 0
    valid_pos = ~no_tag & ~zero_pos

    # Initialize rewards
    rewards = torch.zeros_like(min_positions, dtype=torch.float)

    # No closing tag
    rewards[no_tag] = 0.0

    # Closing tag at start
    rewards[zero_pos] = 0.5

    # Valid positions
    if valid_pos.any():
        base_reward = 1.0
        content_reward = torch.minimum(0.1 * min_positions[valid_pos],
                                       torch.tensor(1.0, device=device))
        rewards[valid_pos] = base_reward + content_reward

    return rewards


# Flat training loop
num_epochs = 5
group_size = 5  # Number of samples per input
max_completion_length = 200  # Maximum completion length
beta = 0.2  # KL penalty coefficient
log_every = 50  # Log every N batches
ref_model_every = 5  # Update reference model every N epochs
device = model.device
start_time = datetime.now()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(
            f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


# Initialize tracking metrics
running_stats = {
    'losses': [],
    'rewards': [],
    'kl_divs': [],
    'policy_losses': []
}

for epoch in range(num_epochs):
    epoch_stats = {
        'losses': [],
        'rewards': [],
        'kl_divs': [],
        'policy_losses': []
    }

    for batch_idx in range(len(dataloader)):
        optimizer.zero_grad()
        batch = next(iter(dataloader))
        input_ids = batch["input"].input_ids  # Shape: (B, S)
        attention_mask = batch["input"].attention_mask  # Shape: (B, S)
        target = batch["target"]  # Shape: (B,)
        batch_size = input_ids.size(0)

        # Generate multiple outputs for each input in batch
        # This will give us (B*G, S) outputs where G is num_generations
        outputs = model.generate(
            **batch["input"],
            max_new_tokens=max_completion_length,
            do_sample=True,
            num_return_sequences=group_size,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        # Get completion lengths and mask
        prompt_length = input_ids.size(1)
        completion_ids = outputs[:, prompt_length:]  # Shape: (B*G, C)

        # Mask everything after EOS token
        is_eos = completion_ids == tokenizer.eos_token_id  # Shape: (B*G, C)
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(
            1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[
            is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(
            1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(
            1)).int()  # Shape: (B*G, C)

        # Prepare attention mask for full sequence
        prompt_mask_repeated = attention_mask.repeat_interleave(
            group_size, dim=0)  # Shape: (B*G, S)
        attention_mask_full = torch.cat(
            # Shape: (B*G, S+C)
            [prompt_mask_repeated, completion_mask], dim=1)

        # Get per-token log probs for current model
        # Shape: (B*G, S+C-1, V)
        logits = model(input_ids=outputs,
                       attention_mask=attention_mask_full).logits[:, :-1]
        log_probs = logits.log_softmax(dim=-1)
        curr_token_logprobs = torch.gather(
            log_probs,
            dim=-1,
            index=outputs[:, 1:].unsqueeze(-1)
        ).squeeze(-1)  # Shape: (B*G, S+C-1)

        # Get per-token log probs for reference model
        with torch.no_grad():
            ref_logits = ref_model(
                input_ids=outputs, attention_mask=attention_mask_full).logits[:, :-1]
            ref_log_probs = ref_logits.log_softmax(dim=-1)
            ref_token_logprobs = torch.gather(
                ref_log_probs,
                dim=-1,
                index=outputs[:, 1:].unsqueeze(-1)
            ).squeeze(-1)  # Shape: (B*G, S+C-1)

        # Calculate KL divergence per token
        per_token_kl = torch.exp(ref_token_logprobs - curr_token_logprobs) - \
            (ref_token_logprobs - curr_token_logprobs) - 1
        # Shape: (B*G, S+C-1)

        # Compute rewards
        completions = tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True)
        rewards = compute_rewards_batched(
            completion_ids, target, tokenizer) + compute_format_rewards_batched(completion_ids, tokenizer)

        # Compute group-wise statistics for advantages
        # Reshape rewards to (B, G) to compute stats per batch item
        rewards_grouped = rewards.view(batch_size, group_size)
        mean_grouped_rewards = rewards_grouped.mean(dim=1)  # Shape: (B)
        std_grouped_rewards = rewards_grouped.std(dim=1)  # Shape: (B)

        # Expand means and stds to match B*G shape
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            group_size)  # Shape: (B*G)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            group_size)  # Shape: (B*G)

        # Compute advantages
        advantages = (rewards - mean_grouped_rewards) / \
            (std_grouped_rewards + 1e-4)  # Shape: (B*G)

        # Compute final loss
        per_token_loss = torch.exp(
            curr_token_logprobs - curr_token_logprobs.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - beta *
                           per_token_kl)  # Shape: (B*G, S+C-1)

        full_completion_mask = torch.cat([
            torch.zeros_like(prompt_mask_repeated),  # zeros for prompt tokens
            completion_mask  # original mask for completion tokens
        ], dim=1)[:, :-1]  # remove last position to match per_token_loss shape

        # Apply completion mask and average
        masked_loss = (per_token_loss * full_completion_mask)
        loss = (masked_loss.sum(dim=1) /
                full_completion_mask.sum(dim=1)).mean()

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Update statistics per batch
        epoch_stats['losses'].append(loss.item())
        epoch_stats['rewards'].append(rewards.mean().item())
        epoch_stats['kl_divs'].append(per_token_kl.mean().item())
        epoch_stats['policy_losses'].append(mean_grouped_rewards.mean().item())

        if batch_idx % log_every == 0:
            # Log batch-level metrics
            logging.info(
                f"\nBatch Progress:\n"
                f"Epoch: [{epoch+1}/{num_epochs}] "
                f"Batch: [{batch_idx+1}/{len(dataloader)}]\n"
                f"Current Metrics:\n"
                f"├── Loss: {loss.item():.4f}\n"
                f"├── Mean Reward: {rewards.mean().item():.4f}\n"
                f"├── Per-batch Reward Std: {rewards_grouped.std(dim=1).mean().item():.4f}\n"
                f"├── KL Divergence: {per_token_kl.mean().item():.4f}\n"
                f"├── Mean Advantage: {mean_grouped_rewards.mean().item():.4f}\n"
                f"└── Avg Completion Length: {completion_mask.sum(1).float().mean().item():.1f}"
            )
            # Save model at the end of each epoch
            save_dir = f'model_checkpoint_batch_{batch+1}'
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            logging.info(f"Saved model checkpoint to {save_dir}")
