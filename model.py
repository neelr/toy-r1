from transformers import AutoTokenizer
import logging
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
import torch.nn.functional as F
import torch as T
from dataset import ReasoningHashDataset
from torch.utils.data import DataLoader
import os

model_path = "model_checkpoint_sft2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=T.bfloat16
)

# save directory 
main_dir = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(main_dir, exist_ok=True)

hash_length = 4
# Create a small test dataset
dataset = ReasoningHashDataset(
    tokenizer=tokenizer,
    num_samples=10000,  # Small number for testing
    hash_length=hash_length,# Shorter hashes for testing
    chains=[2, 3, 4, 5, 6],  # Simpler chain lengths
    vary_hash=True,
    num_chains=3,
    device=model.device
)

# Create reference model (copy of initial weights)
ref_model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

stop_tokens = [
            tokenizer.encode("</circle", add_special_tokens=False),
            tokenizer.encode(" </circle", add_special_tokens=False)
]

# Custom stopping criteria class
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        for stop_ids in self.stop_token_ids:
            # Safety checks
            if len(input_ids[0]) < len(stop_ids):
                continue
                
            # Get the last n tokens where n is length of stop_ids
            last_tokens = input_ids[0][-len(stop_ids):].tolist()
            
            if last_tokens == stop_ids:
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_tokens)])

def compute_rewards_batched(output_ids: torch.Tensor, target_ids: torch.Tensor,
                            tokenizer: AutoTokenizer) -> torch.Tensor:
    """
    Compute rewards for circle tag content matching by converting to string space.
    
    Args:
        output_ids: Tensor of shape [batch_size, seq_len] containing token ids  
        target_ids: Tensor of shape [batch_size, target_len] containing target token ids
        tokenizer: Tokenizer for decoding ids to text
    
    Returns:
        Tensor of shape [batch_size] containing reward values
    """
    device = output_ids.device
    batch_size = output_ids.shape[0]
    rewards = torch.zeros(batch_size, device=device)
    
    for i in range(batch_size):
        output = tokenizer.decode(output_ids[i], skip_special_tokens=True)
        target = tokenizer.decode(target_ids[i//output_ids.size(0)], skip_special_tokens=True)
        
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
                if content == target:
                    rewards[i] = 20.0  # Full reward
                elif len(content) > 0:
                    if content in target:
                        rewards[i] += 2.0
                    if len(content) == len(target):
                        rewards[i] += 3.0
                        # check number of characters in common
                        common = 0
                        for c in content:
                            if c in target:
                                common += 1
                        rewards[i] += common
                    else:
                        rewards[i] = 1 # Partial reward
    
    return rewards


def compute_format_rewards_batched(output_ids: torch.Tensor,
                                   tokenizer: AutoTokenizer) -> torch.Tensor:
    """
    Compute rewards for think tag completion in a batched manner, operating in string space.

    Args:
        output_ids: Tensor of shape [batch_size, seq_len] containing token ids
        tokenizer: Tokenizer for decoding tokens to text

    Returns:
        Tensor of shape [batch_size] containing reward values
    """
    batch_size = output_ids.shape[0]
    device = output_ids.device

    # Define closing tags
    closing_tags = ['</think>', ' </think>', '\n</think>']
    
    # Initialize results tensor
    rewards = torch.zeros(batch_size, dtype=torch.float, device=device)
    
    # Decode each sequence in the batch
    decoded_sequences = []
    for i in range(batch_size):
        # Skip special tokens to get clean text
        decoded = tokenizer.decode(output_ids[i], skip_special_tokens=True)
        decoded_sequences.append(decoded)
    
    # Process each sequence
    for idx, sequence in enumerate(decoded_sequences):
        # Find earliest closing tag
        positions = []
        for tag in closing_tags:
            pos = sequence.find(tag)
            if pos != -1:  # Found tag
                positions.append(pos)
        
        # Get earliest position if any tags found
        min_pos = min(positions) if positions else float('inf')
        
        # Calculate rewards based on position
        if min_pos == float('inf'):
            # No closing tag found
            rewards[idx] = 0.0
        elif min_pos == 0:
            # Tag at start
            rewards[idx] = 0.5
        else:
            # Valid position - base reward plus content reward
            # Scale content reward based on character length instead of tokens
            base_reward = 1.0
            # Cap content reward at 1.0, scale based on characters
            content_reward = max(min(0.01 * min_pos - 0.01 * (len(sequence) - min_pos), 3.0),0 )
            rewards[idx] = base_reward + content_reward
            
        # Add penalty for multiple closing tags
        if sum(1 for tag in closing_tags if tag in sequence) > 1:
            rewards[idx] *= 0.8  # 20% penalty for multiple closing tags

        # check if there is a </circle> tag before the </think> tag
        if sequence.find("</circle>") < min_pos:
            rewards[idx] -= 0.5
            rewards[idx] = max(rewards[idx], 0.0)

    return rewards


# Flat training loop
num_epochs = 1
group_size = 8  # Number of samples per input
max_completion_length = 400  # Maximum completion length
beta = 0.1  # KL penalty coefficient
log_every = 25  # Log every N batches
ref_model_every = 50  # Update reference model every N epochs
batch_size = 3  # Batch size
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

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


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
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_completion_length,
            do_sample=True,
            num_return_sequences=group_size,
            temperature=0.6,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            #stopping_criteria=stopping_criteria
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
            # Get a sample input and output to log
            sample_idx = 0  # Take first example from batch
            input_text = tokenizer.decode(input_ids[sample_idx], skip_special_tokens=True)
            output_text = tokenizer.decode(outputs[sample_idx], skip_special_tokens=True)
            target_text = tokenizer.decode(target[sample_idx], skip_special_tokens=True)
            
            # Log batch-level metrics and sample text
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
                f"├── Avg Completion Length: {completion_mask.sum(1).float().mean().item():.1f}\n"
                f"Sample Input-Output:\n"
                f"└── Output: {output_text}"
                f"└── Target: {target_text}"
            )

        # Update reference model
        if batch_idx % ref_model_every == 0:
            ref_model.load_state_dict(model.state_dict())
            ref_model.eval()
            logging.info("Updated reference model weights")

            # Save model at logging interval
            save_dir = f"{main_dir}/model_checkpoint_batch_{batch_idx}"
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            logging.info(f"Saved model checkpoint to {save_dir}")
