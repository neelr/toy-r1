import random
from typing import List, Tuple
from torch.utils.data import Dataset
from torch.utils.data import Dataset
import random
from typing import List, Tuple


class ReasoningHashDataset(Dataset):
    def __init__(self,  tokenizer, num_samples: int = 1000, hash_length: int = 5,
                 chains: List[int] = [2, 3, 4, 5, 6], vary_hash: bool = True, num_chains: int = 5, device="cpu"):
        self.device = device
        self.tokenizer = tokenizer
        self.data = []
        self.hash_length = hash_length
        self.max_length = sum(chains) * (hash_length + 3) + 100  # 100 is a buffer
        for _ in range(num_samples):
            if vary_hash:
                # Step 1: Sample 1 chain between the smallest and second biggest
                smallest_chain = min(chains)
                second_biggest_chain = sorted(chains)[-2]
                first_chain = random.choice(
                    [c for c in chains if smallest_chain <= c <= second_biggest_chain])

                # Step 2: Add random chains bigger than the selected chain
                larger_chains = [c for c in chains if c > first_chain]
                if num_chains - 1 <= len(larger_chains):
                    selected_chains = [first_chain] + \
                        random.sample(larger_chains, num_chains - 1)
                else:
                    selected_chains = [first_chain] + \
                        random.choices(larger_chains, k=num_chains - 1)
            else:
                # Take the first num_chains elements
                selected_chains = chains[:num_chains]

            selected_chains.sort()  # Sort to ensure the shortest is first

            hash_list, start, shortest_target = self.generate_hash_list(
                hash_length=hash_length,
                chains=selected_chains
            )
            prompt = f"Map:\n"
            random.shuffle(hash_list)
            for key, value in hash_list:
                prompt += f"{key}=>{value}\n"
            prompt += f"""Start: {
                start}\nTask: Find the shortest end {hash_length} character hash in the chains. Think hard in tag! Circle your answer in <circle>HERE</circle> after </think>\n\n"""
            full_text = f"{prompt} {shortest_target}"
            self.data.append(
                (full_text, hash_list, start, shortest_target, prompt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_text, _, _, target, prompt = self.data[idx]
        prompt += "<think>"

        # Tokenize with left padding and truncation
        encoded_input = self.tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            padding_side='left',  # Ensure left padding
            return_tensors="pt"
        ).to(self.device)

        # Tokenize target with padding
        encoded_target = self.tokenizer(
            target,
            padding='max_length',
            truncation=True,
            max_length=self.hash_length + 2,  # Add small buffer for hash length
            padding_side='left',  # Ensure left padding
            return_tensors="pt"
        ).to(self.device)

        encoded_input["input_ids"] = encoded_input["input_ids"].squeeze(0)
        encoded_input["attention_mask"] = encoded_input["attention_mask"].squeeze(0)
        encoded_target["input_ids"] = encoded_target["input_ids"].squeeze(0)

        return {
            "input": encoded_input,
            "text": prompt,
            "target": encoded_target["input_ids"]
        }

    def get_eval_item(self, idx):
        return self.data[idx]

    @ staticmethod
    def generate_hash_list(hash_length: int, chains: List[int]) -> Tuple[List[Tuple[str, str]], str, str]:
        chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        hash_list = []

        # Generate start hash
        start = ''.join(random.choice(chars) for _ in range(hash_length))

        shortest_length = min(chains)
        shortest_target = None

        # Generate chains
        for chain_length in chains:
            current = start
            for _ in range(chain_length - 1):  # -1 because start is already counted
                next_hash = ''.join(random.choice(chars)
                                    for _ in range(hash_length))
                hash_list.append((current, next_hash))
                current = next_hash

            if chain_length == shortest_length:
                shortest_target = current

        return hash_list, start, shortest_target

    @ staticmethod
    def find_shortest_path(hash_list: List[Tuple[str, str]], start: str) -> int:
        graph = {}
        for src, dest in hash_list:
            if src not in graph:
                graph[src] = []
            graph[src].append(dest)

        visited = set()
        queue = [(start, 0)]

        while queue:
            current, length = queue.pop(0)
            if current not in graph:
                return length  # Return the length of the path

            if current not in visited:
                visited.add(current)
                for next_hash in graph[current]:
                    queue.append((next_hash, length + 1))

        return -1  # This should never happen if the hash list is correctly generated
