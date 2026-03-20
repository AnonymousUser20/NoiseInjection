import requests
import json
import pandas as pd
import time
import os
from tqdm import tqdm
from typing import Dict, List
from pathlib import Path
import pickle
import re
import random
from collections import defaultdict
from math import sqrt, log
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class LLMResponseGenerator:
    def __init__(self, api_key: str, model: str = "microsoft/phi-4"):

        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/dataset-eval",
            "X-Title": "Mathematical Dataset Response Generation",
            "Connection": "keep-alive",
        }
        # --- Model-specific safety defaults ---
        # For o3-mini, we want more output tokens and to exclude reasoning
        self.max_tokens_main = 8192  # for query_llm
        self.max_tokens_sample = 4096  # for _query_llm_sample

        # Only apply reasoning controls for o3-family models
        if self.model.startswith("openai/o3"):
            self.reasoning_config = {
                "effort": "low",  # or "medium" if you prefer
                "exclude": True  # do NOT send reasoning trace in output
            }
        else:
            self.reasoning_config = None

        # ------------------------------
        # 🔥 IMPORTANT: USE A SINGLE SESSION WITH RETRIES
        # ------------------------------
        self.session = requests.Session()

        # Retry also on POST (allowed_methods=False) with backoff
        retries = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=1.0,  # 1s, 2s, 4s, 8s, ...
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=False,  # retry POST as well
            raise_on_status=False,
        )

        # Connection pool sizes don't need to be huge unless you use many threads
        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=retries,
        )

        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # File that stores partial results so you can resume interrupted runs
        self.progress_file = "response_generation_progress.pkl"

    # -----------------------------------------------------------------------
    # Nested helper for Monte-Carlo Tree Search
    class _MCTSNode:
        """
        Lightweight container for a partial reasoning state during MCTS.
        Stored *inside* LLMResponseGenerator so we can reference it with
        `self._MCTSNode(...)`.
        """

        __slots__ = ("state", "parent", "children", "visits", "value")

        def __init__(self, state: str, parent: "LLMResponseGenerator._MCTSNode" = None):
            self.state: str = state
            self.parent = parent
            self.children: list["LLMResponseGenerator._MCTSNode"] = []
            self.visits: int = 0
            self.value: float = 0.0  # cumulative reward

        # ------------ convenience helpers ----------------------------------
        def ucb_score(self, c: float = 1.4) -> float:
            """
            Upper-Confidence Bound score for tree policy selection.
            """
            if self.parent is None:
                return float("inf")  # root always selected first
            exploitation = self.value / (self.visits + 1e-6)
            exploration = c * sqrt(log(self.parent.visits + 1) / (self.visits + 1e-6))
            return exploitation + exploration

        def add_child(self, child_state: str) -> "LLMResponseGenerator._MCTSNode":
            """
            Append a new child with the given partial state and return it.
            """
            child = LLMResponseGenerator._MCTSNode(state=child_state, parent=self)
            self.children.append(child)
            return child

        # Optional: nicer printout when debugging
        def __repr__(self):
            return (f"_MCTSNode(state_len={len(self.state)}, "
                    f"visits={self.visits}, value={self.value:.2f}, "
                    f"children={len(self.children)})")

    # -----------------------------------------------------------------------
    # Monte-Carlo Tree Search over reasoning trajectories
    def query_llm_mcts(
            self,
            question: str,
            simulations: int = 40,
            max_depth: int = 4,
            c_ucb: float = 1.4,
    ) -> str:
        """
        Perform a small-budget MCTS to explore alternative reasoning paths.

        Returns the highest-reward completion (JSON string) produced during search.
        Reward = 1 if the JSON parses + passes a light numeric/arith check, else 0.
        """

        base_prompt = self.create_prompt(question, method="cot")
        root = self._MCTSNode(state="")  # start with empty partial reasoning

        # ---------- helpers -------------------------------------------------
        def expand(node: "LLMResponseGenerator._MCTSNode"):
            """Generate one more reasoning chunk and attach as a child."""
            expansion_prompt = base_prompt.replace(
                "\"Let's solve it step by step...\"",
                f"\"Current reasoning: {node.state}\nContinue reasoning:\""
            )
            chunk = self._query_llm_sample(
                expansion_prompt, temperature=0.7, top_p=0.95, max_tokens=50
            )
            return node.add_child(chunk.strip())

        def rollout(node: "LLMResponseGenerator._MCTSNode"):
            """Complete the reasoning to an answer from the given partial state."""
            completion_prompt = base_prompt.replace(
                "\"Solve it step by step...\"",
                f"\"Current reasoning: {node.state}\nFinish reasoning and give answer:\""
            )
            full_resp = self._query_llm_sample(
                completion_prompt, temperature=0.7, top_p=0.9
            )
            try:
                json.loads(full_resp)  # parses?
                reward = 1.0  # simple binary reward
            except Exception:
                reward = 0.0
            return full_resp, reward

        best_resp, best_reward = None, -1.0

        # ---------- main loop ----------------------------------------------
        for _ in range(simulations):

            # Selection
            node, depth = root, 0
            while node.children and depth < max_depth:
                node = max(node.children, key=lambda n: n.ucb_score(c_ucb))
                depth += 1

            # Expansion (if depth limit not reached)
            if depth < max_depth:
                node = expand(node)

            # Roll-out
            response, reward = rollout(node)

            # Back-propagate
            while node:
                node.visits += 1
                node.value += reward
                node = node.parent

            if reward > best_reward:
                best_resp, best_reward = response, reward

        return best_resp

    def load_csv_dataset(self, file_path: str) -> List[Dict]:
        """Load a CSV dataset file"""
        data = []
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Loaded {len(df)} samples from {file_path}")

            # For noisy datasets, we have: original, low_noise, medium_noise, high_noise
            for _, row in df.iterrows():
                data.append({
                    'original': str(row.get('original', '')),
                    'low_noise': str(row.get('low_noise', '')),
                    'medium_noise': str(row.get('medium_noise', '')),
                    'high_noise': str(row.get('high_noise', '')),
                    'row_data': row.to_dict()
                })

        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
        except Exception as e:
            print(f"❌ Error loading {file_path}: {str(e)}")
        return data

    def load_all_datasets(self, base_path: str = "datasets_300_noisy/") -> Dict[str, List[Dict]]:
        """Load all specified datasets"""
        datasets = {}

        # Define target datasets and their expected CSV files
        target_datasets = {
            # "gsm8k_original": ["train.csv", "test.csv"],
            # "gsm8k_original": ["test.csv"],
            "gsm_symbolic": ["symbolic.csv"],
             "omni_math": ["test.csv"],
             "math_500": ["test.csv"],

        }

        print("Loading Mathematical Datasets...")
        print("=" * 50)

        for dataset_name, expected_files in target_datasets.items():
            dataset_path = Path(base_path) / dataset_name

            if not dataset_path.exists():
                print(f"⚠️  Dataset directory not found: {dataset_path}")
                continue

            # Load all CSV files in the dataset directory
            csv_files = list(dataset_path.glob("*.csv"))

            if not csv_files:
                print(f"⚠️  No CSV files found in {dataset_path}")
                continue

            dataset_data = []
            for csv_file in csv_files:
                file_data = self.load_csv_dataset(str(csv_file))
                dataset_data.extend(file_data)
                print(f"  - {csv_file.name}: {len(file_data)} samples")

            if dataset_data:
                datasets[dataset_name] = dataset_data
                print(f"  Total for {dataset_name}: {len(dataset_data)} samples")

        print(f"\nTotal datasets loaded: {len(datasets)}")
        return datasets

    def create_prompt(self, question: str, method: str) -> str:
        if method == "normal":
            return f"""You are a math expert. Solve the following math problem and give the final answer only.
                    Respond in JSON format with the final answer wrapped in <answer></answer> tags.

                    Question:
                    {question}

                    Answer:
                    {{
                      "final_answer": "<answer>YOUR_FINAL_ANSWER_HERE</answer>"
                    }}"""

        elif method == "cot":
            return f"""You are a math expert. Solve the following math problem step-by-step.

                    Respond in JSON format with:
                    - "reasoning": your step-by-step explanation.
                    - "final_answer": the final answer wrapped in <answer></answer> tags.

                    Question:
                    {question}

                    Answer:
                    {{
                      "reasoning": "Let's solve it step by step...",
                      "final_answer": "<answer>YOUR_FINAL_ANSWER_HERE</answer>"
                    }}"""

        elif method == "denoise_cot":
            return f"""You are a math expert. The following math question may contain irrelevant or noisy information.

                    Your task:
                    1. Identify irrelevant/noisy sentences (that are not needed to solve the math problem).
                    2. Provide a cleaned version of the question without the noise.
                    3. Solve the cleaned question step by step.

                    Respond in JSON format using the following keys:
                    - "noisy_sentences": Wrap the list in <noisy_sentences></noisy_sentences> tags.
                    - "denoised_question": Wrap the cleaned question in <denoised_question></denoised_question> tags.
                    - "solution_steps": The step-by-step reasoning.
                    - "final_answer": Wrap the final answer in <answer></answer> tags.

                    Now analyze and solve:
                    {question}

                    Answer:
                    {{
                      "noisy_sentences": "<noisy_sentences>LIST_NOISY_SENTENCES</noisy_sentences>",
                      "denoised_question": "<denoised_question>CLEANED_QUESTION_HERE</denoised_question>",
                      "solution_steps": "Step-by-step reasoning here...",
                      "final_answer": "<answer>YOUR_FINAL_ANSWER_HERE</answer>"
                    }}"""

        elif method == "macro_action":
            return f"""You are a math expert. The question below may contain irrelevant or noisy information.

                    Your task:
                    - Assume: Identify quantities and facts that are likely relevant for solving the problem.
                    - Simplify: Rewrite the problem by removing irrelevant details.
                    - Verify: Check that the simplified problem still preserves the original question's intent.
                    - Solve: Answer the problem step-by-step.

                    Follow these four stages exactly and clearly label your output at each stage.

                    Respond in JSON format using the following keys:
                    - "assumptions": Relevant quantities and facts.
                    - "simplified_question": The rewritten problem statement.
                    - "verification": Verification notes.
                    - "solution_steps": The detailed step-by-step reasoning.
                    - "final_answer": Wrap the final answer in <answer></answer> tags.

                    Question:
                    {question}

                    Answer:
                    {{
                      "assumptions": "Relevant facts here...",
                      "simplified_question": "Simplified question here...",
                      "verification": "Verification notes here...",
                      "solution_steps": "Step-by-step reasoning here...",
                      "final_answer": "<answer>YOUR_FINAL_ANSWER_HERE</answer>"
                    }}"""

        else:
            raise ValueError(f"Unsupported method: {method}")

    # ---------- 1.  PRM utilities ---------------------------------------------
    def _extract_numbers(self, text):
        """Return all numbers (integers / floats) appearing in a string."""
        # FIX: Ensure text is a string
        if isinstance(text, list):
            text = ' '.join(str(item) for item in text)
        elif not isinstance(text, str):
            text = str(text)
        return re.findall(r"-?\d+(?:\.\d+)?", text)

    def _arith_consistency_score(self, response_json: dict) -> float:
        """
        Very light-weight arithmetic check:
        – counts "=" symbols (proxy for shown work)
        – penalises obvious division-by-zero
        """
        steps = response_json.get("solution_steps", "")
        # FIX: Ensure steps is a string
        if isinstance(steps, list):
            steps = ' '.join(str(item) for item in steps)
        elif not isinstance(steps, str):
            steps = str(steps)

        if not steps:
            return 0.0
        eq = steps.count("=")
        bad = steps.lower().count("/0")
        return max(eq - 3 * bad, 0)

    def _prm_score(self, response: str, cleaned_q: str) -> float:
        """Heuristic score ⇒ higher = better."""
        try:
            resp_json = json.loads(response)
        except Exception:
            return 0.0

        nums_in_q = set(self._extract_numbers(cleaned_q))

        solution_steps = resp_json.get("solution_steps", "")
        nums_in_ans = set(self._extract_numbers(solution_steps))

        num_check = 1.0 if nums_in_ans <= nums_in_q else 0.0
        arith_check = min(self._arith_consistency_score(resp_json) / 10.0, 1.0)
        jitter = random.random() * 1e-3
        return num_check + arith_check + jitter

    def _build_payload(self, prompt: str,
                       temperature: float,
                       top_p: float,
                       max_tokens: int) -> dict:
        """Build a chat.completions payload, with o3-mini safeguards."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        # Add reasoning controls for o3-mini/o3 if configured
        if self.reasoning_config is not None:
            payload["reasoning"] = self.reasoning_config

        return payload

    def _query_llm_sample(
            self,
            prompt: str,
            top_p: float = 0.9,
            temperature: float = 0.7,
            max_tokens: int = None,
            max_retries: int = 4,
            base_delay: float = 1.0,
    ) -> str:
        """Single call with stochastic decoding (used by PRM + MCTS playouts)."""
        time.sleep(0.15)

        if max_tokens is None:
            max_tokens = self.max_tokens_sample

        payload = self._build_payload(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    url=self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=(10, 60),
                )

                if response.status_code == 200:
                    res = response.json()
                    choices = res.get("choices", [])
                    if not choices:
                        return f"Error: No choices in response - {res}"

                    choice = choices[0]
                    msg = choice.get("message", {})
                    content = (msg.get("content") or "").strip()
                    finish_reason = choice.get("finish_reason")

                    # If we got some content, use it, even if truncated
                    if content:
                        if finish_reason == "length":
                            print("[sample] Warning: output truncated by max_tokens.")
                        return content

                    # No content at all; if truncated, this is bad => treat as error
                    if finish_reason == "length":
                        return f"Error: Empty content and truncated by length - {res}"

                    return f"Error: Empty content from model - {res}"

                if response.status_code in (429, 500, 502, 503, 504):
                    wait = base_delay * (2 ** attempt)
                    print(f"[sample] HTTP {response.status_code}, retrying in {wait:.1f}s...")
                    time.sleep(wait)
                    continue

                return f"Error: HTTP {response.status_code} - {response.text}"

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt == max_retries - 1:
                    return f"Error: Request failed after retries - {e}"
                wait = base_delay * (2 ** attempt)
                print(f"[sample] Connection error: {e} – retrying in {wait:.1f}s...")
                time.sleep(wait)

            except requests.exceptions.RequestException as e:
                return f"Error: Request failed - {e}"

        return "Error: Max retries exceeded in _query_llm_sample"

    # ---------------------------------------------------------------------------
    # === 3.  Process-Reward-Modeling (PRM) =====================================
    def query_llm_prm(self, question: str, n_candidates: int = 6) -> str:
        """
        1. Uses the denoise-CoT template to generate multiple candidates.
        2. Scores each candidate with heuristic PRM filters.
        3. Returns the highest-scoring answer.
        """
        prm_prompt = self.create_prompt(question, method="denoise_cot")

        # First pass: obtain cleaned question once (deterministic call)
        cleaned_resp = self.query_llm(prm_prompt.replace("temperature\": 0", "temperature\": 0"))
        try:
            cleaned_json = json.loads(cleaned_resp)
            cleaned_q_text = cleaned_json.get("denoised_question", "")
            # FIX: Ensure cleaned_q_text is a string
            if isinstance(cleaned_q_text, list):
                cleaned_q_text = ' '.join(str(item) for item in cleaned_q_text)
            elif not isinstance(cleaned_q_text, str):
                cleaned_q_text = str(cleaned_q_text)
        except Exception:
            cleaned_q_text = question  # fallback

        # Second pass: sample diverse candidate solutions
        candidates = []
        for _ in range(n_candidates):
            cand_resp = self._query_llm_sample(prm_prompt, temperature=0.9, top_p=0.9)
            candidates.append(cand_resp)

        # Score & select
        scored = [(cand, self._prm_score(cand, cleaned_q_text)) for cand in candidates]
        best = max(scored, key=lambda x: x[1])[0]
        return best

    def query_llm(
            self,
            prompt: str,
            max_retries: int = 6,
            base_delay: float = 1.0,
            max_tokens: int = None,
    ) -> str:
        """Query the LLM with greedy decoding settings (deterministic)."""
        time.sleep(0.05)

        if max_tokens is None:
            max_tokens = self.max_tokens_main

        payload = self._build_payload(
            prompt=prompt,
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
        )

        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    url=self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=(10, 120),
                )

                if response.status_code == 200:
                    result = response.json()
                    choices = result.get("choices", [])
                    if not choices:
                        return f"Error: No choices in response - {result}"

                    choice = choices[0]
                    msg = choice.get("message", {})
                    content = (msg.get("content") or "").strip()
                    finish_reason = choice.get("finish_reason")

                    if content:
                        if finish_reason == "length":
                            print("[main] Warning: output truncated by max_tokens.")
                        return content

                    if finish_reason == "length":
                        return f"Error: Empty content and truncated by length - {result}"

                    return f"Error: Empty content from model - {result}"

                if response.status_code in (429, 500, 502, 503, 504):
                    wait_time = base_delay * (2 ** attempt)
                    print(f"[main] HTTP {response.status_code}, retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue

                return f"Error: HTTP {response.status_code} - {response.text}"

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt == max_retries - 1:
                    return f"Error: Request failed after retries - {e}"
                wait_time = base_delay * (2 ** attempt)
                print(f"[main] Connection error: {e} – retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)

            except requests.exceptions.RequestException as e:
                return f"Error: Request failed - {e}"

        return "Error: Max retries exceeded in query_llm"

    def save_progress(self, progress_data: Dict):
        """Save progress to pickle file"""
        try:
            with open(self.progress_file, 'wb') as f:
                pickle.dump(progress_data, f)
        except Exception as e:
            print(f"Warning: Could not save progress: {e}")

    def load_progress(self) -> Dict:
        """Load progress from pickle file"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load progress: {e}")
        return {}

    def save_method_results(self, results: List[Dict], dataset_name: str,
                            noise_level: str, method: str, output_dir: str = "gpto3_results"):
        """Save results for a specific method to organized subfolder structure"""

        # Create organized directory structure: gpto3_results/dataset_name/
        dataset_output_dir = Path(output_dir) / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        # Save as CSV in the dataset-specific folder
        result_filename = f"{noise_level}_{method}.csv"
        csv_path = dataset_output_dir / result_filename

        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)

        print(f"  ✅ Saved method results: {csv_path} ({len(results)} responses)")

        return str(csv_path)

    def generate_responses_for_dataset(self, data: List[Dict], dataset_name: str,
                                       noise_level: str, method: str, num_samples: int = 1,
                                       use_cot: bool = True, output_dir: str = "gpto3_results") -> List[Dict]:
        """Generate LLM responses for a specific dataset, noise level, and method"""

        print(f"\n{'=' * 70}")
        print(f"Generating responses: {dataset_name.upper()} - {noise_level.upper()} - {method.upper()}")
        print(f"  - Model: {self.model}")
        print(f"  - Chain of Thought: {use_cot}")
        print(f"  - Samples: {num_samples}")
        print(f"  - Noise Level: {noise_level}")
        print(f"{'=' * 70}")

        # Load existing progress
        progress = self.load_progress()
        progress_key = f"{dataset_name}_{noise_level}_{method}"

        # Take first num_samples
        sampled_data = data[:num_samples]

        # Check if we have existing results
        if progress_key in progress:
            existing_results = progress[progress_key]
            print(f"📋 Found {len(existing_results)} existing results, continuing from there...")
            start_idx = len(existing_results)
            results = existing_results.copy()
        else:
            results = []
            start_idx = 0

        # Continue generation from where we left off
        for i, sample in enumerate(tqdm(sampled_data[start_idx:],
                                        desc=f"Generating {dataset_name}_{noise_level}_{method}",
                                        initial=start_idx, total=len(sampled_data))):
            # time.sleep(1.5)
            actual_idx = start_idx + i

            # Get the question based on noise level
            question = sample.get(noise_level, '')

            if not question:
                print(f"Warning: No question found for noise level '{noise_level}' at index {actual_idx}")
                continue

            # Generate response based on method
            if method == "prm":
                llm_response = self.query_llm_prm(question)
                prompt = self.create_prompt(question, "denoise_cot")
            elif method == "mcts":
                llm_response = self.query_llm_mcts(question)
                prompt = self.create_prompt(question, "cot")
            else:
                prompt = self.create_prompt(question, method)
                llm_response = self.query_llm(prompt)

            # Store result
            result = {
                'dataset': dataset_name,
                'noise_level': noise_level,
                'method': method,
                'index': actual_idx,
                'question': question,
                'prompt': prompt,
                'llm_response': llm_response,
                'original_question': sample.get('original', ''),
                'low_noise_question': sample.get('low_noise', ''),
                'medium_noise_question': sample.get('medium_noise', ''),
                'high_noise_question': sample.get('high_noise', '')
            }
            results.append(result)

            # Save progress every 5 generations
            if (actual_idx + 1) % 5 == 0:
                progress[progress_key] = results
                self.save_progress(progress)

            # Add delay to avoid rate limiting
            time.sleep(0.2)

            # Progress update
            if (actual_idx + 1) % 10 == 0:
                print(f"Progress: {actual_idx + 1}/{len(sampled_data)} responses generated")

        # Final save of progress
        progress[progress_key] = results
        self.save_progress(progress)

        # Save method-specific results to organized folder structure
        self.save_method_results(results, dataset_name, noise_level, method, output_dir)

        print(f"\nCompleted {dataset_name}_{noise_level}_{method}: {len(results)} responses generated")

        return results

    def run_full_generation(self, datasets: Dict[str, List[Dict]], num_samples: int = 1,
                            output_dir: str = "gpto3_results") -> Dict[str, List[Dict]]:
        """Generate responses for all combinations of datasets, noise levels, and methods"""

        all_results = {}

        # We define the noise levels and methods
        noise_levels = ["original", "low_noise", "medium_noise", "high_noise"]
        # noise_levels = ["low_noise", "medium_noise", "high_noise"]
        methods = [
            # ("normal", False),  # Direct prompt
            # ("cot", True),  # Chain of Thought
            #("denoise_cot", False),  # Denoise Chain of Thought
             ("macro_action", False),  # Macro Actions
            #("prm", False),  # Process-Reward Modeling
            # ("mcts", False),  # Monte-Carlo Tree Search
        ]

        total_combinations = len(datasets) * len(noise_levels) * len(methods)
        current_combination = 0

        for dataset_name, data in datasets.items():
            print(f"\n🔍 Starting response generation for {dataset_name}...")

            for noise_level in noise_levels:
                for method_name, use_cot in methods:
                    time.sleep(1.0)
                    current_combination += 1
                    print(
                        f"\n📊 [{current_combination}/{total_combinations}] Processing {noise_level} + {method_name}...")

                    results = self.generate_responses_for_dataset(
                        data=data,
                        dataset_name=dataset_name,
                        noise_level=noise_level,
                        method=method_name,
                        num_samples=num_samples,
                        use_cot=use_cot,
                        output_dir=output_dir
                    )

                    result_key = f"{dataset_name}_{noise_level}_{method_name}"
                    all_results[result_key] = results

        return all_results

    def save_results(self, results: Dict[str, List[Dict]], output_dir: str = "gpto3_results"):
        """Save all results in both organized subfolders and combined formats"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\n💾 Finalizing results in {output_dir}...")

        # Results are already saved in organized subfolders during generation
        # This method now creates the combined summary files

        # Save combined results
        all_results = []
        for result_data in results.values():
            all_results.extend(result_data)

        if all_results:
            all_df = pd.DataFrame(all_results)
            all_df.to_csv(output_path / "all_responses_combined.csv", index=False)

            with open(output_path / "all_responses_combined.jsonl", 'w', encoding='utf-8') as f:
                for result in all_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')

        # Create summary of what was generated
        summary = {}
        for result_key, result_data in results.items():
            if result_data:
                dataset_name = result_data[0]['dataset']
                if dataset_name not in summary:
                    summary[dataset_name] = {}

                noise_level = result_data[0]['noise_level']
                method = result_data[0]['method']

                if noise_level not in summary[dataset_name]:
                    summary[dataset_name][noise_level] = {}

                summary[dataset_name][noise_level][method] = len(result_data)

        # Save summary
        with open(output_path / "generation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✅ All results organized in {output_dir}/")
        print(f"   - Individual results saved in dataset subfolders")
        print(f"   - Combined results: all_responses_combined.csv")
        print(f"   - Generation summary: generation_summary.json")
        print(f"Total responses: {len(all_results)}")
        return str(output_path)


def main():
    """Main function to run LLM response generation"""

    # We used OPEN ROUTER FOR THE API
    API_KEY = "YOUR_API_KEY"

    # We changed the model and sample size here.
    MODEL = "microsoft/phi-4"
    NUM_SAMPLES = 300
    OUTPUT_DIR = "microsoft_phi-4_results_re_execution_macro"
    BASE_PATH = "datasets_300_noisy/"

    print("LLM Response Generation for Mathematical Datasets")
    print("=" * 80)
    print(f"Model: {MODEL}")
    print(f"Samples per combination: {NUM_SAMPLES}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Base path: {BASE_PATH}")
    print()
    print("Target Datasets:")
    print("  1. GSM8K Original")
    print("  2. GSM8K Symbolic (Apple)")
    print("  3. Omni-MATH")
    print("  4. MATH-500")

    print()
    print("Noise Levels:")
    print("  1. original")
    print("  2. low_noise")
    print("  3. medium_noise")
    print("  4. high_noise")
    print()
    print("Methods:")
    print("  1. denoise_cot (denoise + CoT)")
    print("  2. macro_action")
    print("  3. prm (Process-Reward Modeling)")
    print("  4. mcts (Monte-Carlo Tree Search)")
    print()

    # Initialize generator
    generator = LLMResponseGenerator(API_KEY, MODEL)

    # Load datasets
    print("Step 1: Loading datasets...")
    datasets = generator.load_all_datasets(BASE_PATH)

    if not datasets:
        print("❌ No datasets found! Please check your file paths.")
        return

    # Calculate total API calls
    total_methods = 1  # denoise_cot, macro_action, prm, mcts
    total_combinations = len(datasets) * 4 * total_methods  # 4 noise levels * 4 methods
    total_calls = total_combinations * NUM_SAMPLES
    estimated_time = total_calls * 1.0 / 60

    print(f"\nStep 2: Generating responses...")
    print(f"Total combinations: {total_combinations}")
    print(f"This will make approximately {total_calls} API calls")
    print(f"Estimated time: ~{estimated_time:.1f} minutes")
    print(f"Progress will be saved every 5 generations to 'response_generation_progress.pkl'")
    print(f"Individual method results will be saved to organized subfolders in '{OUTPUT_DIR}/'")

    # Ask for confirmation
    confirm = input("\nDo you want to proceed? (y/N): ")
    if confirm.lower() != 'y':
        print("Response generation cancelled.")
        return

    # Run full generation
    results = generator.run_full_generation(datasets=datasets, num_samples=NUM_SAMPLES, output_dir=OUTPUT_DIR)

    # Save results
    print(f"\nStep 3: Finalizing results...")
    output_path = generator.save_results(results, OUTPUT_DIR)

    # Print summary
    print(f"\n{'=' * 80}")
    print("RESPONSE GENERATION COMPLETE!")
    print(f"{'=' * 80}")

    print(f"\nResults organized in: {output_path}")
    print(f"Progress file: response_generation_progress.pkl")
    print(f"\nFolder structure:")
    print(f"  {OUTPUT_DIR}/")
    for dataset_name in datasets.keys():
        print(f"    ├── {dataset_name}/")
        print(f"    │   ├── original_denoise_cot.csv")
        print(f"    │   ├── original_macro_action.csv")
        print(f"    │   ├── original_prm.csv")
        print(f"    │   ├── original_mcts.csv")
        print(f"    │   ├── low_noise_denoise_cot.csv")
        print(f"    │   ├── ... (all noise levels × methods)")
        print(f"    │   └── high_noise_mcts.csv")
    print(f"    ├── all_responses_combined.csv")
    print(f"    ├── all_responses_combined.jsonl")
    print(f"    └── generation_summary.json")
    print(f"\nTo resume interrupted generation, simply run the script again!")

    # Show what was generated
    print(f"\nGenerated responses for:")
    for result_key in results.keys():
        print(f"  - {result_key}")


if __name__ == "__main__":
    main()
