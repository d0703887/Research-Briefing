import json
import math
import argparse
import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
from tqdm import tqdm  # You might need to pip install tqdm
from textwrap import dedent
from pydantic import BaseModel, Field

# Rich imports for pretty printing
from rich.console import Console
from rich.markdown import Markdown

# Import your existing structures
from a import PaperMetadata, ArxivCurator
from llm import query_ollama

logger = logging.getLogger(__name__)

class Cluster(BaseModel):
    id: str  # Unique ID (e.g., "cluster_01")
    topic: str # Short name (e.g., "Vision Language Models")
    description: str # One sentence scope (e.g., "Papers focusing on aligning visual and textual modalities.")
    paper_ids: List[str] = Field(default_factory=list)

class ClusteringDecision(BaseModel):
    paper_id: str
    action: str  # "ASSIGN", "RENAME", "NEW"
    target_cluster_id: Optional[str] = None # Required for ASSIGN/RENAME
    new_topic_name: Optional[str] = None    # Required for RENAME/NEW
    new_description: Optional[str] = None   # Required for RENAME/NEW
    reasoning: str # Why did you make this decision?

class TopicCluster:
    def __init__(self, model_name: str = "llama3.1", max_workers: int = 1):
        """
        Args:
            model_name: The Ollama model to use (e.g., 'llama3', 'mistral').
            max_workers: Number of parallel requests.
                         Keep at 1 for most local GPUs to avoid Out-Of-Memory errors.
                         Increase if using a hosted API or a massive multi-GPU setup.
        """
        self.model_name = model_name
        self.clusters: Dict[str, Cluster] = {}
        self.max_workers = max_workers

    def _format_existing_clusters(self) -> str:
        """Creates a lean string representation of current clusters for the prompt."""
        if not self.clusters:
            return "No existing clusters. (All papers must be NEW)."

        lines = []
        for cid, cluster in self.clusters.items():
            lines.append(f"- ID: {cid} | Topic: {cluster.topic}")
            lines.append(f"  Desc: {cluster.description}")
        return "\n".join(lines)

    def _format_new_papers(self, papers: List['PaperMetadata']) -> str:
        lines = []
        for p in papers:
            # We use the tags we generated in the previous step!
            lines.append(f"Paper ID: {p.arxiv_id}")
            lines.append(f"Title: {p.title}")
            lines.append(f"Abstract: {p.abstract}")
            lines.append("---")
        return "\n".join(lines)

    def _construct_topic_prompt(self, paper: 'PaperMetadata') -> str:
        return dedent(f"""
        You are a generic research librarian. Your goal is to categorize the following paper based on WHAT it achieves, not HOW it achieves it.

        ### Instructions
        1. Return a valid JSON list of 1 to 3 strings.
        2. The first string should be the **General Research Field** (e.g., "Video Language Models", "Robotics").
        3. The second and third string (optional) should be the **Specific Goal or Constraint** (e.g., "Efficiency", "Real-time Execution", "High Fidelity").
        4. **CRITICAL:** Do NOT include specific method's details' (e.g., do NOT say "Codec Primitives", "Contrastive Learning", "Q-LoRA"). Focus on the problem being solved.

        ### Examples
        Example 1:
        Title: CoPE-VideoLM: Codec Primitives For Efficient Video Language Models
        Abstract: ...leverage video codec primitives... reduces the time-to-first-token by up to 86%...
        Output: 
        ```json
        ["Video Language Models", "Efficient Video Language Models"]
        ```

        Example 2:
        Title: Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution
        Abstract: ...optimized for high performance and fast and smooth real-time execution...
        Output: 
        ```json
        ["VLA Model", "Real-time Execution"]
        ```

        Example 3:
        Title: LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models
        Abstract: ...introduces two new tokens to support long video understanding ...
        Output: 
        ```json
        ["Vision Language Models", "Long Video Understanding"]
        ```
        
        ### Input
        Title: {paper.title}
        Abstract: {paper.abstract} 
        
        ### Output
        """)
        # Truncate abstract to ~1500 chars to save context window/speed

    def _clean_and_parse_response(self, response: str) -> List[str]:
        """
        Robustly parses the LLM output, handling cases where it adds conversational filler.
        """
        try:
            # 1. Try direct JSON parsing
            return json.loads(response.split("```json")[-1].split("```")[0])
        except json.JSONDecodeError:
            logger.error("Tagging: LLM failed to return JSON. Skipping paper.")
            return


    def process_single_paper(self, paper: 'PaperMetadata') -> 'PaperMetadata':
        """
        Worker function to process a single paper.
        """
        sys_prompt = "You are a helpful research assistant. You output only valid JSON lists."
        user_prompt = self._construct_topic_prompt(paper)

        response = query_ollama(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            model=self.model_name,
            temperature=0.1  # Low temp for deterministic formatting
        )

        topics = self._clean_and_parse_response(response)
        paper.topics = topics

        return paper

    def extract_topics(self, papers: List['PaperMetadata']) -> List['PaperMetadata']:
        logger.info(f"Extracting topics for {len(papers)} papers using {self.model_name}...")

        results = []

        # Use TQDM for a progress bar
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Map the process function to the papers
            future_to_paper = {executor.submit(self.process_single_paper, p): p for p in papers}

            for future in tqdm(as_completed(future_to_paper), total=len(papers), desc="Tagging Papers"):
                try:
                    updated_paper = future.result()
                    results.append(updated_paper)
                except Exception as e:
                    paper = future_to_paper[future]
                    logger.error(f"Failed to process paper {paper.arxiv_id}: {e}")
                    # Return original paper with empty topics on failure
                    paper.topics = []
                    results.append(paper)

        return results

    def _construct_cluster_prompt(self, papers_batch: List['PaperMetadata']) -> str:
        existing_str = self._format_existing_clusters()
        papers_str = self._format_new_papers(papers_batch)

        return dedent(f"""
        You are an expert research librarian. Group the following research papers into 1-indexed clusters.

        ### CURRENT STATE
        The following clusters already exist:
        {existing_str}

        ### NEW PAPERS TO SORT
        {papers_str}

        ### INSTRUCTIONS
        For EACH new paper, output a JSON object with one of these 3 actions:
        1. "ASSIGN": Fits perfectly into an existing cluster.
        2. "RENAME": Fits into an existing cluster, but the cluster name/desc needs to be generalized to include this new paper.
        3. "NEW": Does not fit ANY existing cluster. Create a new one.

        ### OUTPUT FORMAT
        Return a JSON LIST of objects. Example:
        [
          {{
            "paper_id": "2310.12345",
            "action": "ASSIGN",
            "target_cluster_id": "cluster_1",
            "reasoning": "Paper discusses CLIP, fits Vision-Language."
          }},
          {{
            "paper_id": "2310.67890",
            "action": "RENAME",
            "target_cluster_id": "cluster_2",
            "new_topic_name": "Generative Audio & Speech",
            "new_description": "Updated to include speech synthesis alongside audio generation.",
            "reasoning": "This paper is about speech, previous topic was just Audio."
          }},
           {{
            "paper_id": "2310.11111",
            "action": "NEW",
            "new_topic_name": "Robotics Control",
            "new_description": "Papers dealing with motor control and actuators.",
            "reasoning": "No existing cluster matches robotics."
          }}
        ]
        """)

    def process_batch(self, batch: List['PaperMetadata']):
        """
        Runs the LLM on a batch and updates internal state.
        """
        prompt = self._construct_cluster_prompt(batch)
        system_prompt = "You are a precise JSON generator. Output ONLY valid JSON."

        # Call LLM
        response_str = query_ollama(prompt, system_prompt, self.model_name, temperature=0.1)

        try:
            # reuse your cleaning logic from before
            decisions_json = json.loads(response_str.split("```json")[-1].split("```")[0])
            # In production, iterate and validate against ClusteringDecision pydantic model
        except json.JSONDecodeError:
            logger.error("LLM failed to return JSON. Skipping batch.")
            return False

        # EXECUTE DECISIONS
        for d in decisions_json:
            p_id = d.get('paper_id')
            action = d.get('action')

            if action == "ASSIGN":
                cid = d.get('target_cluster_id')
                if cid in self.clusters:
                    self.clusters[cid].paper_ids.append(p_id)
                else:
                    logger.warning(f"LLM hallucinated cluster ID {cid}. Treating as NEW.")
                    self._create_new_cluster(p_id, d)

            elif action == "RENAME":
                cid = d.get('target_cluster_id')
                if cid in self.clusters:
                    # Update metadata
                    self.clusters[cid].topic = d.get('new_topic_name')
                    self.clusters[cid].description = d.get('new_description')
                    self.clusters[cid].paper_ids.append(p_id)
                else:
                    self._create_new_cluster(p_id, d)

            elif action == "NEW":
                self._create_new_cluster(p_id, d)

        return True

    def _create_new_cluster(self, paper_id, decision_dict):
        new_cid = f"cluster_{len(self.clusters.keys()) + 1}"
        new_cluster = Cluster(
            id=new_cid,
            topic=decision_dict.get('new_topic_name', 'Uncategorized'),
            description=decision_dict.get('new_description', 'No description'),
            paper_ids=[paper_id]
        )
        self.clusters[new_cid] = new_cluster

    def cluster_topics(self, all_papers: List['PaperMetadata'], batch_size=10):
        # Sort by relevance so high-impact papers define the initial clusters
        # This acts as "canonical initialization"
        sorted_papers = sorted(all_papers, key=lambda x: x.relevance_score, reverse=True)
        num_papers = len(sorted_papers)
        for i in range(0, num_papers, batch_size):
            batch = sorted_papers[i: i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{math.ceil(num_papers // batch_size)}...")

            success = False
            while not success:
                success = self.process_batch(batch)

        return self.clusters

    def generate_markdown_report(self, all_papers: List[PaperMetadata]) -> str:
        """
        Generates a structured Markdown string from the clustering results.
        """
        # Create a lookup for easy access to paper details
        paper_map = {p.arxiv_id: p for p in all_papers}

        md_lines = []
        md_lines.append("# 📚 ArXiv Research Cluster Report")
        md_lines.append(f"> Generated on {datetime.now().strftime('%Y-%m-%d')} | Total Papers: {len(all_papers)}")
        md_lines.append("---")

        # Sort clusters by number of papers (descending)
        sorted_clusters = sorted(self.clusters.values(), key=lambda c: len(c.paper_ids), reverse=True)

        for cluster in sorted_clusters:
            if not cluster.paper_ids:
                continue

            # Cluster Header
            md_lines.append(f"## 📂 {cluster.topic}")
            md_lines.append(f"_{cluster.description}_")
            md_lines.append("")  # Empty line for spacing

            # List Papers
            for pid in cluster.paper_ids:
                paper = paper_map.get(pid)
                if not paper:
                    continue

                # Format: - [Title](URL) (Date)
                # We use a slight hack for terminals: most modern terminals support clickable links.
                # Markdown format: [Link Text](URL)

                # Clean up authors (first 3 et al)
                author_str = ", ".join(paper.authors[:2])
                if len(paper.authors) > 2:
                    author_str += " et al."

                # Create the list item
                # **Title** - Authors
                # [📄 Link] | [🏷️ Tags]
                md_lines.append(f"### [{paper.title}]({paper.arxiv_url})")
                md_lines.append(f"- **Authors:** {author_str}")
                md_lines.append(f"- **Date:** {paper.published_date.strftime('%Y-%m-%d')}")

                if paper.topics:
                    tags = ", ".join([f"`{t}`" for t in paper.topics])
                    md_lines.append(f"- **Tags:** {tags}")

                md_lines.append("")  # Spacing between papers

            md_lines.append("---")

        return "\n".join(md_lines)

    def run(self, papers, batch_size):
        self.extract_topics(papers)
        self.cluster_topics(papers, batch_size)

        console = Console()

        # Generate the Markdown string
        report = self.generate_markdown_report(papers)

        # Render it beautifully
        console.print(Markdown(report))

        return report


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ArXiv Paper Curator & Clusterer")

    # Arguments
    parser.add_argument("--model", type=str, default="ministral-3:8b",
                        help="Ollama model name (default: ministral-3:8b)")
    parser.add_argument("--days", type=int, default=1,
                        help="Number of past days to fetch papers from (default: 1)")
    parser.add_argument("--max-results", type=int, default=50,
                        help="Max papers to fetch from ArXiv (default: 50)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Number of papers to cluster at once (default: 5)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for topic extraction (default: 1)")

    args = parser.parse_args()

    # 1. Fetch Papers
    logger.info(f"Fetching papers from the last {args.days} days...")
    curator = ArxivCurator(lookback_days=args.days, max_results=args.max_results)
    papers = curator.run()

    if not papers:
        logger.info("No papers found matching criteria.")
        exit(0)

    logger.info(f"Fetched {len(papers)} papers.")

    # 2. Initialize Clusterer
    clusterer = TopicCluster(model_name=args.model, max_workers=args.workers)

    # 3. Extract Topics (Tagging)
    clusterer.run(papers, args.batch_size)

