import arxiv
import requests
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from pydantic import BaseModel, Field

# --- Configuration & Constants ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Data Models (Pydantic) ---
class PaperMetadata(BaseModel):
    """Schema for a single research paper."""
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    published_date: datetime
    arxiv_url: str
    # Signals
    is_hf_daily: bool = False  # Is it trending on HuggingFace?
    hf_upvotes: int = 0  # Placeholder for future HF API expansion
    # Topics
    topics: List[str] = Field(default_factory=list)

    # Final Metrics
    relevance_score: float = 0.0



# --- The Curator Agent ---
class ArxivCurator:
    def __init__(self, lookback_days: int = 7, max_results: int = 200):
        self.lookback_days = lookback_days
        self.max_results = max_results
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=3,
            num_retries=3
        )

    def _get_date_range(self):
        """Calculates the date range for the ArXiv query."""
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=self.lookback_days)

        # For HF API
        iso_format_str = end_date.strftime("%G-W%V")

        return start_date, end_date, iso_format_str

    # TODO: There are trending papers and weekly papers, stick to weekly papers for now
    def fetch_hf_daily_papers(self, iso_format_string) -> List[str]:
        """
        Fetches the list of ArXiv IDs currently trending on Hugging Face.
        This acts as a 'Community Filter'.
        """
        try:
            # Hugging Face Daily Papers API endpoint, weekly papers
            url = f"https://huggingface.co/api/daily_papers?week={iso_format_string}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Extract arxiv IDs from the HF response
                # Note: The API structure changes occasionally, robust parsing is needed.
                # Usually returns a list of objects with 'paper' -> 'id'
                trending_ids = []
                for day_data in data:
                    # specific parsing logic depends on current HF API response structure
                    # Assuming list of papers with 'id' field which is the arxiv ID
                    if 'paper' in day_data and 'id' in day_data['paper']:
                        trending_ids.append(day_data['paper']['id'])
                logger.info(f"Fetched {len(trending_ids)} trending papers from Hugging Face.")
                return trending_ids
            else:
                logger.warning(f"Failed to fetch HF papers. Status: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error fetching HF data: {e}")
            return []

    def calculate_score(self, paper: PaperMetadata):
        """
        The 'Industrial' Scoring Formula.
        """
        #TODO: add more influence factors.
        # Github stars, Author impact, Social media, Affiliation
        score = 0.0

        # Factor 1: Community Hype (Highest Weight)
        if paper.is_hf_daily:
            score += 95  # Massive boost if it's on HF Daily Papers

        # Factor 4: Collaboration (Heuristic)
        # Papers with many authors often imply larger experiments/labs
        if len(paper.authors) > 5:
            score += 5.0

        paper.relevance_score = score

        return paper

    def run(self) -> List[PaperMetadata]:
        logger.info("Starting ArXiv Curator Node...")

        start_date, end_date, iso_format_str = self._get_date_range()

        # 1. Get Trending IDs first (to tag papers as they come in)
        hf_trending_ids = self.fetch_hf_daily_papers(iso_format_str)

        # 2. Build ArXiv Query (CV and NLP)
        # Query format: cat:cs.CV OR cat:cs.CL
        search = arxiv.Search(
            query="cat:cs.CV OR cat:cs.CL OR cat:cs.AI OR cat:cs.LG",
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        papers_collected = []

        # 3. Iterate and Process
        for result in self.client.results(search):
            # Stop if we go past our lookback window
            if result.published < start_date:
                break

            # Create Pydantic Model
            clean_id = result.get_short_id().split("v")[0]
            paper_meta = PaperMetadata(
                arxiv_id=clean_id,
                title=result.title,
                abstract=result.summary.replace("\n", " "),
                authors=[a.name for a in result.authors],
                published_date=result.published,
                arxiv_url=result.entry_id,
                is_hf_daily=(clean_id in hf_trending_ids),
            )

            # Score
            self.calculate_score(paper_meta)
            papers_collected.append(paper_meta)

        # 4. Filter and Sort
        # Sort by score descending
        papers_collected.sort(key=lambda x: x.relevance_score, reverse=True)

        logger.info(
            f"Collected {len(papers_collected)} papers. Top score: {papers_collected[0].relevance_score if papers_collected else 0}")

        # Return top 50
        return papers_collected[:50]


# --- Usage Example ---
if __name__ == "__main__":
    curator = ArxivCurator(lookback_days=7, max_results=2000)  # Small test
    top_papers = curator.run()

    print(f"\nTop {len(top_papers)} Papers:")
    for p in top_papers:
        print(f"[{p.relevance_score:.1f}] {p.title}")
        if p.is_hf_daily:
            print("   -> Trending on HuggingFace!")