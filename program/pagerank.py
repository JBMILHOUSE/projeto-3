import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus  
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next, given a current page.
    With probability `damping_factor`, choose a link at random linked to by `page`. With probability `1 - damping_factor`, choose a link at random chosen from all pages in the corpus.
    """

    probabilities = dict()
    num_pages = len(corpus)
    links = corpus[page]
    num_links = len(links)

    if num_links > 0:
        for link in corpus:
            probabilities[link] = (1 - damping_factor) / num_pages
            if link in links:
                probabilities[link] += damping_factor / num_links
    else:
        for link in corpus:
            probabilities[link] = 1 / num_pages
    
    return probabilities


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages according to transition model, starting with a page at random.
    Return a dictionary where keys are page names, and values are their estimated PageRank value (a value between 0 and 1). All PageRank values should sum to 1.
    """

    page_rank = { page: 0 for page in corpus }
    page = random.choice(list(corpus.keys()))

    for _ in range(n):
        page_rank[page] += 1
        probabilities = transition_model(corpus, page, damping_factor)
        page = random.choices(list(probabilities.keys()), weights=probabilities.values(), k=1)[0]

    total_visits = sum(page_rank.values())
    for page in page_rank:
        page_rank[page] /= total_visits

    return page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating PageRank values until convergence.
    Return a dictionary where keys are page names, and values are their estimated PageRank value (a value between 0 and 1). All PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    page_rank = { page: 1 / num_pages for page in corpus }
    new_ranks = page_rank.copy()
    tolerance=0.001

    while True:
        for page in corpus:
            rank_sum = 0
            for possible_page in corpus:
                if len(corpus[possible_page]) > 0:
                    if page in corpus[possible_page]:
                        rank_sum += page_rank[possible_page] / len(corpus[possible_page])
                    
                else:
                    rank_sum += page_rank[possible_page] / num_pages

            new_ranks[page] = (1 - damping_factor) / num_pages + damping_factor * rank_sum
        
        differences = [abs(new_ranks[page] - page_rank[page]) for page in corpus]
        if max(differences) < tolerance:
            break

        page_rank = new_ranks.copy()
    
    return page_rank


if __name__ == "__main__":

    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")

    corpus = crawl(sys.argv[1])

    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")    

    print(f"PageRank Results from Iteration")
    ranks = iterate_pagerank(corpus, DAMPING)
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")   