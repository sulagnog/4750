"""
Reddit Data Collection Module
Scrapes comments from startup-related subreddits using Reddit's JSON API.
No API key required - uses public JSON endpoints.
"""

import requests
import pandas as pd
import time
import re
from typing import List, Dict, Optional
from pathlib import Path

# Reddit JSON API settings
REDDIT_BASE_URL = "https://www.reddit.com"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Target subreddits for startup-related discussions
TARGET_SUBREDDITS = ["startups", "Startup_Ideas", "Entrepreneur", "smallbusiness", "SaaS"]

# Keywords to identify startup idea threads
STARTUP_KEYWORDS = ["idea", "mvp", "startup", "feedback", "saas", "launch", "validate", "business", "product"]


def get_subreddit_posts(subreddit: str, limit: int = 25, sort: str = "hot") -> List[Dict]:
    """
    Fetch posts from a subreddit using Reddit's JSON API.
    
    Args:
        subreddit: Name of the subreddit (without r/)
        limit: Number of posts to fetch (max 100)
        sort: Sort method - 'hot', 'new', 'top', 'rising'
    
    Returns:
        List of post data dictionaries
    """
    url = f"{REDDIT_BASE_URL}/r/{subreddit}/{sort}.json"
    params = {"limit": min(limit, 100)}
    headers = {"User-Agent": USER_AGENT}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        posts = []
        for child in data.get("data", {}).get("children", []):
            post_data = child.get("data", {})
            posts.append({
                "post_id": post_data.get("id"),
                "subreddit": subreddit,
                "title": post_data.get("title", ""),
                "selftext": post_data.get("selftext", ""),
                "score": post_data.get("score", 0),
                "num_comments": post_data.get("num_comments", 0),
                "created_utc": post_data.get("created_utc"),
                "permalink": post_data.get("permalink", ""),
            })
        
        return posts
    
    except requests.RequestException as e:
        print(f"Error fetching posts from r/{subreddit}: {e}")
        return []


def get_post_comments(subreddit: str, post_id: str, limit: int = 50) -> List[Dict]:
    """
    Fetch comments from a specific post using Reddit's JSON API.
    
    Args:
        subreddit: Name of the subreddit
        post_id: Reddit post ID
        limit: Maximum number of comments to fetch
    
    Returns:
        List of comment data dictionaries
    """
    url = f"{REDDIT_BASE_URL}/r/{subreddit}/comments/{post_id}.json"
    params = {"limit": limit, "depth": 2}
    headers = {"User-Agent": USER_AGENT}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        comments = []
        
        # Comments are in the second element of the response array
        if len(data) > 1:
            comment_data = data[1].get("data", {}).get("children", [])
            comments = extract_comments_recursive(comment_data, subreddit, post_id)
        
        return comments[:limit]
    
    except requests.RequestException as e:
        print(f"Error fetching comments for post {post_id}: {e}")
        return []


def extract_comments_recursive(children: List, subreddit: str, post_id: str, depth: int = 0) -> List[Dict]:
    """
    Recursively extract comments from nested Reddit comment structure.
    """
    comments = []
    
    for child in children:
        if child.get("kind") != "t1":  # t1 = comment
            continue
        
        comment_data = child.get("data", {})
        body = comment_data.get("body", "")
        author = comment_data.get("author", "")
        
        # Skip deleted/removed comments and bot comments
        if body in ["[deleted]", "[removed]"] or author in ["AutoModerator", "[deleted]"]:
            continue
        
        comments.append({
            "comment_id": comment_data.get("id"),
            "post_id": post_id,
            "subreddit": subreddit,
            "author": author,
            "text": body,
            "score": comment_data.get("score", 0),
            "created_utc": comment_data.get("created_utc"),
            "depth": depth,
        })
        
        # Get replies (nested comments)
        replies = comment_data.get("replies")
        if isinstance(replies, dict) and depth < 2:
            reply_children = replies.get("data", {}).get("children", [])
            comments.extend(extract_comments_recursive(reply_children, subreddit, post_id, depth + 1))
    
    return comments


def is_startup_related(title: str, selftext: str = "") -> bool:
    """
    Check if a post is related to startup ideas based on keywords.
    """
    text = (title + " " + selftext).lower()
    return any(keyword in text for keyword in STARTUP_KEYWORDS)


def collect_startup_comments(
    subreddits: List[str] = TARGET_SUBREDDITS,
    posts_per_subreddit: int = 20,
    comments_per_post: int = 30,
    delay: float = 1.0
) -> pd.DataFrame:
    """
    Collect comments from startup-related threads across multiple subreddits.
    
    Args:
        subreddits: List of subreddit names to scrape
        posts_per_subreddit: Number of posts to fetch per subreddit
        comments_per_post: Maximum comments to fetch per post
        delay: Delay between requests (to avoid rate limiting)
    
    Returns:
        DataFrame with all collected comments
    """
    all_comments = []
    
    for subreddit in subreddits:
        print(f"\n📥 Fetching posts from r/{subreddit}...")
        posts = get_subreddit_posts(subreddit, limit=posts_per_subreddit)
        
        # Filter for startup-related posts
        startup_posts = [p for p in posts if is_startup_related(p["title"], p["selftext"])]
        print(f"   Found {len(startup_posts)} startup-related posts out of {len(posts)}")
        
        # If not enough startup posts, use all posts
        if len(startup_posts) < 5:
            startup_posts = posts
        
        for post in startup_posts:
            time.sleep(delay)  # Rate limiting
            
            print(f"   📝 Fetching comments from: {post['title'][:50]}...")
            comments = get_post_comments(subreddit, post["post_id"], limit=comments_per_post)
            
            # Add post title to each comment for context
            for comment in comments:
                comment["post_title"] = post["title"]
            
            all_comments.extend(comments)
            print(f"      Found {len(comments)} comments")
    
    # Create DataFrame
    df = pd.DataFrame(all_comments)
    
    if len(df) > 0:
        # Remove duplicates
        df = df.drop_duplicates(subset=["comment_id"])
        print(f"\n✅ Total unique comments collected: {len(df)}")
    else:
        print("\n⚠️ No comments collected!")
    
    return df


def save_raw_data(df: pd.DataFrame, output_path: str = "data/raw_comments.csv"):
    """Save raw comments to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"💾 Saved {len(df)} comments to {output_path}")


if __name__ == "__main__":
    # Collect comments from startup subreddits
    print("🚀 Reddit Startup Comments Collector")
    print("=" * 50)
    
    comments_df = collect_startup_comments(
        subreddits=TARGET_SUBREDDITS,
        posts_per_subreddit=15,
        comments_per_post=25,
        delay=1.5  # Be respectful to Reddit's servers
    )
    
    if len(comments_df) > 0:
        save_raw_data(comments_df, "data/raw_comments.csv")
        
        # Print summary
        print("\n📊 Collection Summary:")
        print(f"   Total comments: {len(comments_df)}")
        print(f"   Subreddits: {comments_df['subreddit'].nunique()}")
        print(f"   Unique posts: {comments_df['post_id'].nunique()}")
        print(f"\n   Comments per subreddit:")
        print(comments_df['subreddit'].value_counts().to_string())

