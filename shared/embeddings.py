import hashlib


def get_embedding(client, text):
    """
    Converts text into a vector embedding using Gemini.
    Raises on failure so callers can handle it explicitly.
    Single source of truth — used by both ingest.py and agent.py.
    """
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values


def get_embeddings_batch(client, texts):
    """
    Embeds a list of texts in one API call instead of N calls.
    Use this in ingest.py to avoid per-chunk round trips.
    Returns a list of embedding vectors in the same order as input.
    """
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
    )
    return [e.values for e in result.embeddings]


def content_fingerprint(text):
    """
    MD5 hash of the full chunk content.
    More reliable than first-50-chars for deduplication
    since it won't falsely flag chunks that happen to share
    an opening phrase but differ in body content.
    """
    return hashlib.md5(text.encode()).hexdigest()