"""Stage-1 prompt + response parser for Food-101 description extraction.

Single source of truth — the SLURM script and analyze_pilot.py both import
from here so a prompt edit does not require changes in multiple places.
"""
import re


STAGE1_PROMPT = (
    "List 10 generic, visually-observable properties of the food in this image. "
    "Each property must be a predicate that could also apply to many other "
    "unrelated dishes — describe general traits, not dish identity. "
    "Avoid naming the dish or any unique signature ingredient. Prefer traits "
    "about serving vessel, base component, cooking method, texture, dominant "
    "color, garnish, sauce presence, and arrangement.\n"
    "\n"
    "Format every line as a lowercase snake_case predicate using one of:\n"
    "  served_in_<vessel>         (e.g. served_in_bowl, served_on_plate)\n"
    "  contains_<ingredient>      (e.g. contains_rice, contains_cheese)\n"
    "  has_<feature>              (e.g. has_sauce, has_green_garnish)\n"
    "  is_<property>              (e.g. is_fried, is_baked, is_round, is_golden_brown, is_charred, is_colorful)\n"
    "  texture_<type>             (e.g. texture_crispy, texture_creamy)\n"
    "\n"
    "Use broad, reusable words (rice, cheese, sauce, bread, meat, vegetable, "
    "bowl, plate, fried, baked, grilled). Do NOT use dish-specific nouns "
    "(pizza, sushi, donut, burger). Respond as a numbered list 1-10, one "
    "predicate per line, nothing else."
)

N_PHRASES = 10

# Matches lines like "1. crispy golden crust" or "3) thin tomato slices".
# Group 1 = index, group 2 = phrase text.
_PHRASE_RE = re.compile(r"^\s*(\d+)[.)\s]+\s*(.+?)\s*$", re.MULTILINE)


def parse_phrases(text: str, n: int = N_PHRASES) -> list[str]:
    """Extract up to `n` numbered phrases from a VLM response.

    Returns a list of length <= n, preserving numeric order. Missing indices
    are skipped (rather than padded) so a short list is a reliable signal
    that parsing under-delivered.
    """
    found: dict[int, str] = {}
    for m in _PHRASE_RE.finditer(text):
        idx = int(m.group(1))
        if 1 <= idx <= n and idx not in found:
            phrase = m.group(2).strip().rstrip(".,;:")
            if phrase:
                found[idx] = phrase
    return [found[i] for i in sorted(found) if i in found]
