"""Prompt set and score interpretation for MobileCLIP-S0.

CLIP cosine similarity is *not* a probability.  The values below are used as
a relative ranking signal only, and both the absolute positive score and the
margin against the negative prompts must clear a threshold before an image is
auto-saved.  Everything else is either queued for review or dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

POSITIVE_REAL: tuple[str, ...] = (
    "a photo of a man wearing a dress",
    "a photo of a man wearing a skirt",
    "a photo of a man in women clothing",
    "a photo of a male crossdresser",
    "a photo of a man in drag",
    "a photo of a feminine man",
)

POSITIVE_ANIME: tuple[str, ...] = (
    "an anime boy wearing a dress",
    "an anime boy wearing a skirt",
    "an anime boy in feminine clothing",
    "an anime femboy",
    "an anime crossdressing boy",
    "an anime illustration of a boy dressed as a girl",
)

NEGATIVE: tuple[str, ...] = (
    "a photo of a man in a suit",
    "a photo of a man in a shirt",
    "a photo of a woman",
    "a photo of a woman in a dress",
    "an anime girl",
    "an anime boy in a school uniform",
    "a landscape",
    "a pet",
    "a meme",
    "a still life",
    "a group photo of people",
    "a screenshot of text",
)

# Index ranges in the flat prompt list.
REAL_START = 0
REAL_END = len(POSITIVE_REAL)
ANIME_START = REAL_END
ANIME_END = ANIME_START + len(POSITIVE_ANIME)
NEG_START = ANIME_END
NEG_END = NEG_START + len(NEGATIVE)

PROMPTS: tuple[str, ...] = POSITIVE_REAL + POSITIVE_ANIME + NEGATIVE

assert len(PROMPTS) == NEG_END


@dataclass(frozen=True)
class Assessment:
    positive_score: float
    negative_score: float
    margin: float
    top_group: str
    """real_feminine / anime_feminine / negative"""
    top_prompt: str
    decision: str
    """saved / pending / rejected"""
    top5: tuple[tuple[str, float], ...]

    def as_dict(self) -> dict:
        return {
            "positive_score": round(self.positive_score, 6),
            "negative_score": round(self.negative_score, 6),
            "margin": round(self.margin, 6),
            "top_group": self.top_group,
            "top_prompt": self.top_prompt,
            "decision": self.decision,
            "top5": [(p, round(float(s), 6)) for p, s in self.top5],
        }


def assess(
    similarities: list[float] | tuple[float, ...],
    *,
    high_threshold: float = 0.040,
    pending_threshold: float = 0.010,
    min_positive_score: float = 0.115,
    min_pending_score: float = 0.095,
) -> Assessment:
    """Turn 24 cosine similarities into a conservative decision.

    ``saved`` requires *both* an absolute positive score and a positive margin
    over every negative prompt.  ``pending`` is the uncertain band.  A low
    score, or a negative top prompt, is rejected.
    """
    if len(similarities) != len(PROMPTS):
        raise ValueError(f"expected {len(PROMPTS)} similarities, got {len(similarities)}")

    real = max(similarities[REAL_START:REAL_END])
    anime = max(similarities[ANIME_START:ANIME_END])
    negative = max(similarities[NEG_START:NEG_END])
    positive = max(real, anime)
    margin = positive - negative

    top_idx = max(range(len(similarities)), key=lambda i: similarities[i])
    if REAL_START <= top_idx < REAL_END:
        top_group = "real_feminine"
    elif ANIME_START <= top_idx < ANIME_END:
        top_group = "anime_feminine"
    else:
        top_group = "negative"

    if positive < min_pending_score or margin < pending_threshold:
        decision = "rejected"
    elif positive >= min_positive_score and margin >= high_threshold:
        decision = "saved"
    else:
        decision = "pending"

    order = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:5]
    return Assessment(
        positive_score=float(positive),
        negative_score=float(negative),
        margin=float(margin),
        top_group=top_group,
        top_prompt=PROMPTS[top_idx],
        decision=decision,
        top5=tuple((PROMPTS[i], float(similarities[i])) for i in order),
    )
