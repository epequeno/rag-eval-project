"""
Phase 2 controlled corpus: engineered failure classes for the metric-coverage
question.

Post 2's differentiator (per plan.md) is not taxonomy breadth — Leung et al.
(arXiv 2510.13975) already enumerate 16 error types. The question here is:
*which Phase 1 metric catches which engineered failure?* So each class is a
pair of arms — ``control`` (failure absent) and ``engineered`` (failure
applied) — built from the same queries, differing only in the engineered
perturbation. Every arm is deterministic and self-authored: no network, no
randomness, no LLM at build time.

Answers are pre-authored where the class engineers the generator's behavior
(chunk boundaries, ranking, retrieval miss) and left to the real generator
where the phenomenon *is* the generator's behavior (lost-in-the-middle,
hallucination, multi-hop synthesis).
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class EngineeredQuery:
    """
    One engineered query, including the retrieved-context window exactly as the
    generator and the metrics will see it.

    ``gold_facts`` feed the deterministic retrieval metrics (substring
    containment, whitespace-normalized); ``contexts`` are rank-ordered. When
    ``answer`` is set, the runner uses it verbatim; when ``None`` the real
    generator produces it (for classes where generation is the phenomenon).
    """

    query: str
    gold_answer: str
    gold_facts: tuple[str, ...]
    contexts: tuple[str, ...]
    answer: str | None = None


@dataclass(frozen=True, slots=True)
class FailureSlice:
    """
    One failure class: paired control/engineered arms over identical queries.

    ``diagnostic`` is Post 2's per-class question ("which metric catches
    this?"), used in the coverage matrix header.
    """

    name: str
    diagnostic: str
    control: tuple[EngineeredQuery, ...]
    engineered: tuple[EngineeredQuery, ...]


# --------------------------------------------------------------------------- #
# Synthetic world: 12 fictional companies, fully self-authored
# --------------------------------------------------------------------------- #
# (name, founded_year, founder, state, product, ship_year, raised_millions, sector)

COMPANIES = (
    ("Aster Diagnostics", 1999, "Ada Lovelace", "Delaware", "molecular imaging systems", 2001, 34, "biotech"),
    ("Bramble Robotics", 2002, "Grace Hopper", "California", "warehouse drones", 2004, 52, "robotics"),
    ("Cinder Foundry", 1995, "Edith Clarke", "Ohio", "high-alloy castings", 1997, 18, "materials"),
    ("Driftwave Audio", 2008, "Alan Turing", "Washington", "studio monitors", 2010, 9, "audio"),
    ("Epoch Textiles", 1987, "Margaret Hamilton", "North Carolina", "technical fabrics", 1990, 25, "apparel"),
    ("Ferrum Optics", 2001, "Katherine Johnson", "Arizona", "precision lenses", 2003, 41, "optics"),
    ("Graveline Energy", 2012, "Barbara Liskov", "Texas", "grid batteries", 2015, 96, "energy"),
    ("Hollow Creek Foods", 1998, "Grace Murray", "Vermont", "plant-based spreads", 2000, 12, "food"),
    ("Ionvale Semiconductors", 2005, "Frances Allen", "Oregon", "low-power chips", 2007, 73, "semiconductors"),
    ("Juniper Carto", 2010, "Dorothy Vaughan", "Colorado", "field-mapping software", 2013, 33, "software"),
    ("Keystone Sailworks", 1992, "Mary Jackson", "Maine", "racing sails", 1995, 8, "marine"),
    ("Lumenforge", 2016, "Radia Perlman", "New York", "laser welders", 2018, 61, "manufacturing"),
)

# Filler context pool for lost-in-the-middle: plausible but irrelevant news.
FILLERS = (
    "The municipal budget hearing ran past midnight, with council members split on bridge repairs.",
    "A heat wave pushed grid demand to a record high across the eastern seaboard.",
    "The port authority approved a second container terminal after a year of studies.",
    "Inflation eased to 2.4 percent, the slowest pace in three years, on cooling rents.",
    "A rare frost damaged citrus groves in the northern part of the valley.",
    "The university's board voted to freeze tuition for a fourth consecutive year.",
    "Ferry operators cut weekend service after ridership fell sharply last autumn.",
    "The state lottery posted a record surplus, earmarked for rural schools.",
    "Meteorologists warned of coastal flooding as a slow storm lingered offshore.",
    "The symphony announced a farewell season for its longtime music director.",
)


def _fact(name: str, year: int, founder: str, state: str) -> str:
    return f"{name} was founded in {year} by {founder} in {state}."


def _fact_sentence(name: str, year: int, founder: str, state: str, product: str) -> str:
    return (
        f"{_fact(name, year, founder, state)} "
        f"Their first product, {product}, shipped a year later."
    )


# --------------------------------------------------------------------------- #
# Failure classes
# --------------------------------------------------------------------------- #


def chunk_boundary_slice() -> FailureSlice:
    """
    Retrieval is perfect (both halves of the fact are in the window) but the
    fact sentence is split across two chunks, so no single chunk contains the
    gold fact. The answer is unaffected — the generator sees the year.
    """
    control: list[EngineeredQuery] = []
    engineered: list[EngineeredQuery] = []
    for name, year, founder, state, product, *_ in COMPANIES:
        fact = _fact(name, year, founder, state)
        query = f"In what year was {name} founded?"
        control.append(
            EngineeredQuery(
                query=query,
                gold_answer=f"{year}",
                gold_facts=(fact,),
                contexts=(_fact_sentence(name, year, founder, state, product),),
                answer=f"{year}",
            )
        )
        engineered.append(
            EngineeredQuery(
                query=query,
                gold_answer=f"{year}",
                gold_facts=(fact,),
                contexts=(
                    f"{name} was founded in {year} by {founder} in",
                    f"{state}. Their first product, {product}, shipped a year later.",
                ),
                answer=f"{year}",
            )
        )
    return FailureSlice(
        name="chunk_boundary",
        diagnostic="Does recall@k catch a gold fact split across two retrieved chunks?",
        control=tuple(control),
        engineered=tuple(engineered),
    )


def false_neighbor_slice() -> FailureSlice:
    """
    A semantically adjacent but factually irrelevant chunk (same market,
    different company) ranks above the true context. Same answers, same
    window, different order.
    """
    control: list[EngineeredQuery] = []
    engineered: list[EngineeredQuery] = []
    n = len(COMPANIES)
    for i, (name, year, founder, state, product, *_other) in enumerate(COMPANIES):
        other = COMPANIES[(i + 1) % n]
        fact = _fact(name, year, founder, state)
        true_ctx = _fact_sentence(name, year, founder, state, product)
        distractor = (
            f"{other[0]} operates in the same market as {name} and "
            f"was founded in {other[1]}."
        )
        query = f"In what year was {name} founded?"
        control.append(
            EngineeredQuery(
                query=query,
                gold_answer=f"{year}",
                gold_facts=(fact,),
                contexts=(true_ctx, distractor),
                answer=f"{year}",
            )
        )
        engineered.append(
            EngineeredQuery(
                query=query,
                gold_answer=f"{year}",
                gold_facts=(fact,),
                contexts=(distractor, true_ctx),
                answer=f"{year}",
            )
        )
    return FailureSlice(
        name="false_neighbor",
        diagnostic="Does any metric see that the right evidence is ranked second, not first?",
        control=tuple(control),
        engineered=tuple(engineered),
    )


def retrieval_miss_slice() -> FailureSlice:
    """
    The true context never reaches the window; the generator is handed only
    an unrelated chunk and (correctly) refrains. recall goes to zero; the
    refusal is the template's fixed phrase.
    """
    control: list[EngineeredQuery] = []
    engineered: list[EngineeredQuery] = []
    for name, year, founder, state, _product, _ship, _raised, sector in COMPANIES:
        fact = _fact(name, year, founder, state)
        query = f"In what year was {name} founded?"
        control.append(
            EngineeredQuery(
                query=query,
                gold_answer=f"{year}",
                gold_facts=(fact,),
                contexts=(_fact_sentence(name, year, founder, state, "their flagship product"),),
                answer=f"{year}",
            )
        )
        engineered.append(
            EngineeredQuery(
                query=query,
                gold_answer=f"{year}",
                gold_facts=(fact,),
                contexts=(
                    f"The {sector} sector grew 12 percent last quarter, according to industry analysts.",
                ),
                answer="Insufficient information.",
            )
        )
    return FailureSlice(
        name="retrieval_miss",
        diagnostic="Does faithfulness flag an answer withheld because retrieval missed?",
        control=tuple(control),
        engineered=tuple(engineered),
    )


def lost_in_middle_slice() -> FailureSlice:
    """
    The evidence chunk is embedded in a run of irrelevant fillers — first
    (control) vs. fifth (engineered). Retrieval metrics see the fact either
    way; only the position of the evidence within the window changes.
    Answers are left to the real generator (Liu et al. 2024).
    """
    control: list[EngineeredQuery] = []
    engineered: list[EngineeredQuery] = []
    for name, year, founder, state, *rest in COMPANIES:
        fact = _fact(name, year, founder, state)
        query = f"In what year was {name} founded?"
        control.append(
            EngineeredQuery(
                query=query,
                gold_answer=f"{year}",
                gold_facts=(fact,),
                contexts=(fact, *FILLERS[:8]),
            )
        )
        engineered.append(
            EngineeredQuery(
                query=query,
                gold_answer=f"{year}",
                gold_facts=(fact,),
                contexts=(*FILLERS[:4], fact, *FILLERS[4:8]),
            )
        )
    return FailureSlice(
        name="lost_in_middle",
        diagnostic="Does answer correctness drop when the evidence sits mid-window?",
        control=tuple(control),
        engineered=tuple(engineered),
    )


def hallucination_slice() -> FailureSlice:
    """
    The correct context is present, but a parallel sentence about a second
    company offers a look-alike year the generator might swap in. Control has
    no distractor. Answers are left to the real generator.
    """
    control: list[EngineeredQuery] = []
    engineered: list[EngineeredQuery] = []
    n = len(COMPANIES)
    for i, (name, year, founder, state, *rest) in enumerate(COMPANIES):
        partner = COMPANIES[(i + 1) % n]
        fact = _fact(name, year, founder, state)
        query = f"In what year was {name} founded?"
        control.append(
            EngineeredQuery(
                query=query,
                gold_answer=f"{year}",
                gold_facts=(fact,),
                contexts=(_fact_sentence(name, year, founder, state, "their flagship product"),),
            )
        )
        engineered.append(
            EngineeredQuery(
                query=query,
                gold_answer=f"{year}",
                gold_facts=(fact,),
                contexts=(
                    _fact_sentence(name, year, founder, state, "their flagship product"),
                    f"{partner[0]} was founded in {partner[1]} by {partner[2]}.",
                ),
            )
        )
    return FailureSlice(
        name="hallucination",
        diagnostic="Does answer correctness drop when a look-alike fact is in context?",
        control=tuple(control),
        engineered=tuple(engineered),
    )


def evidence_split_slice() -> FailureSlice:
    """
    Answering requires combining two facts (funding year + shipped program).
    Control puts both facts in one context; engineered splits them across two
    contexts. Both arms contain every gold fact, so retrieval metrics are
    identical — only the synthesis demand differs.
    """
    control: list[EngineeredQuery] = []
    engineered: list[EngineeredQuery] = []
    for name, year, _founder, _state, product, ship_year, raised, *_ in COMPANIES:
        fact_a = f"{name} raised ${raised} million in {year}."
        fact_b = f"The {product} program shipped in {ship_year}."
        query = f"Which program did {name} ship, and in what year did it ship?"
        gold = f"{product} in {ship_year}"
        control.append(
            EngineeredQuery(
                query=query,
                gold_answer=gold,
                gold_facts=(fact_a, fact_b),
                contexts=(f"{fact_a} {fact_b}",),
            )
        )
        engineered.append(
            EngineeredQuery(
                query=query,
                gold_answer=gold,
                gold_facts=(fact_a, fact_b),
                contexts=(fact_a, fact_b),
            )
        )
    return FailureSlice(
        name="evidence_split",
        diagnostic="Do retrieval metrics notice that the answer requires two sources to combine?",
        control=tuple(control),
        engineered=tuple(engineered),
    )


ALL_SLICES: tuple[FailureSlice, ...] = (
    chunk_boundary_slice(),
    false_neighbor_slice(),
    retrieval_miss_slice(),
    lost_in_middle_slice(),
    hallucination_slice(),
    evidence_split_slice(),
)

SLICES_BY_NAME = {s.name: s for s in ALL_SLICES}