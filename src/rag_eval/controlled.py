"""
Phase 2 controlled corpus: engineered failure classes for the metric-coverage
question.

Post 2's differentiator (per plan.md) is not taxonomy breadth — Leung et al.
(arXiv 2510.13975) already enumerate 16 error types. The question here is:
*which Phase 1 metric catches which engineered failure?* So each class is a
pair of arms — ``control`` (failure absent) and ``engineered`` (failure
applied) — built from the same queries.

Design rules (revised 2026-10-17)
---------------------------------
The first version claimed the arms differ "only in the engineered
perturbation", but three classes also changed the window size or the evidence
rank, so their metric deltas were mechanical. Now:

* ``contexts`` is the **metric** order (retrieval rank). ``prompt_order`` is an
  optional permutation used only when rendering the generator prompt, so a
  position failure can be engineered without moving the evidence in the
  retrieval ranking.
* ``hallucination`` keeps a two-context window in both arms and pre-authors the
  answer, so the class measures whether a supported-but-wrong answer is caught,
  not whether a particular model happens to hallucinate.
* ``lost_in_middle`` keeps the evidence at rank 1 in both arms and moves it
  only in the rendered prompt. Answers are left to the real generator, because
  the phenomenon *is* the generator's behaviour; if it does not appear, that is
  a null result and must be reported as one.

Every arm is deterministic and self-authored: no network, no randomness, no
LLM at build time. Answers are pre-authored where the class engineers the
answer; generated live only where generation is the phenomenon.
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
    metrics see it.

    ``contexts`` are rank-ordered and feed the retrieval metrics.
    ``prompt_order`` optionally permutes them for the generator prompt only.
    ``gold_facts`` feed the deterministic retrieval metrics (substring
    containment, whitespace-normalized). When ``answer`` is set, the runner uses
    it verbatim; when ``None`` the real generator produces it.
    """

    query: str
    gold_answer: str
    gold_facts: tuple[str, ...]
    contexts: tuple[str, ...]
    answer: str | None = None
    prompt_order: tuple[int, ...] | None = None

    def prompt_contexts(self) -> tuple[str, ...]:
        if self.prompt_order is None:
            return self.contexts
        if sorted(self.prompt_order) != list(range(len(self.contexts))):
            raise ValueError("prompt_order must be a permutation of contexts")
        return tuple(self.contexts[i] for i in self.prompt_order)


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
# Synthetic world: 24 fictional companies, fully self-authored
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
    ("Northwind Analytics", 2011, "Ida Rhodes", "Illinois", "supply-chain forecasts", 2014, 27, "software"),
    ("Ochre Ceramics", 1989, "Evelyn Boyd Granville", "New Mexico", "kiln-fired tiles", 1992, 14, "materials"),
    ("Pallas Instruments", 2003, "Maryam Mirzakhani", "Massachusetts", "mass spectrometers", 2006, 58, "instruments"),
    ("Quarry Lane Brewing", 1996, "Annie Easley", "Wisconsin", "sour ales", 1998, 6, "beverages"),
    ("Redwood Logistics", 2000, "Karen Uhlenbeck", "California", "freight routing", 2002, 44, "logistics"),
    ("Sable Aerospace", 2007, "Mae Jemison", "Alabama", "composite airframes", 2010, 82, "aerospace"),
    ("Tessellate Software", 2014, "Shafi Goldwasser", "New York", "geometry kernels", 2016, 21, "software"),
    ("Umbra Metrology", 2009, "Ingrid Daubechies", "Colorado", "laser trackers", 2012, 37, "metrology"),
    ("Verdant Agritech", 2013, "Lynn Conway", "Iowa", "soil sensors", 2015, 29, "agriculture"),
    ("Wexford Marine", 1991, "Barbara McClintock", "Rhode Island", "tidal turbines", 1994, 19, "marine"),
    ("Xanthe Fabrics", 1985, "Mary Cartwright", "Georgia", "flame-resistant cloth", 1988, 11, "apparel"),
    ("Yarrow Health", 2017, "Jennifer Doudna", "Minnesota", "diagnostic swabs", 2019, 47, "health"),
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

    This is a metric **false alarm**, not a blind spot: recall and MRR collapse
    while the system still answers correctly.
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
        diagnostic="Does recall@k falsely flag a split fact whose answer is still present?",
        control=tuple(control),
        engineered=tuple(engineered),
    )


def false_neighbor_slice() -> FailureSlice:
    """
    A semantically adjacent but factually irrelevant chunk (same market,
    different company) ranks above the true context. Same answers, same
    window size, different order.
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
    The evidence chunk is first in the retrieval ranking in **both** arms; only
    its position in the rendered prompt changes (first vs fifth of nine).

    Retrieval metrics are therefore identical by construction. The answer is
    left to the real generator (Liu et al. 2024): if correctness does not drop,
    the phenomenon is absent and the matrix says so.
    """
    control: list[EngineeredQuery] = []
    engineered: list[EngineeredQuery] = []
    for name, year, founder, state, *rest in COMPANIES:
        fact = _fact(name, year, founder, state)
        query = f"In what year was {name} founded?"
        window = (fact, *FILLERS[:8])
        control.append(
            EngineeredQuery(
                query=query,
                gold_answer=f"{year}",
                gold_facts=(fact,),
                contexts=window,
            )
        )
        engineered.append(
            EngineeredQuery(
                query=query,
                gold_answer=f"{year}",
                gold_facts=(fact,),
                contexts=window,
                # Same rank for the metrics; context 0 renders fifth.
                prompt_order=(1, 2, 3, 4, 0, 5, 6, 7, 8),
            )
        )
    return FailureSlice(
        name="lost_in_middle",
        diagnostic="Does answer correctness drop when the evidence sits mid-prompt?",
        control=tuple(control),
        engineered=tuple(engineered),
    )


def hallucination_slice() -> FailureSlice:
    """
    A supported-but-wrong answer: the window contains the true fact at rank 1
    and a contradictory claim about the **same company** at rank 2. Both arms
    have the same window size and the same rank for the true evidence; only the
    second context differs. The engineered answer repeats the contradictory
    claim verbatim, so every claim in it is supported by a retrieved context.

    The answer is pre-authored, so this class tests the metrics: a faithful,
    context-supported answer can still be factually wrong. If the faithfulness
    judge ignores support and checks consistency, it will flag the answer
    anyway; that is a finding about the judge, not a failed experiment.
    """
    control: list[EngineeredQuery] = []
    engineered: list[EngineeredQuery] = []
    for name, year, founder, state, *rest in COMPANIES:
        fact = _fact(name, year, founder, state)
        true_ctx = _fact_sentence(name, year, founder, state, "their flagship product")
        wrong_year = year + 5
        contradictory = f"{name} was founded in {wrong_year} by {founder} in {state}."
        query = f"In what year was {name} founded?"
        control.append(
            EngineeredQuery(
                query=query,
                gold_answer=f"{year}",
                gold_facts=(fact,),
                contexts=(true_ctx, FILLERS[0]),
                # Full sentence so the claim is attributable: supported and correct.
                answer=f"{name} was founded in {year}.",
            )
        )
        engineered.append(
            EngineeredQuery(
                query=query,
                gold_answer=f"{year}",
                gold_facts=(fact,),
                contexts=(true_ctx, contradictory),
                # Verbatim from the second context: supported, and wrong.
                answer=f"{name} was founded in {wrong_year}.",
            )
        )
    return FailureSlice(
        name="hallucination",
        diagnostic="Does correctness catch a context-supported wrong answer that faithfulness misses?",
        control=tuple(control),
        engineered=tuple(engineered),
    )


def evidence_split_slice() -> FailureSlice:
    """
    Answering requires combining two facts. Control puts both facts in one
    context; engineered splits them across two contexts.

    Both arms contain every gold fact, so recall@4/@10 and MRR are identical;
    recall@1 differs because the engineered window has a second context that
    the control folds into the first. The class tests whether end-to-end
    metrics register a synthesis demand that window-level retrieval metrics
    mostly cannot see.
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
        diagnostic="Do end-to-end metrics notice a synthesis demand retrieval metrics cannot see?",
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
