# Oracle search — natural-language search over what a card *does*

2026-08-17. Status: **proposed — awaiting review**. Not approved, not implemented. No code written yet.

---

## What this is for

One query, from the person who asked for it:

> **"enchantments in green that let me draw and cost 5 or less"**

Today the only way to ask that is Scryfall's `t:enchantment id:g o:draw cmc<=5`, which requires
knowing four pieces of syntax and still gets the last part wrong. `o:` is **literal substring
matching**. `o:draw` finds "draw a card" and "draws two cards" — and also "drawn", "withdraw",
and every card that says "you may draw" in a mode you will never use. It misses nothing spelled
"draw" and it understands nothing. Ask it for "card advantage" and it returns cards containing
the literal string "card advantage", which is approximately zero cards, because Magic's
templating never uses that phrase.

The gap is real and it is narrow: **the structured half of a deckbuilding question is already
solved, and the mechanical half is not.** This design solves the mechanical half with retrieval
and a judge, keeps the structured half in SQL where it belongs, and is honest about which half
is carrying the query.

> [!IMPORTANT]
> **No artwork is involved in this feature, anywhere.** No vision model, no art crops, no
> `describe` pass, no `illustration_id`, no thumbnails in the output, no columns holding image
> URLs. The art corpus — 5,530 artworks, 170,487 propositions, ~16 hours of vision calls — is
> never read, never re-described, never re-embedded, and never joined against. That is not a
> scoping convenience; it is the property that makes this feature cheap enough to be worth
> building, and *An explicitly separate database* below turns it into a structural guarantee
> rather than a promise.

---

## Why this is viable at all

The art pipeline is expensive because of one stage: `describe` writes 5,530 open-ended vision
descriptions against an 81GB model, a ~16-hour pass on an RTX PRO 6000. Everything downstream —
embedding, indexing, retrieval — is minutes.

**This feature has no describe stage.** Scryfall already ships the text. There is no lossy
intermediate to generate, so the entire build is: download a 24MB file, split some strings, and
embed ~95,000 short texts. A prior run in this repo embedded **98,746 propositions in tens of
minutes**; the chunk count here is the same order of magnitude and the texts are shorter.

The consequence is worth stating in one line, because it justifies the whole proposal:

> **There is no expensive stage in this feature.** The first full build is under an hour of
> wall clock and the weekly increment is a few minutes. Nothing here competes for the 81GB
> vision model, so this adds **zero VRAM** to the serving process — `judge_model`
> (`qwen3.6:latest`, ~27GB) is already resident for `/scry` and is the only model an oracle
> search touches end to end.

---

## The shape of the query

`"enchantments in green that let me draw and cost 5 or less"` decomposes into **four** parts,
and three of them are structured:

| Clause | Kind | Where it is answered |
| :-- | :-- | :-- |
| `enchantments` | categorical | SQL — `card_types` |
| `in green` | categorical | SQL — `color_identity` |
| `cost 5 or less` | **numeric, with an operator** | SQL — `cards.cmc <= 5` |
| `that let me draw` | mechanical intent | retrieval over oracle-text chunks, then a judge |

**Three of the four clauses are structured, and that is typical of deckbuilding questions, not
an artefact of this example.** "Cheap white removal", "black creatures with deathtouch under 3
mana", "artifacts that make treasure" — the same shape every time. This shifts where the design
effort belongs relative to the art search: `/scry`'s hard problem is entirely semantic, because
"lonely" has no column. `/oracle`'s hard problem is **half semantic and half a parsing problem
that fails silently**, and the parsing half gets at least as much of this document as the
embedding half. That is deliberate and it is argued for in *Filters versus semantics* below.

---

## Mechanical precision — the make-or-break risk

Everything else in this design is tractable engineering. This is the part that decides whether
the feature is useful or actively harmful, so it goes first.

### Why it is worse here than in art search

A near-miss on `/scry` is a shrug. You asked for lonely, you got wistful, you scroll past. The
cost of a wrong result is one second of a human's attention.

A near-miss on `/oracle` is a **wrong card in a deck**. If the system claims a card draws and it
actually loots, the user builds around card advantage they do not have and finds out three games
later. A search that is confidently wrong about rules is worse than no search, because Scryfall's
`o:draw` — crude as it is — is never *wrong*. It is only incomplete. Replacing a tool that
under-answers with one that mis-answers is a downgrade, and this design has to earn its way past
that bar.

The specific danger is that the near-misses are **semantically adjacent and rules-wise distinct**.
An embedding model has no reason to separate these; they use the same words, in the same order,
about the same zones:

| Mechanic | Templating | Is it a draw? |
| :-- | :-- | :-- |
| **Draw** | "draw a card", "draws two cards", "draw a card for each…" | **Yes.** Library → hand, unconditionally. |
| **Loot** | "draw a card, then discard a card" | Partly. Net zero cards, filters quality. |
| **Rummage** | "discard a card, then draw a card" | Partly, and the order matters — you pay first. |
| **Impulse draw** | "exile the top card… you may play it this turn" | **No.** Never enters hand; expires. |
| **Reveal to hand** | "reveal the top card… put it into your hand" | **Yes, effectively** — a draw by another name. |
| **Reveal to bottom** | "reveal the top card… put it on the bottom" | **No.** |
| **Surveil** | "surveil 2" | **No.** Library ordering + self-mill only. |
| **Scry** | "scry 2" | **No.** Library ordering only. |
| **Tutor** | "search your library for a card… put it into your hand" | **No**, though it answers a related question. |
| **Wheel** | "each player discards their hand, then draws seven" | **Yes**, symmetrically. |

Cosine similarity between "draw a card" and "surveil 2" is high enough that pure vector retrieval
will interleave them. **Retrieval cannot fix this and is not asked to.** Retrieval's job is
recall; precision is the judge's, and the judge is given everything it needs to get these right.

### Four mechanisms, in order of how much they buy

**1. The corpus is ground truth, not a model's summary of ground truth.**

This is the single biggest advantage over `/scry` and it is free. The art judge reasons over
propositions a vision model wrote about a picture — a lossy intermediate that can lose a detail
permanently, which is why `verify_finalists` exists at all and why the README's *Known limits*
opens with hallucinated text-in-art. The oracle judge reasons over **Wizards' own Oracle text,
verbatim, byte for byte as Scryfall publishes it**. There is no intermediate to be wrong. When
the judge misreads a card, it misread text that is right in front of it, which is a much smaller
class of error than "the describer never recorded that detail".

A direct consequence: **there is no verification stage in this pipeline and there should not be
one.** `verify_finalists` exists because the art judge's evidence is second-hand. Here it is
first-hand. Adding a second model pass to re-read text the first pass already read is cost with
no new information.

**2. Chunking by ability, so the vector is about one thing.** See the next section.

**3. A judge prompt that names the distinctions instead of hoping.**

`JUDGE_RULES` in `cts/judge.py` already establishes the pattern: continuous fit, one-sentence
rationale, cited evidence ids, an explicit instruction not to credit outside knowledge. The
oracle judge's rules block keeps all four and adds a **mechanics rubric** — the table above,
compressed, in the system prompt, with the discriminating test for each family spelled out:

```
Distinguish these. They use the same words and mean different things:
  DRAW      cards move library -> hand. "draw a card", "draws N cards".
  LOOT      a draw with a discard attached in the same ability. Say so.
  RUMMAGE   discard first, then draw. The cost is paid before the card arrives.
  IMPULSE   "exile the top ... you may play it" — the card NEVER enters hand.
            This is not drawing. A query asking to draw is not satisfied by this.
  SURVEIL / SCRY   reorder or bin from the top. No card is drawn. Not a draw.
  REVEAL    "reveal the top card and put it into your hand" IS a draw in all but
            name. "...and put it on the bottom" is not.
  TUTOR     "search your library for a card and put it into your hand" is not
            drawing, though it answers a related question. Score it low for a
            draw query and say why in the rationale.
If the ability only draws under a condition the card itself never establishes,
say so and score it below 0.5.
```

This is a prompt, so it is the softest of the four mechanisms and the easiest to silently break.
It gets a test asserting the rubric still enumerates every named family (*Testing*, below) —
because a prompt edit that deletes two lines is invisible in review and catastrophic in output.

**4. The user sees the actual oracle text, always.**

The judge cites chunk ids using **exactly the batch-local renumbering trick `judge.number_batch`
already uses** — global ids are six digits, models copy them wrong, and the repo already fixed
this once (Defect 2, `tests/test_judge_props.py`). Citations outside a candidate's own declared
range are dropped and counted as misattributed, same as today.

But a cited id only proves the model *pointed at* a chunk, not that the chunk *says* what the
model claimed. So the rendered result carries the card's **full oracle text, verbatim**, with the
cited chunk marked. The rationale sits *below* it, not above.

That ordering is a deliberate inversion of the `/scry` embed, where `rationale` leads. On `/scry`
the "evidence" is itself model-written, so leading with the model's sentence costs nothing. Here
the evidence is authoritative and the rationale is the only model-written thing on screen, so the
authoritative text goes first and the claim about it goes second. **A loot-for-draw error is then
visible in the two seconds it takes to read the card.** We do not make the error impossible. We
make it *checkable at a glance*, which is the honest achievable goal and is strictly more than
`o:draw` offers.

### What error rate to expect, stated as a projection and not a measurement

This spec has not been run. The numbers below are **predictions**, and step 7 of *Order of work*
exists specifically to replace them with measurements before the Discord surface ships.

| Query class | Example | Expected precision@5 |
| :-- | :-- | :-- |
| Single unambiguous mechanic | "counter target spell", "gain life when a creature dies" | **85–95%** |
| Mechanic with a well-known near-miss | "that let me draw", "that make tokens" | **75–85%** |
| Vague strategic language | "card advantage", "value engine", "cheap interaction" | **55–70%** |
| Rules-interaction questions | "that get around ward", "that dodge board wipes" | **poor — see below** |

The middle row is the one this design is built for and the one the eval set must weight most
heavily. The bottom row is a failure this design does not attempt to fix.

### What cannot be fixed, named plainly

- **Recall failures are invisible.** The system can tell you the five cards it found. It cannot
  tell you about the sixth card it missed. A user who gets five good results has no way to know
  whether there were fifty. This is true of `o:draw` too, and true of every retrieval system, and
  it is still the largest honest limitation here. The output says how many cards passed the
  filters versus how many were judged, which at least bounds the gap.
- **The judge cannot reason about rules interactions.** It reads one card's text in isolation. It
  does not know the state of a board, does not know that a conditional trigger is unreachable in
  practice, and cannot evaluate a replacement effect against a layer system it has never been
  shown. Questions of the form "does X get around Y" will produce plausible garbage.
- **Templating drifts across 30 years.** Old cards say things modern cards do not, and modern
  Oracle updates have not normalised everything. A query phrased in current templating will
  under-retrieve pre-2003 cards. Nothing in this design fixes that; the BM25 arm of the hybrid
  helps a little because it matches the literal words that *are* shared.
- **Silver-bordered and acorn cards are in the corpus** (they are paper Magic, and the user asked
  for all paper Magic). Their oracle text is deliberately absurd and will occasionally rank. A
  `legal:commander` filter removes nearly all of them, and the *Not building* section explains
  why they are not excluded by default.
- **"Let me" is doing unexamined work in the example query.** "Enchantments that let me draw"
  arguably excludes symmetric wheels (which let *everyone* draw) and cards that make an opponent
  draw. The judge will read the pronoun and mostly get it right; it will not always. This is a
  genuine semantic subtlety and it is not solved, only surfaced through the rationale.

---

## Chunking

**Decision: split each card's oracle text on newlines into one chunk per ability, plus one
whole-card chunk. Embed both kinds. Retrieve over chunks, collapse to cards.**

### Why not one blob per card

Consider a two-ability enchantment:

```
Whenever a creature you control dies, draw a card.
At the beginning of your end step, you lose 1 life.
```

Embedded as one string, the vector sits between "creatures dying draws cards" and "you lose life
every turn". It is a weaker match for *both* queries than either clause would be alone, and the
dilution scales with ability count — so the cards that get hurt most are exactly the dense,
multi-ability cards that are most interesting to search for. A five-ability Saga or a Class
enchantment becomes an average of five unrelated ideas, which is a vector pointing at nothing.

This is not a hypothetical: it is the same failure `cts/index.py` already avoids on the art side
by embedding ~25 atomic propositions per artwork rather than one description per artwork, and by
collapsing to the card only *after* ranking (`search.collapse`'s docstring: "Collapsing earlier
would average one matching printing together with five non-matching ones and bury the hit"). The
structure here is the exact analogue:

```
art side:     props  ->  artwork  ->  card
oracle side:  chunks ->  card
```

**props : artwork :: chunks : card.** The retrieval, RRF fusion, best-chunk-per-card
deduplication and collapse logic are the same shape and should be written to look the same.

### Why the newline is the right boundary, and why we do not split further

Scryfall's `oracle_text` separates abilities with `\n`. That boundary is **authored by Wizards
and normalised by Scryfall** — it is not a heuristic we invented, it costs nothing to use, and it
is correct essentially by definition. Sagas, Classes, Levelers and modal cards all split cleanly
on it.

Splitting *further* — into sentences or clauses — is rejected. Magic text is hostile to sentence
tokenisation: `{T}:` contains a colon, `{1}{W}` contains braces, "1/1" and "+1/+1" contain
slashes, ability costs end with periods that are not sentence ends, and reminder text contains
whole nested sentences inside parentheses. A sentence splitter would be a pile of regexes with a
long tail of wrong answers, and the payoff is small: most single-line abilities express one idea
already. **The newline is free and correct; anything finer is expensive and approximate.** This
is the limit, and it means a compound single-line ability ("When this enters, draw a card and
lose 1 life") stays fused. That is an accepted, named cost.

### The whole-card chunk, and why both

The per-ability chunks lose two things: cross-ability context (an Aura's "enchanted creature
gets…" only makes sense with the "Enchant creature" line above it), and the card's gestalt (some
queries genuinely are about the whole card — "an engine that does three things"). One additional
chunk per card holds `type_line` + the full `oracle_text`, so both readings are retrievable.

Scoring is **max-over-chunks per card, via RRF rank fusion, never a sum**. Summing would reward
verbosity: a ten-ability card would outrank a perfect one-ability answer for having more chances
to score. `retrieve()` on the art side already does exactly the right thing here — it keeps "each
artwork's best-ranked proposition" per list before fusing — and the oracle version keeps each
card's best-ranked chunk the same way.

### Three text transforms before embedding, each justified

The chunk stored for display is **verbatim**. The chunk stored for *embedding* differs in three
ways, held in a separate `text_embedded` column so the display text is never contaminated:

1. **Substitute the card's own name with `this card`.** Oracle text names the card ("Sylvan
   Library draws you cards"). A proper noun embeds as whatever the words mean — "Bloodthirsty
   Blade" drags the vector toward blood and weapons, which is noise for a mechanical query.
   Substitution is an exact string replacement on a known value, so it is deterministic and
   cannot go wrong. Scryfall also supplies short-name variants for cards whose text uses a first
   name only; both forms are replaced.
2. **Keep reminder text.** The instinct is to strip parentheses as boilerplate. That is wrong
   here, and it is the single most load-bearing chunking choice after the newline split:
   **reminder text is the only English gloss a keyword ability has.** "Cycling {2}" embeds near
   nothing; "Cycling {2} (**{2}, Discard this card: Draw a card.**)" embeds near "draw", which is
   correct — cycling genuinely draws. Strip it and every keyword-only card becomes invisible to
   semantic search. Reminder text is also present inconsistently across printings, so Scryfall's
   Oracle text is used as the canonical form and the inconsistency is accepted.
3. **Do not prefix chunks with the type line.** Tempting, because "draw" means different things
   on an instant and an enchantment. Rejected: the type is a **hard SQL filter**, so the semantic
   layer never needs to carry it, and prefixing 95,000 chunks with a mostly-constant string
   dilutes every one of them. The whole-card chunk carries the type line for the queries that
   want it.

### Sizing

| | Estimate | Basis |
| :-- | --: | :-- |
| Cards (paper, unique oracle ids) | ~32,700 | verified live: Scryfall `game:paper` = 32,726 |
| Ability chunks | ~62,000 | ~1.9 newline-delimited lines per card; vanilla creatures contribute 0 |
| Whole-card chunks | ~32,700 | one each |
| **Total chunks** | **~95,000** | |
| Embedding matrix | **~292MB** | 95,000 × 768 × 4 bytes |
| BM25 structures | ~250MB | `rank_bm25` over 95,000 short docs |
| Index build | **~3–5s** | scaled from the art index's measured 4–7s over 170,487 rows |

These are estimates. Step 3 of *Order of work* replaces them with the real counts, and the README
gets the real numbers, not these.

---

## Filters versus semantics

Structured constraints are **hard SQL**, evaluated before anything is embedded. Only the
mechanical intent goes through retrieval. The router's job is to split the query in two, and
**getting that split right matters at least as much as the retrieval quality**, because three of
the four clauses in the user's own example land on the structured side.

### The filter algebra

**UNION within a field, AND across fields.** `type: planeswalker, artifact` means planeswalker OR
artifact; `type: artifact` plus `colors: G` means artifact AND green. This matches the design
already in discussion for the art search and it is the only algebra a user can hold in their head
without documentation.

There is no `NOT`, no nesting and no parentheses in v1 — see *Not building*.

| Field | Column / table | Semantics |
| :-- | :-- | :-- |
| `types` | `card_types(kind, value)` | matches supertype, type or subtype; UNION within |
| `colors` | `color_identity` (default) or `colors` | see the two modes below |
| `mv` | `cards.cmc` | numeric, with an operator |
| `power` / `toughness` | `power_num` / `toughness_num` | numeric; NULL for `*`/`X` — see below |
| `legal` | `card_legalities(format, status)` | UNION within; `status = 'legal'` |

**Colours have two modes and the output always says which one ran.** `identity` (default) means
`color_identity ⊆ requested`, which is the deckbuilding reading and matches `/scry --colors`'s
existing `color_set(c) <= wanted`. `includes` means the card's `colors` contains every requested
letter. They differ on multicolour cards: "enchantments in green" under `identity` **excludes**
Green-White enchantments, under `includes` it keeps them. Both readings are defensible for that
English sentence. The default is `identity` because its failure direction is conservative — too
few results, which is visible — rather than too many, which looks fine and is quietly wrong. The
echo line names the mode, and the Scryfall link (below) shows the user exactly how to flip it.

### Numeric filters, and the silent-failure problem

This is the highest-risk parsing surface in the design, and the risk is not that it fails — it is
that **it fails without any symptom**.

A mis-parsed categorical filter is self-announcing: ask for enchantments, get planeswalkers, and
the user sees it instantly. A mis-parsed *inequality* is not. Flip `cmc <= 5` to `cmc >= 5` and
the user gets five real green enchantments that really do draw cards and really are legal — they
just all cost seven mana. Every result is individually correct. Nothing looks wrong. The user has
to notice the pattern themselves, and a user who trusted the tool will not.

Five mechanisms, in order of how much they buy:

**1. The explicit path has no operator parsing at all.**

The Discord command exposes `mv_min` and `mv_max` as **integer options**, not a string like
`"<=5"`. Two integers with an inclusive-bounds convention cannot flip an inequality, because
there is no inequality to flip — the operator is encoded in which parameter you filled in, and
Discord's own UI labels and validates them. Same on the API: `mv_min: int | None`,
`mv_max: int | None`. **The only path that can silently invert a comparison is the router's, and
that path is soft and echoed.** Reducing the blast radius is worth more than any amount of
careful prompt engineering on a path that did not have to exist.

**2. The router emits `op` and `value` as separate schema-constrained fields.**

Never a free-form string. `op` is a JSON-schema `enum` of `["<", "<=", "=", ">=", ">", "between"]`
and `value` is a number (`lo`/`hi` for `between`), so the model cannot emit `"<=5"` and have a
character silently swallowed by a regex. `cts/search.py::ROUTER_SCHEMA` already establishes this
pattern with `SLOT_OPS`.

**3. A calibration table in the prompt, covering the traps.**

The boundary cases are where English is genuinely ambiguous and models genuinely get it wrong:

```
"5 or less" / "no more than 5" / "up to 5" / "at most 5" / "5 and under"  -> <= 5
"under 5" / "less than 5" / "below 5"                                     -> <  5
"5 or more" / "at least 5" / "5 and up"                                   -> >= 5
"over 5" / "more than 5" / "above 5"                                      -> >  5
"exactly two" / "two-drop" / "costs 3"                                    -> =  N
"three to five" / "between 3 and 5"                                       -> between 3, 5
```

Note that `<=` and `<` are one word apart in English and the difference is a whole mana value.

**4. Vague quantities produce no filter at all, and say so.**

"Cheap", "expensive", "big", "low to the ground", "top-end" have no defined numeric value.
Picking one — `cheap → cmc <= 3` — is a silent editorial decision that deletes correct answers
and that the user cannot see, disagree with, or discover. The router is instructed to emit **no
numeric filter** for a vague term.

But leaving "cheap" in the semantic string is nearly useless too, because oracle text never says
"cheap". So the response carries a note that names the gap and hands over the fix:

> `"cheap" has no defined mana value, so no cost filter was applied. Add mv_max:3 if that is
> what you meant.`

That is the honest answer *and* it teaches the escape hatch, which is better than either guessing
or silently ignoring the word.

**5. Every parsed filter is echoed back, in the first line of output, every time.**

```
filters: type = enchantment · colors ⊆ {G} (identity) · mv ≤ 5 · semantic: "let me draw"
```

Not in a debug field, not behind a flag, not only when something went wrong. **First line, always,
including when everything parsed correctly** — because a mis-parse is only visible against the
habit of reading a correct one. This is the mechanism that turns a silent failure into a visible
one, and it is the single most important line in this section. It gets a test.

Alongside it, a **Scryfall search link** that reproduces the structured half in Scryfall's own
syntax: `https://scryfall.com/search?q=t%3Aenchantment+id%3Ag+cmc%3C%3D5`. It is a second
independent rendering of the same parse, in a syntax the user may already know, on a site that
will show them the complete filtered set. If the parse is wrong, two things on screen are wrong
together — and if the user wanted the structured half all along, they now have it.

**A guard on absurd values.** A parsed `mv` outside 0–30 is dropped with a note. Magic's highest
real mana value in paper is 16 (Draco, Gleemax), so anything beyond 30 is a parse artefact rather
than a request, and applying it would silently return nothing.

**Rejected: a second model call to re-derive the operator.** It doubles the router cost to
double-check a value that the echo line already puts in front of the human who typed the query,
and two calls to the same model on the same ambiguous sentence are correlated, not independent.
The echo is cheaper and more reliable.

### Power and toughness: yes, but only where they are numbers

`power` and `toughness` are `TEXT` in Scryfall's data because they are frequently `*`, `X`, `1+*`,
`*/*` or `?`. Storing them as numbers loses cards; comparing them as strings gives nonsense
(`"10" < "2"`).

**Decision: store both.** `power` / `toughness` as `TEXT` verbatim for display, and derived
`power_num` / `toughness_num` as `REAL`, populated only when the value parses as a plain integer
and NULL otherwise. A `power >= 5` filter matches on `power_num` and therefore **excludes every
`*`/`X` creature** — and the response says so:

> `142 cards whose power is not a fixed number (*, X, 1+*) cannot be compared and were excluded.`

Justification for excluding rather than including: a `*/*` creature's power is a
characteristic-defining ability evaluated against a board state we do not have, so we genuinely
cannot answer whether it is ≥ 5. Including them would be guessing; excluding them silently would
hide a real class of answers. Naming the count is the only honest option. This also matches
Scryfall's own behaviour for `pow>=5`, so it will not surprise anyone who has used the syntax.

Loyalty gets the same treatment and the same NULL semantics.

### Hard versus soft, and what happens when the router is wrong

This follows the rule already agreed for the art search:

- **Explicit filters — the ones the user typed into a command option — are HARD.** They are never
  dropped, never relaxed, never widened. They may legitimately return zero results, and zero
  results with an honest message is the correct answer to a question with no answer. The user
  asked a precise question; second-guessing it is worse than answering it.
- **Router-inferred filters are SOFT.** They are applied broadest-first, and any one that would
  empty the pool is dropped with a note naming it. Numeric filters follow the same rule.

Note one deliberate difference from `search.allowed_illustrations`, which drops a slot filter as
soon as it would leave fewer than `MIN_FILTERED_POOL = 25` artworks. **That floor is not ported.**
It exists because the art slots are *independently incomplete* — an angel's wings land in
`species` on one artwork and `clothing` on another — so an intersection of two slot filters
describes no single describer's habits. Oracle columns have no such problem: they are complete,
authoritative and exact. A type filter leaving three cards has left the *correct* three cards, and
dropping it would be discarding a right answer for being small. **Soft filters here drop only at
zero.**

When the router mis-parses:

| Failure | Consequence | What the user sees |
| :-- | :-- | :-- |
| Router unreachable / returns garbage | No filters at all; pure semantic retrieval over all ~32,700 cards | `⚠️ query routing failed — searched the whole corpus with no filters. Results may be off-type.` Plus the results, which are degraded but real. |
| Inferred a filter the user did not mean | It is soft, so it applies — but it is on the echo line | The user reads `type = enchantment` and sees it was never asked for |
| Missed a filter the user did mean | Extra clause stays in the semantic string; recall suffers, precision does not | The echo line's `semantic:` half shows the leftover words |
| Inverted an operator | Plausible-looking wrong results | `mv ≥ 5` on the echo line, against a query that said "5 or less" |
| Vague quantity | No filter | An explicit note naming the word and the fix |

### The structural-only fast path

If the router finds **no mechanical intent at all** — "green enchantments costing 5 or less", with
no verb — then there is nothing to retrieve and nothing to judge. The SQL answer *is* the answer.

That path skips retrieval, embedding and judging entirely and returns the filtered set ordered by
`edhrec_rank` (present in the bulk data, a popularity proxy, and the only ranking signal this
corpus has). It completes in **under a second**, and it says what it did:

> `no mechanical intent in this query — these are 214 cards matching the filters, most-played
> first. Nothing was judged.`

This is not an optimisation bolted on; it falls out of the decomposition, and it makes `/oracle`
a competent structured search engine in addition to a semantic one.

---

## Corpus and schema

### An explicitly separate database

**`data/oracle.db`.** New config key `oracle_db_path`, defaulting to `data/oracle.db`, sitting
beside `db_path` in `config.toml` with a comment explaining that the two corpora are deliberately
disjoint.

Separate because they are different things at every level: different grain (one row per
`oracle_id` versus one row per `illustration_id`), different scope (32,726 paper cards versus
3,202 commander-legal ones), different refresh costs, different failure modes. Sharing a file
would mean every schema migration, every `VACUUM`, every restore-from-backup and every accidental
`DELETE` puts ~16 hours of vision work at risk to serve a corpus that can be rebuilt from scratch
in under an hour. **The separation is the backup strategy.**

### Schema

```sql
-- one row per gameplay identity, straight from Scryfall's oracle_cards bulk
CREATE TABLE IF NOT EXISTS cards (
  oracle_id      TEXT PRIMARY KEY,
  name           TEXT,
  type_line      TEXT,
  oracle_text    TEXT,           -- verbatim; faces joined with "\n//\n" as ingest.py does
  mana_cost      TEXT,
  cmc            REAL,
  colors         TEXT,           -- WUBRG-ordered, "" for colourless
  color_identity TEXT,           -- WUBRG-ordered
  power          TEXT,           -- verbatim: may be "*", "X", "1+*"
  toughness      TEXT,
  loyalty        TEXT,
  power_num      REAL,           -- NULL unless power parses as a plain integer
  toughness_num  REAL,
  loyalty_num    REAL,
  keywords       TEXT,           -- JSON array, Scryfall's own
  layout         TEXT,
  reserved       INTEGER,
  edhrec_rank    INTEGER,        -- the only popularity signal in this corpus
  released_at    TEXT,
  set_code       TEXT,           -- first printing, for the footer only
  scryfall_uri   TEXT
);

-- normalized so "type: planeswalker, artifact" is one indexed IN (...)
CREATE TABLE IF NOT EXISTS card_types (
  oracle_id TEXT,
  kind      TEXT,                -- supertype | type | subtype
  value     TEXT                 -- lowercased
);

CREATE TABLE IF NOT EXISTS card_legalities (
  oracle_id TEXT,
  format    TEXT,
  status    TEXT                 -- legal | not_legal | banned | restricted
);

-- one row per ability, plus one whole-card row. See "Chunking".
CREATE TABLE IF NOT EXISTS chunks (
  id            INTEGER PRIMARY KEY,
  oracle_id     TEXT,
  face_index    INTEGER,         -- 0 front, 1 back
  ordinal       INTEGER,         -- position within the face; used to mark the match
  kind          TEXT,            -- ability | whole
  text          TEXT,            -- VERBATIM, for display
  text_embedded TEXT             -- name-substituted; what actually got embedded
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
  chunk_id INTEGER PRIMARY KEY,
  vec      BLOB                  -- float32 numpy tobytes(), same as embeddings.vec
);

-- the same Phase 12 bookkeeping, keyed on oracle_id instead of illustration_id
CREATE TABLE IF NOT EXISTS queries    (id INTEGER PRIMARY KEY, text TEXT, kind TEXT,
                                       params TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS retrievals (query_id INTEGER, oracle_id TEXT, rank INTEGER,
                                       score REAL, method TEXT, chunk_id INTEGER);
CREATE TABLE IF NOT EXISTS judgments  (query_id INTEGER, oracle_id TEXT, fit REAL,
                                       rationale TEXT, chunk_ids TEXT, model TEXT,
                                       source TEXT);
CREATE TABLE IF NOT EXISTS meta       (key TEXT PRIMARY KEY, value TEXT);

CREATE INDEX IF NOT EXISTS idx_chunks_oracle_id   ON chunks(oracle_id);
CREATE INDEX IF NOT EXISTS idx_card_types_value   ON card_types(value, kind);
CREATE INDEX IF NOT EXISTS idx_card_types_oracle  ON card_types(oracle_id);
CREATE INDEX IF NOT EXISTS idx_legalities_format  ON card_legalities(format, status);
CREATE INDEX IF NOT EXISTS idx_cards_cmc          ON cards(cmc);
CREATE INDEX IF NOT EXISTS idx_judgments_query_id ON judgments(query_id);
```

`source` on `judgments` reuses `db.HUMAN_SOURCES` (`{"human", "discord"}`) — that constant exists
precisely so a new human-facing surface adds its value in one place, and `tests/test_provenance.py`
already guards it. The `queries` / `retrievals` / `judgments` tables have **no consumer in v1**;
they exist so the data accrues from the first search rather than from whenever an exporter is
written, which is the habit the repo already has.

`oracle_db.connect()` sets the same three pragmas as `cts/db.py::connect()` — `journal_mode=WAL`,
`foreign_keys=ON`, and `busy_timeout=30000`. That last one is not optional: the whole argument in
the service design for why it belongs in `db.connect()` rather than at one call site applies
identically to a second database file.

Estimated size on disk: ~20MB cards, ~15MB chunks, ~292MB vectors, ~50MB indexes — **~380MB**.

### Do the two databases ever need joining?

**No, and they must not be.** Both key on `oracle_id`, so a join is trivially available, which is
exactly why the temptation needs naming and refusing.

The thing that would tempt it is a genuinely appealing feature: *"green enchantments that draw,
that also look serene."* Two reasons that is still not a SQL join:

1. **The corpora have different scopes.** `commanders.db` holds 3,202 commander-legal cards;
   `oracle.db` holds ~32,700 paper cards. Any inner join silently drops ~90% of the oracle corpus,
   and the result looks like a working search that has quietly become commander-only.
2. **They have independent refresh cadences and independent staleness.** A cross-file `ATTACH`
   would read two snapshots that may be from different weeks — and worse, from two *different
   index generations*, since each has its own fingerprint. That is a correctness bug of exactly
   the silent kind the service design spends a page eliminating.

If the combined search is ever wanted, the correct shape is **two independent searches intersected
on `oracle_id` in the API layer**, where both sides' freshness and coverage are explicit and can
be reported. No `ATTACH`, no cross-database query, ever.

### What the API holds resident

| | Size |
| :-- | --: |
| Art index — 170,487 × 768 float32 | 523MB |
| Art BM25 (`rank_bm25` over 170,487 propositions) | ~400MB |
| **Oracle index — ~95,000 × 768 float32** | **~292MB** |
| **Oracle BM25 + chunk texts** | **~270MB** |
| Python, FastAPI, numpy, two SQLite connections | ~200MB |
| **Total RSS, steady state** | **~1.7GB** |

Against 62GB of RAM this is not close to a constraint, and it is worth saying so plainly rather
than engineering around a limit that does not exist. Peak roughly doubles for *one* index during
its rebuild — the new one is built before the old is dropped, same as today — and the two indexes
never rebuild simultaneously because both rebuilds run under the single existing engine lock.

`MemoryMax=` stays unset on `scrying-api.service`, for the reason already recorded: an OOM kill
mid-rebuild is worse than the spike it would prevent.

---

## Ingest, and reusing `cts/ingest.py` without touching the commander corpus

### A different bulk file, deliberately

Scryfall publishes an **`oracle_cards`** bulk type: one card object per Oracle ID. Verified live
today — `https://api.scryfall.com/bulk-data` lists it at 24,529,284 bytes compressed, alongside
`default_cards` at 77,517,651. Like every other type it now exposes only `jsonl_download_uri`.

Using `oracle_cards` rather than reusing the already-downloaded `default_cards` file:

- **It is exactly the right grain.** One row per `oracle_id` is what this feature searches. Reusing
  `default_cards` would mean re-implementing `parse_bulk`'s grouping and `pick_default`'s
  tie-breaking to collapse ~110,000 printings back down — machinery that exists on the art side
  only because *artwork* is per-printing. Here it is pure overhead with a bug surface.
- **It is a third of the size.** 24MB versus 77MB, once a week.
- **The two corpora stay independent.** Each keeps its own `meta.scryfall_updated_at`, so each
  skips its own work when its own bulk file has not moved. Coupling them to one stamp would mean
  a change in either forces work in both.

### What is reused, and the one edit to `cts/ingest.py`

Everything about *how* the download works is already correct in `cts/ingest.py` and is reused
verbatim: `USER_AGENT` (CONTRACT.md's network etiquette), `_download`'s chunked streaming through
a `.part` file so an interrupt never leaves a truncated file that later looks complete,
`iter_card_objects`'s three-format reader, `data_dir`, the `(15, 300)` timeouts, and the
`updated_at`-versus-`meta` skip.

The **one edit**: `bulk_entry()` currently hardcodes `BULK_TYPE = "default_cards"`. It gains a
parameter, `bulk_entry(bulk_type: str = BULK_TYPE)`, defaulting to the existing constant so no
existing caller changes behaviour. `_pick_source` gains the same treatment for its filename. That
is the whole change to `cts/` outside of the new modules — the same discipline the service design
applied when it edited exactly one line of `db.py`.

### Filtering to paper

`oracle_cards` covers every game. The filter is:

- `"paper" in card["games"]` — the decision, verbatim. Not commander-legal-only, not digital.
- Exclude `layout` in `{token, double_faced_token, emblem, art_series}` and `set_type` in
  `{token, memorabilia}` — tokens, emblems, art cards and oversized promos have no rules text
  worth searching and would pollute both retrieval and the count.

**The expected result is near 32,726, not exactly it.** Scryfall's web search applies its own
`-is:extra` semantics which this approximates rather than reproduces. The ingest checkpoint prints
the number it actually got, in the style `cts/ingest.py::checkpoint` already uses — it does not
assert an expected value. If the number drifts from 32,726 by more than a percent or so, the
exclusion list is wrong and the printed number is how anyone finds out.

### New modules

Flat, in `cts/`, with a shared `oracle_` prefix:

```
cts/oracle_db.py       schema, connect (three pragmas), meta helpers
cts/oracle_ingest.py   oracle_cards bulk -> cards, card_types, card_legalities
cts/oracle_chunk.py    oracle_text -> chunks. Pure functions, no I/O, no network.
cts/oracle_embed.py    chunks with no vector -> chunk_embeddings, batches of 32
cts/oracle_index.py    OracleIndex: matrix + BM25 + by_oracle_id groupings
cts/oracle_filters.py  router filter object -> SQL WHERE + params; the echo line
cts/oracle_search.py   route -> expand -> retrieve -> filter -> judge -> select
```

Flat rather than a `cts/oracle/` subpackage because every module in `cts/` is flat and named for
its phase; a prefix keeps `grep` working the way it does today, makes the two corpora visibly
separate in one `ls`, and avoids introducing the repo's first subpackage for no gain.

`cts/oracle_embed.py` is a near-copy of `cts/embed.py` — same `BATCH_SIZE = 32`, same
"select chunks with no `chunk_embeddings` row" idempotence, same `_prune_orphans` safety net, same
dimension-mismatch check with the same error message telling you to clear the table. It is a copy
rather than a generalisation of `cts/embed.py`: the two differ in table names, id column and the
`layer` concept that only one of them has, and parameterising three things to save forty lines
would make the art pipeline's embed stage harder to read in exchange for nothing.

CLI, matching `cts/__main__.py`'s existing style (lazy import per handler):

```
python -m cts oracle-ingest      bulk download + cards/types/legalities + chunk
python -m cts oracle-embed       embed chunks with no vector
python -m cts oracle "QUERY"     [--types] [--colors] [--mv-max] [--mv-min] [--legal] [-k] [--json]
```

---

## The search pipeline

```
route          1 judge_model call -> {filters, semantic_intent, notes}
  |
  +-- no semantic intent? -> SQL only, ordered by edhrec_rank, done in <1s
  |
expand         1 judge_model call -> 6-8 mechanical-register phrasings
sql filter     hard + soft filters -> the allowed oracle_id set
retrieve       dense + BM25 over chunks, masked to allowed, RRF-fused
collapse       best-ranked chunk per card
judge          4 batches of 10, chunk-id citations, mechanics rubric
select         diversify (see below), then k
```

**One expansion call, not two.** The art search runs two independent expansions because its corpus
has two registers — literal and interpretive — and the whole SPEC.md argument is that neither
alone serves both ends of the theme spectrum. Oracle text has **one register**: Wizards'
templating. So there is one expansion, and its job is the highest-leverage step in the pipeline —
bridging natural language to the exact phrasings the corpus actually uses:

```
"let me draw"  ->  "draw a card"
                   "draws two cards"
                   "draw a card for each ..."
                   "draw cards equal to ..."
                   "you may draw a card"
                   "target player draws a card"
```

That expansion is what `o:draw` cannot do and it is most of the feature's value.

**No layer weighting, no route blend.** There is no literal/interpretive split, so
`literal_weight` / `interpretive_weight` and `MIN_ROUTE_WEIGHT` have no analogue here. The router
returns filters and an intent string; that is all.

**`POOL_SIZE = 40`, not 100.** The art search judges 100 candidates because retrieval over
model-written descriptions is noisy and needs depth. Here the SQL filters have usually already
cut 32,700 cards to a few hundred *exactly*, and retrieval runs over authoritative text, so the
pool arrives much cleaner. 40 candidates in 4 batches of 10 is the right trade, and it is most of
why this is faster than `/scry`.

**Diversity: no colour cap, no MMR.** `judge.color_cap` exists because a visual theme attracts one
visual convention. A mechanical query has no such pull, and capping colours would actively harm
the user's own example — "enchantments in green" should return five *green* cards, and a colour
cap of 2 would refuse to. MMR over chunk vectors is likewise wrong: five cards that all draw cards
in slightly different ways is the correct answer to "cards that draw", not a redundancy to break
up. The one diversity rule kept is **at most one result per card**, which the collapse already
guarantees.

**`PASS_FIT = 0.5` and the STRETCH label are kept unchanged**, so the two commands agree on what a
weak result looks like and `serve/render.py`'s existing conventions carry over.

### Expected latency

| Stage | Estimate | Why |
| :-- | --: | :-- |
| route | 2–4s | one short structured JSON response |
| expand | 3–5s | 6–8 short statements, ~200 output tokens |
| embed expansions | <1s | one batched `nomic-embed-text` call |
| SQL filters | <50ms | indexed columns, ~32,700 rows |
| retrieve (dense + BM25) | <200ms | 95,000 × 768 dot product is milliseconds; BM25 over short docs likewise |
| **judge — 4 batches of 10** | **18–30s** | ~40 rationales ≈ 1,200 output tokens total |
| render | negligible | |
| **Total** | **~25–40s typical, ~55s worst** | |

**The judge dominates, at roughly 65–80% of wall clock**, and it dominates for one reason:
generating text is the only thing in the pipeline that is not either a dot product or a SQL seek.
The retrieval that people assume is the expensive part costs a fifth of a second.

Against `/scry`'s measured 76.8s mean and 106.7s worst, this should be **roughly half**, and the
savings are entirely attributable: no vision verification (8 multimodal calls against image crops),
no second expansion call, a pool of 40 instead of 100, and no model eviction — `judge_model` is the
only model involved end to end, so unlike `/scry` there is never a moment where 27GB and 81GB are
both wanted.

The structural-only fast path is **under a second**, and no-Ollama degraded mode (BM25 only,
unjudged) is likewise sub-second.

**These are projections from the art pipeline's measured per-call costs, not measurements.** Step
7 of *Order of work* measures them before anything ships, and the README gets the measured
numbers.

---

## Weekly refresh

**Three stages appended to the end of the existing `cts/refresh.py::run` stage list. One timer,
one journald log, one answer to "did the refresh fire".**

```
1. ingest      scryfall default_cards + cards/arts          (unchanged)
2. edhrec      edhrec, all cards, ~45 min at 1 req/s        (unchanged)
3. power       power scores                                 (unchanged)
4a. art        art downloads                                (unchanged)
4b. describe   vision pass, new artwork only                (unchanged)
4c. embed      embeddings                                   (unchanged)
6. oracle-ingest   oracle_cards bulk -> cards/types/legalities   NEW
7. oracle-chunk    re-chunk cards whose oracle_text changed      NEW
8. oracle-embed    embed chunks with no vector                   NEW
```

**Appended at the end, not inserted, and the position is the point.** `refresh.run` returns 1 on
the first stage that raises. New, less-proven code placed anywhere but last would be able to block
the art pipeline that has been working. Last, it cannot: every art stage is committed and done
before the oracle stages start, and a failure in stage 7 leaves a refresh that did its established
job and reported exactly which new stage broke — which is what `refresh.run`'s existing failure
message already says ("earlier stages are committed and idempotent").

**Not a second timer.** Two timers writing two databases at overlapping times invent a contention
problem that does not need to exist, double the journald surface, and give two different answers
to "when did this last update". The existing `cts-refresh.service` is oneshot with overlap
protection and `Persistent=true`; it needs no change beyond the three added stages, and
`install-timer.sh` needs no change at all.

**Cost added to a refresh: 3–15 minutes.** A quiet week is a 24MB download that is skipped because
`updated_at` has not moved, plus three no-op SQL selects — under a minute. A set release is ~400
new cards, ~1,200 new chunks, one batched embed pass — a few minutes. The stages are idempotent and
resumable in exactly the way the existing ones are: each selects only the rows lacking its output.

Against a run that already spends ~45 minutes polling EDHREC at 1 req/s, this is noise.

**Re-chunking when text changes.** Oracle text is *not* immutable the way artwork is — Wizards
issues Oracle updates, errata and templating normalisations. Stage 7 therefore stores a hash of
each card's `oracle_text` and re-chunks any card whose hash moved, deleting that card's chunks and
their embeddings together so stage 8 re-embeds them. This is the one place the oracle pipeline is
*more* complex than the art one, and it is because the underlying assumption differs: `describe.run`
can skip any `illustration_id` it has seen because artwork never changes. Text does.

### Index staleness

**Its own fingerprint, the same mechanism, no new machinery.** `serve/api.py` already has
`FINGERPRINT_SQL`, `corpus_fingerprint(conn)`, `Engine.ensure_current(force=)` and
`Engine.poll_once()` returning `skipped` / `suppressed` / `debounced` / `rebuilt` / `current`. The
oracle index gets a second fingerprint over its own database:

```sql
SELECT (SELECT value          FROM meta             WHERE key = 'last_oracle_refresh_at'),
       (SELECT MAX(id)        FROM chunks),
       (SELECT MAX(chunk_id)  FROM chunk_embeddings);
```

Same three-field rationale as the art one: a "something happened" marker for runs that changed
data the index cannot see, `MAX(id)` as a single seek to the end of an `INTEGER PRIMARY KEY`
(never `COUNT(*)`, which is a full scan), and `MAX(chunk_id)` to catch embed lagging behind chunk.
Same blind spots, same escape hatch (`POST /admin/reload`).

Same two placements: synchronously before every oracle search, so correctness is unconditional and
survives a manual `python -m cts oracle-ingest` in a terminal; and on the **same** 60-second
background tick, with the same 5-minute debounce and the same suppression while
`cts-refresh.service` is active. One poller checks both fingerprints — a second background task
would be a second thing to reason about for a check that costs three index seeks.

Same blocking-rebuild decision, and it is easier to defend here: ~3–5 seconds against a search
whose placeholder promised ~30s. `service.oracle_index_rebuilt: true` on the response makes the
extra seconds attributable.

### SQLite contention

Two database files, both WAL, both with `busy_timeout=30000`. The art API's reads never block the
refresh's writes and vice versa; writers still serialise per file, which is what the busy timeout
covers. Because the two files are separate, **the oracle stages of a refresh cannot block an
in-flight `/scry` at all**, and the art stages cannot block an `/oracle`. That is a real benefit of
the separate-database decision and worth noting alongside the backup argument.

**The GPU is not a concern here**, exactly as the brief states: no oracle stage runs a vision
model. The one interaction that remains is the existing one — a refresh whose `describe` stage is
loading the 81GB model will evict `judge_model`, so an `/oracle` search during that window pays the
same eviction ping-pong a `/scry` does. Same cause, same non-fix, same honest placeholder message.
The window is usually empty, for the reason already argued: most weeks describe nothing.

---

## Command surface

### `/oracle`

**Proposed name: `/oracle`.** Reasoning, since two other candidates were on the table:

- **`/oracle` wins on audience.** "Oracle text" is Wizards' own official term for a card's current
  rules text — the Oracle database is literally the canonical source, and Scryfall's `o:` is named
  after it. Every Magic player reads the word correctly on sight. The audience for this bot is
  Magic players.
- **`/rules` was the first instinct and is a trap.** It reads as "ask me a rules question", and
  someone will type `/oracle`-shaped input into it — *"can I respond to a triggered ability"* —
  and get five cards. The name would be actively inviting the one query class this design cannot
  answer.
- **`/cards` is what every command here does.** It distinguishes nothing.

The obvious objection to `/oracle` is that it sits in the same divinatory register as `/scry`, in a
project called Scrying Pool, so the two could read as variants of each other. Accepted, and handled
where it actually matters: **Discord shows each command's description string in the picker as the
user types**, so the disambiguation lands at the exact moment of choosing:

| Command | Description string | Backing |
| :-- | :-- | :-- |
| `/scry` | *find commanders by what their ARTWORK depicts, means or evokes* | local art corpus, 5,530 artworks, ~80s |
| `/oracle` | *find cards by what their RULES TEXT does — not a rules Q&A* | local oracle corpus, ~32,700 cards, ~30s |
| `/search` | *look up one card by name* (planned) | Scryfall API, no local DB, sub-second |

Three commands, three different questions, and the distinguishing word is capitalised in two of
them because that is the word that decides which one you wanted.

### Coexistence, and the two guards that make it work

`/search <name>` is planned and not part of this design, but the boundary is, because the
predictable user error is picking the wrong one of three commands and paying 30 seconds to find
out.

**Guard 1: `/oracle` detects a card name before spending anything.** If the query string matches a
row in `cards.name` exactly or near-exactly (case-insensitive, punctuation-stripped), reply
immediately:

> `"Sol Ring" is a card name. Try /search Sol Ring for the card itself — /oracle searches for
> cards by what they do.`

One indexed SQL lookup, no model calls, and it converts a 30-second wrong answer into an instant
right one. `/search` gets the mirror guard when handed a sentence.

**Guard 2: `/oracle` detects a rules question and refuses honestly.** A query opening with
*"can I"*, *"does"*, *"how does"*, *"what happens if"*, *"when do I"* is a rules question, not a
card search:

> `That reads as a rules question. /oracle searches card text — it does not answer rules
> questions. Try the Comprehensive Rules or a judge.`

Running the search anyway would produce five cards containing vaguely related words and present
them as an answer. **Refusing is the correct output**, and this is precisely the "name the failure
mode rather than manufacture a stretch" ethos the README's *Known limits* opens with.

Both guards are pure functions over a string and a database, and both are tested.

### The command signature

```
/oracle query:<text> [types:<text>] [colors:<WUBRG>] [mv_max:<int>] [mv_min:<int>]
                     [legal:<format>] [k:1-5]
```

Registered inside `serve/bot.py::register_commands` as a second `@tree.command` — nothing else in
the bot changes, since guild sync copies the whole tree. `mv_max` / `mv_min` are
`app_commands.Range[int, 0, 30]`, which is where the *no operator parsing on the explicit path*
property comes from. Every option is optional; `query` alone is the normal case.

---

## Result rendering

`serve/oracle_render.py`, mirroring `serve/render.py` exactly: **pure functions returning plain
dicts matching Discord's embed JSON, importing nothing but the standard library.** That property is
what `tests/test_render.py`'s docstring calls "the reason the suite is worth running", and it is
not being lost for a second surface.

Content line, above the embeds:

```
🔮 "enchantments in green that let me draw and cost 5 or less"
filters: type = enchantment · colors ⊆ {G} (identity) · mv ≤ 5 · semantic: "let me draw"
214 cards passed the filters · 40 judged · 4 of 5 clear the 0.5 fit bar
[refine on Scryfall](https://scryfall.com/search?q=t%3Aenchantment+id%3Ag+cmc%3C%3D5)
```

One embed per result:

| Element | Source |
| :-- | :-- |
| Title | `name` + `mana_cost`, then `· STRETCH` when below the 0.5 bar |
| Colour | green when it clears the bar, grey when it is a stretch |
| **Description** | **the full `oracle_text`, verbatim, in a code block**, with `▸` prefixing the matched chunk's line (located by `chunks.ordinal`, never by string search). Truncated at 700 chars with `…` and a pointer to the Scryfall link. |
| Field: Type | `type_line`, verbatim |
| Field: Mana value / Colours | `cmc` and `color_identity`, `C` when empty |
| Field: Fit | `fit` to two decimals, always shown |
| Field: Legal | up to six formats where `status = 'legal'`, comma-joined; plus `Banned in <format>` when the card is banned in a format the user filtered on |
| Rationale | one line, **below** the oracle text — see *Mechanical precision* |
| Footer | `first printed <SET> <year>` |
| Links | `[Scryfall](…)` |

**No image, no thumbnail, ever.** This is a decision, not an omission: an art thumbnail would
invite the reader to judge a mechanical result on the picture, and the whole feature exists to
separate those two questions. The oracle database stores no image URLs at all, so the embed cannot
grow one by accident. It gets a test asserting the embed dict contains neither an `image` nor a
`thumbnail` key.

**Only two verified/passing colours, not three.** `serve/render.py` has `COLOR_VERIFIED`,
`COLOR_PASSING` and `COLOR_STRETCH`, where the first two are distinguished by vision verification.
There is no verification stage here, so `COLOR_VERIFIED` has no meaning and reusing it would imply
a check that never ran.

**Feedback buttons**, reusing the existing pattern: `discord.ui.DynamicItem` with the identity in
the `custom_id` so buttons survive the bot restarts that the two-process split exists to make
cheap. Prefix `sp:o1:` rather than `sp:v1:`, carrying `oracle_id` instead of `illustration_id` —
a distinct prefix so `serve/render.py`'s `decode_custom_id` and the new one can never claim each
other's buttons. `oracle_id` is a UUID, so the id fits Discord's 100-character limit with the same
margin the existing one has.

---

## API

**Same app, same process, same port, same `scrying-api.service`.** No second service, no second
unit, no second entry in the machine inventory beyond the new database file.

| Endpoint | Shape |
| :-- | :-- |
| `POST /oracle/search` | `{query, k, filters: {types, colors, color_mode, mv_min, mv_max, legal}}` → `execute`'s dict passed through unchanged, plus a `service` block |
| `POST /oracle/feedback` | mirrors `/feedback`, keyed on `oracle_id`, `source='discord'`, idempotent by delete-then-insert |
| `GET /health` | gains an `oracle` block |
| `POST /admin/reload` | gains `{"index": "art" \| "oracle" \| "all"}`, defaulting to `"all"` |

Following the patterns `serve/api.py` already establishes:

- A second pydantic request model beside `SearchRequest`, with `Field` constraints and a
  `field_validator` for colours — the same validator logic, so `/scry` and `/oracle` cannot
  disagree about what `"wubrg"` means.
- A second injected callable on `Engine.__init__` alongside `search_fn` / `index_builder` —
  `oracle_search_fn` and `oracle_index_builder` — so `tests/serve_support.py` can stub them the
  same way. **Dependency injection, not import-time monkeypatching**, which is the convention the
  existing serving tests are built on.
- **One shared `asyncio.Lock` across both surfaces, not two.** They contend for the same Ollama
  instance and the same `judge_model` weights; two locks would only move the queue somewhere it
  cannot be reported. The queue cap stays at **4** — the binding arithmetic was 4 × 106.7s ≈ 7.1
  minutes against Discord's 15-minute deferred token, and that worst case is still all-`/scry`, so
  admitting faster searches into the same queue cannot make it worse.
- Search runs on `asyncio.to_thread` so `/health` answers during one, and the response carries the
  same shape of `service` block: `oracle_index_rebuilt`, `oracle_index_built_at`, `queued_seconds`,
  `refresh_running`, `degraded`.
- `pool` is returned in full, as today, because `curl … | jq '.pool'` is worth typing.
- The loopback bind guard (`check_bind_host`) is unchanged and already covers this.

`/health` gains:

```json
"oracle": {
  "cards": 32726, "chunks": 94812, "dim": 768,
  "build_seconds": 3.9, "built_at": "2026-08-17T03:44:11Z",
  "age_seconds": 8104, "missing_embeddings": 0, "stale": false,
  "last_oracle_refresh_at": "2026-08-17T03:43:02Z"
}
```

---

## Error handling and honest failure

The principle, unchanged: **always say something specific, and never manufacture a stretch.**

| Failure | What happens | What the user sees |
| :-- | :-- | :-- |
| **Nothing clears the 0.5 bar** | `select` fills with below-bar results, labelled | `no card confidently matches "let me draw". 214 cards passed the filters; the best 5 are all below the 0.5 bar and are shown as stretches.` The counts are the honest bound on what was actually looked at. |
| **Hard filters match zero cards** | SQL returns empty; nothing is relaxed | `no cards match type = enchantment · colors ⊆ {G} · mv ≤ 5. That combination has 0 cards in ~32,700 paper cards.` A precise question with no answer gets the answer "none", not a widened search nobody asked for. |
| **A soft filter would empty the pool** | Applied broadest-first, the offender dropped | `inferred filter mv ≤ 2 matched 0 cards; dropped — the retriever ranks on it instead.` Reported, never silent, exactly as `allowed_illustrations` does today. |
| **Vague quantity ("cheap")** | No numeric filter emitted | `"cheap" has no defined mana value, so no cost filter was applied. Add mv_max:3 if that is what you meant.` |
| **`*`/`X` power excluded** | `power_num` is NULL for them | `142 cards whose power is not a fixed number cannot be compared and were excluded.` |
| **Router unreachable** | No filters; semantic search over the whole corpus | `⚠️ query routing failed — searched all ~32,700 cards with no filters. Results may be off-type.` Plus the results. Degraded and labelled beats refused. |
| **Ollama down entirely** | No route, no expansion, no embedding, no judge. BM25 over chunks still works. | `⚠️ Ollama is unreachable — these are keyword-ranked only, nothing was judged.` Every result flagged stretch. Fast, unjudged, and honest. |
| **Judge batch fails twice** | `_fallback_entry`'s existing behaviour: retrieval order, `fit = None` | Those results carry `fit —` rather than a fabricated number. Never a fake score. |
| **Query is a card name** | Guard 1, before any model call | Pointed at `/search`, instantly |
| **Query is a rules question** | Guard 2, before any model call | Told plainly that this is not that tool |
| **Index rebuild fails** | Old index kept and served, traceback to journald, `oracle.stale: true` | Nothing in chat — a `/health` concern. Results are correct-but-possibly-missing-this-week's-cards, which is the pre-existing state. |
| **Both indexes stale simultaneously** | Independent; each retries on its own next search | `/health` reports each separately, because "the art index is stale" and "the oracle index is stale" are different problems with different causes |

Two things this design deliberately does **not** do:

- **It does not guess at vague quantities.** Covered above, and it is the same instinct as the
  README's refusal to emit a link it cannot build: *"A reference that cannot be built is omitted,
  never guessed and never emitted as a broken link."*
- **It does not hide the filter parse.** The echo line prints on success as well as failure. A
  mis-parse is only recognisable against the habit of reading correct ones.

---

## Testing

The existing suite is **255 tests, no Ollama, no network**, with `real_conn` skipping cleanly when
`data/commanders.db` is absent and `pytest.importorskip` keeping a minimal install green. Every
test below preserves all of that. The `data/` tree is gitignored, so no test may require
`data/oracle.db` to exist without skipping.

| File | Covers |
| :-- | :-- |
| **`tests/test_oracle_chunk.py`** | The heaviest new file, matching where the risk is. The chunker is a pure function of a card dict, so it is table-driven over real card text pasted into the module — the same convention `conftest.py`'s hand-picked `CORPUS` already uses. Cases: a vanilla creature (empty text → the whole-card chunk only); a two-ability enchantment (→ 2 + 1); a Saga; a DFC with `\n//\n`-joined faces (→ `face_index` 0 and 1); a card whose own name appears in its text (asserting `text_embedded` substitutes it and `text` does **not**); a keyword-only card (asserting reminder text is **kept**, because stripping it is the most likely well-intentioned regression); a card with `{T}:` and `+1/+1` (asserting no sentence-splitting happens); ordinals contiguous from 0. |
| **`tests/test_oracle_filters.py`** | The second-highest-risk piece, because it fails silently. The compiler is a pure function from a filter dict to `(sql, params)`. Every English phrase in the calibration table, fed through a **stubbed router response** (never a live call), asserting the resulting operator — including the `<=` versus `<` pairs, which is where an inversion would live. UNION within a field, AND across fields. Hard filters survive an empty result; soft filters drop at zero and only at zero, broadest-first, with a note naming the dropped one. The 0–30 absurd-value guard. `power_num` NULL exclusion and the excluded-count note. The echo string, character for character. The Scryfall URL round-tripping every filter it can express. |
| **`tests/test_oracle_render.py`** | Pure functions over canned result dicts, stdlib only, like `tests/test_render.py`. Full oracle text present verbatim in the description; the `▸` marker on the right line, derived from `ordinal`; the rationale positioned **after** the oracle text; **no `image` and no `thumbnail` key, ever** — that is a decision, so it gets a test, exactly as the "Popularity band, never Power level" string does today; the STRETCH label; the filter echo line; a 4,000-character oracle text truncating rather than 400-ing the message; an empty result set carrying the counts. |
| **`tests/test_oracle_judge.py`** | Batch-local chunk renumbering and citation validation, mirroring `tests/test_judge_props.py`: a citation belonging to another candidate is counted misattributed and dropped; one never shown is counted invented and dropped. **Plus a prompt-content test** asserting the mechanics rubric still names draw, loot, rummage, impulse, surveil, scry, reveal and tutor. A prompt edit that quietly deletes two of those lines is invisible in review and catastrophic in output. |
| **`tests/test_oracle_staleness.py`** | The second fingerprint, mirroring `tests/test_staleness.py`: inserting a chunk moves it, inserting an embedding moves it, changing nothing does not, `meta.last_oracle_refresh_at` moves it. A rebuild that raises leaves the old index in place and marks `stale`. One poll tick checks both fingerprints and rebuilding one does not rebuild the other. |
| **`tests/test_oracle_db.py`** | `init_schema` idempotence; all three pragmas set (mirroring `tests/test_db_pragmas.py`, including `busy_timeout`); and — the load-bearing one — **that opening and writing `oracle.db` opens no connection to `commanders.db`**, asserted by pointing both config paths at temp files and checking the art file is untouched. |
| **`tests/test_oracle_guards.py`** | The card-name guard and the rules-question guard, both pure functions over a string plus an in-memory `cards` table. Three lines each, and they are what stop a user paying 30 seconds for a wrong-command answer. |
| **`tests/test_api.py`** (extended) | `/oracle/search` happy path and validation; `mv_min > mv_max` is a 422; the **shared** lock genuinely serialises an `/oracle` behind a `/scry`; `/health` answers during an oracle search; `/admin/reload` with each of `art` / `oracle` / `all` rebuilds only what it names. |
| **`tests/test_bot.py`** (extended) | The `/oracle` custom-id prefix (`sp:o1:`) never decodes as `sp:v1:` and vice versa — a real collision risk with two button families in one channel. |

Expected total: **~300 tests, still zero network calls, still zero Ollama.**

**Not tested, deliberately:** real Ollama responses, real Scryfall downloads, discord.py's gateway,
and the systemd unit. Those are verified by one manual pass — a real `python -m cts oracle-ingest`,
a real `/oracle`, one 👍, `journalctl --user -u scrying-api` — and that pass is the acceptance
criterion for the deployment step.

**And one thing that is measured rather than tested:** the eval set in step 7 of *Order of work*.
Precision on near-miss mechanics is a property of prompts and embeddings, not of code, so unit
tests cannot assert it and pretending otherwise would be theatre. It needs 40 hand-labelled queries
and a number.

---

## Not building

Listed so nobody re-derives them as gaps:

- **Any artwork involvement of any kind.** No vision model, no art slots, no describe pass, no
  crops, no thumbnails in the embed, no image columns. Stated in the brief, restated here, and
  enforced by the schema.
- **A full Scryfall syntax parser.** No `-t:` negation, no nested parentheses, no `is:`, no `or`
  between fields, no regex `/…/`, no `mana:{G}{G}`. The escape hatch is the Scryfall link on every
  result set, which hands the user the real thing.
- **Negation and nesting in the filter algebra.** UNION within a field, AND across fields, and no
  more. Adding `NOT` means adding precedence, and precedence means a syntax to document.
- **Rules Q&A.** Not what this is. Guard 2 says so out loud rather than failing quietly.
- **Joining the two databases, or `ATTACH`ing one to the other.** Argued above. The combined
  art-plus-rules search, if it ever happens, is an application-level intersection.
- **Combo and synergy detection.** "Cards that go infinite with X" requires reasoning across cards.
  Everything here reads one card at a time.
- **EDHREC data, power scores, prices, or popularity bands in this corpus.** `edhrec_rank` from the
  bulk file is the only popularity signal, used only to order the structural-only fast path. No
  45-minute EDHREC poll is added to the refresh.
- **Card name lookup.** That is the planned `/search`, deliberately Scryfall-API-backed and
  deliberately not in this local corpus.
- **Digital-only cards.** `game:paper`, per the decision. Alchemy rebalances would double some card
  names with different text and make every result ambiguous.
- **Excluding silver-bordered and acorn cards by default.** They are paper Magic, the user asked for
  all paper Magic, and `legal:commander` removes nearly all of them for anyone who cares. A default
  exclusion would be this design quietly deciding what counts as Magic.
- **Rulings.** Scryfall publishes a `rulings` bulk file (5MB) and it is genuinely tempting as judge
  evidence. Out of scope: it is a second corpus with its own chunking, ingest and staleness
  questions, and the value is unproven until the eval in step 7 shows where the judge actually
  fails.
- **A second verification pass.** Argued above: the corpus is already ground truth, so a second
  read adds cost and no information.
- **MMR, colour caps, or any diversity beyond one-result-per-card.** Wrong for mechanical queries,
  and actively harmful on the user's own example.
- **Fine-tuning an embedding model on Magic templating.** Plausibly the largest available precision
  win and entirely out of scope. The `queries` / `retrievals` / `judgments` tables exist so the data
  for it accrues from day one.
- **A second systemd unit, a second port, a second timer, or a second process.** Same app, same
  service, three more refresh stages.
- **Result caching.** Same argument as the art side: a cache serves pre-refresh results after a
  refresh, trading a staleness bug for an optimisation nobody asked for.
- **Streaming progress.** The ~30s estimate carries the load.

---

## Order of work

1. **`cts/oracle_db.py` and `tests/test_oracle_db.py`.** Nothing can be checked until the shape is
   fixed, and the "never touches `commanders.db`" test is the guarantee everything else rests on.
2. **`cts/oracle_chunk.py` and `tests/test_oracle_chunk.py`.** Pure functions, no I/O, no network,
   and the highest-risk single decision in the design. Written before anything depends on it, so
   the chunking argument is settled against real card text rather than in the abstract.
3. **`cts/oracle_ingest.py`**, plus the one-parameter `bulk_entry(bulk_type)` edit to
   `cts/ingest.py`. Run it. **Look at the real card count and the real chunk count**, and put those
   in the README instead of this document's estimates. If the card count is not near 32,726, the
   exclusion list is wrong and this is where that surfaces.
4. **`cts/oracle_embed.py`, then `cts/oracle_index.py`.** Time the embed pass and the index build.
   The "tens of minutes" and "3–5s" estimates become measurements here.
5. **`cts/oracle_filters.py` and `tests/test_oracle_filters.py`.** Tests first — this is the piece
   that fails silently, and it is the piece three of the user's four clauses run through.
6. **`cts/oracle_search.py`** and the `python -m cts oracle "…"` CLI. Route, expand, retrieve,
   filter, judge, print — with the filter echo line — so the whole thing can be exercised against
   real Ollama and the real corpus before any serving code exists.
7. **Build the eval set and measure.** 40 hand-labelled mechanical queries, mirroring
   `eval/queries.jsonl`'s existing 15/15/10 split but weighted differently: ~15 unambiguous, ~15
   near-miss traps (draw vs loot vs impulse vs surveil; tokens vs copies; counter-spell vs
   counter-ability), ~10 vague-strategic. Measure precision@5 and the operator-parse accuracy
   separately. **This step comes before the Discord surface on purpose** — it is what converts this
   document's honest guesses into numbers, and if the near-miss precision lands near 55% rather
   than near 80%, the right response is to fix the judge prompt, not to ship it to people.
8. **`serve/oracle_render.py` and its tests**, then `/oracle/search`, `/oracle/feedback`, the second
   fingerprint, and the `/health` and `/admin/reload` extensions. Verified with `curl` before
   Discord exists.
9. **`serve/bot.py`:** the `/oracle` command, both guards, feedback buttons.
10. **Append the three oracle stages to `cts/refresh.py`.** Run one full refresh by hand and confirm
    the art stages are untouched and the oracle index picks up new data within a minute without a
    restart.
11. **README section** with the measured numbers, and a machine-inventory note in
    `~/.claude/CLAUDE.md`: one new database file (`data/oracle.db`, ~380MB), **no new port, no new
    unit, no new secret, and no new VRAM** — the last one being the fact anyone later wondering
    about GPU pressure should find there.
