# Scrying Pool as a service — local API, Discord bot, and the weekly refresh

2026-08-17. Status: **proposed — awaiting review**. Not approved, not implemented. No code written yet.

---

## What this is for

One person and a few friends want to run `/scry a hooded figure alone at dusk` from a
phone instead of sshing into the box. That is the entire requirement. Everything below
is sized to it: no authentication, no rate limiting, no abuse handling, no cost
recovery, no public exposure, no container, no web UI.

Discord is the interface for two reasons that are both about *not* opening a hole in the
house: the bot dials **out** over a websocket, so there is no port to forward, no tunnel
to run and no login page to write — and Discord already knows who everyone is, so the
identity problem is solved by not having it.

The search takes **76.8s on average and 106.7s at worst**. That is not hidden, worked
around, or optimised in this design. It is told to the user up front and then honoured.

---

## Architecture

Three systemd **user** units on the machine that already holds the corpus:

```
scrying-api.service     uvicorn, one worker, bound 127.0.0.1:8077 and nothing else.
                        Startup: load config, open SQLite, build the SearchIndex,
                        warm the models. Holds all three between requests.
                        One search in flight at a time, on a worker thread.

scrying-bot.service     discord.py. Wants=/After= the API. Holds no corpus state.
                        Restartable at any moment for free.

cts-refresh.timer       ALREADY WRITTEN (install-timer.sh), never enabled.
cts-refresh.service     Sun 03:00, Persistent=true, RandomizedDelaySec=1800.
                        UNCHANGED by this design. No hooks, no handshake.
```

The bot talks to the API over loopback HTTP. Nothing else talks to the API. The refresh
talks to nobody — the API watches the database instead.

### Why two processes and not one

Collapsing these into one process is the obvious simplification and it is wrong here.

The bot is the part that changes. Embed colours, button labels, how a stretch result is
worded, whether the thumbnail is a thumbnail or a full image — that is a dozen restarts
in an evening. The API is the part that is expensive to start: a **4–7s index build**
over 170,487 propositions and a **~17s cold model load** in Ollama. Tying the two
together means paying ~25s and a cold GPU to change a string.

Three more reasons, in descending order of how much they mattered:

- **Crash isolation.** discord.py reconnect logic, gateway resumes and a websocket that
  drops when the ISP hiccups have no business sharing an address space with a 523MB
  float32 matrix. If the bot wedges, `systemctl --user restart scrying-bot` costs
  nothing and the index survives.
- **`curl`-testability.** `curl -s localhost:8077/search -d '{"theme":"..."}' | jq` with
  Discord entirely out of the loop is the difference between debugging one system and
  debugging two coupled ones.
- **The next interface is free.** If a web UI ever happens (it is in *Not building*), it
  is a second client of an API that already exists, not a rewrite.

---

## Contention between serving and the weekly refresh

The API and `cts-refresh` share two resources: the GPU and the SQLite file. They are not
the same kind of problem, and conflating them produces exactly the wrong design.

**The GPU is a performance problem. The database is a correctness problem.** Only the
second one gets a mechanism.

### The GPU: ugly, bounded, and already arbitrated

Refresh's `describe` stage runs `vision_model` = `qwen3.5:122b` (81GB). The API holds
`judge_model` = `verify_model` = `qwen3.6:latest` (~27GB). The card has ~92.8GB usable,
so 81 + 27 = 108GB does not fit and the two cannot be co-resident.

**Ollama already handles this.** Its scheduler offloads a resident model when a request
arrives for one that will not otherwise fit — the "predicted to exceed available memory,
evicting" behaviour visible in this machine's own logs. A `/scry` issued while refresh is
describing does not fail, does not OOM, and does not corrupt anything. It ping-pongs:
load 27GB, judge a batch, evict for 81GB, describe an artwork, evict for 27GB, judge the
next batch. Both sides get slower — a search that normally takes ~80s takes minutes, and
the refresh takes longer too — and everything is correct the whole time. When the refresh
finishes, the next search is normal speed.

`config.toml` already documents the shape of this cost from the era when
`verify_model` pointed at the 81GB model: ~71s of a ~156s query, 46% of wall clock, spent
loading weights that had just been unloaded. That is what a search during the describe
stage looks like. It is bad. It is not broken.

**And the window is usually empty.** `describe.run` selects arts with `art_path` set and
no `descriptions` row, so a refresh only loads the vision model when genuinely new
`illustration_id`s arrived that week. New Magic artwork comes in bursts — set releases,
Secret Lairs, precon alt-arts — every few weeks, not every week. **Most weekly refreshes
describe nothing and never touch the vision model at all**; the whole run is HTTP and SQL
plus ~45 minutes of EDHREC polling at 1 req/s, with the GPU untouched.

So: no drain protocol, no pause endpoint, no `Conflicts=`, no lease. Building a mutual
exclusion mechanism whose window is usually zero-length, to prevent a slowdown that
resolves itself, would be exactly the over-engineering the rest of this design refuses.

What we do instead is **tell the truth about it**: `/health` reports whether a refresh is
currently running, and the bot puts that in the placeholder message so a search that
takes four minutes at 03:20 on a Sunday is explained rather than mysterious. Information,
not enforcement. Details in *The API* and *The Discord bot* below.

### Corpus staleness: the one that needs a mechanism

This is a genuine correctness bug and it is silent, which makes it the worst thing in the
system.

The API builds a `SearchIndex` **once**, at process start: one 170,487 × 768 float32
matrix and one BM25 index over the same rows in the same order. Refresh then inserts new
`cards`, new `arts`, new `descriptions`, new `props` and new `embeddings`. The in-memory
index is a snapshot from before all of that and has no idea. Left alone, the API serves
last week's corpus **indefinitely** — a commander from Sunday's set release is
unfindable, forever, with no error, no warning, and no degradation anyone can see. The
results still look plausible. That is precisely the failure mode this repo's README
spends a page refusing to ship.

Note what is *not* affected, because it narrows the problem usefully:
`search.power_bands()` recomputes the five quantile buckets from the live `power` table on
every single query, through the live connection. A refresh that moves EDHREC deck counts
and recomputes power scores is reflected in the very next search's **Popularity band**
with no index work at all. `candidate_rows()`, `slotvocab.load()` and the query/judgment
logging are likewise all live reads and writes. **Only the vector and BM25 index can go
stale** — but it is the part that decides what gets retrieved at all.

#### The fingerprint

Before every search, inside the lock, the API reads a **corpus fingerprint**: one row,
three index seeks, sub-millisecond.

```sql
SELECT (SELECT value       FROM meta       WHERE key = 'last_refresh_at'),
       (SELECT MAX(id)     FROM props),
       (SELECT MAX(prop_id) FROM embeddings);
```

If it differs from the fingerprint recorded when the current index was built, the index is
rebuilt before the search runs.

Why those three, specifically:

- **`meta.last_refresh_at`** is already stamped by `refresh.run` at the end of every run.
  It catches a refresh that added no propositions but did move EDHREC data and power
  scores — nothing the index needs, but it is the cheapest available "something happened"
  marker and it makes the check honest about runs that changed the corpus in ways the
  other two fields cannot see.
- **`MAX(props.id)`** catches new vision output. `props.id` is `INTEGER PRIMARY KEY`, so
  this is a single seek to the end of the table, not a scan. A `COUNT(*)` over 170k rows
  would be a full scan and is not used.
- **`MAX(embeddings.prop_id)`** catches the embed stage finishing behind the describe
  stage. `load_index` only carries props that have vectors, so an index built between the
  two stages is legitimately incomplete and must rebuild again once embed catches up.
  `load_index` already counts and reports `missing_embeddings`; that number is surfaced in
  `/health` and, when non-zero, appended to the search response's notes.

**Where the fingerprint is blind, stated plainly.** An in-place `UPDATE props SET text=…`
that changed no ids would not be detected — nothing in the pipeline does that, since
`describe` writes new rows for new `illustration_id`s and `--backfill-stale` replaces rows
and takes fresh ids. Clearing the `embeddings` table and re-embedding from scratch returns
`MAX(prop_id)` to its previous value and would also be missed. Both are things a human
does deliberately, and for those there is `POST /admin/reload`.

#### Where the check runs, and why

Two placements, both self-checks, no coupling to the refresh:

1. **Per search, synchronously, as the guarantee.** Every search is preceded by the
   fingerprint check. This is what makes correctness unconditional: it holds for the
   weekly timer, for a manual `python -m cts refresh` in a terminal, for a bare
   `python -m cts ingest`, for a database restored from backup, and for an API that was
   restarted at an awkward moment. Nothing has to remember to tell it.
2. **On a 60-second background timer, as latency hiding.** The same check, run by a
   background task that takes the same lock. If a search holds the lock, the tick is
   skipped and tried again in a minute. In practice this means the rebuild has almost
   always already happened by the time anyone searches, and the synchronous path in (1) is
   a safety net that rarely fires.

The background timer is **debounced to at most one rebuild per 5 minutes**, and it is
**suppressed entirely while `cts-refresh.service` is active**. Without that suppression, a
20-minute embed stage committing batches would move `MAX(embeddings.prop_id)` every tick
and trigger twenty consecutive 5-second rebuilds of an index nobody is querying — burning
CPU and doubling RSS repeatedly against a corpus that is still changing. This is the one
place the refresh-is-running signal is used for anything other than a message, and it is
still not a lock: the synchronous per-search check is **never** suppressed, so a search
issued mid-refresh still gets a current index.

#### Does a request block for the rebuild, or serve stale-but-labelled?

**It blocks.** The rebuild is 4–7 seconds against a search whose placeholder already
promised ~80s — under 9% of a wait the user has already been told to expect, and it
happens once, to the first search after a refresh. The second search is normal.

Serving stale-but-labelled results was considered and rejected. A banner saying "these
results may be missing commanders added in the last week" gives the user nothing they can
act on: they cannot make the index rebuild, they cannot tell whether the thing they were
looking for is one of the missing ones, and the correct response to the banner is to run
the search again — which costs another 80 seconds to save 5. It converts a 5-second delay
into a worse product and a support question. The repo's existing habit is to do the
correct expensive thing and say so (`refresh` re-fetches all of EDHREC weekly; `index`
rebuilds from scratch on every process start because it is cheaper than incremental
state), and this follows it.

The response carries `service.index_rebuilt: true` so the extra seconds are attributable
rather than mysterious.

#### Rejected: refresh signals the API when it finishes

The obvious alternative is an `ExecStopPost=-curl -XPOST localhost:8077/admin/reload` on
`cts-refresh.service` — cheap, and it fires on success, failure and kill alike.

Rejected, because it is strictly worse than the check that has to exist anyway:

- It does not cover a manual `python -m cts refresh`, an `ingest` run, or anything else
  that touches the database outside the unit. So the fingerprint check is still required,
  and the signal becomes a second mechanism for the same job.
- It adds a coupling in the wrong direction. Today `install-timer.sh` writes a refresh
  unit that works on any machine, with or without the serving layer installed. A hook
  pointing at port 8077 means the refresh unit now has an opinion about the API, and the
  drop-in has to be installed, maintained, and remembered when either side changes.
- Its failure mode is silence. If the curl is dropped, mistyped, or the API is restarting
  at that moment, nothing reports it and the corpus goes stale exactly as if the mechanism
  did not exist. The background poll has no such state — it re-derives the answer every
  minute from the database itself.

The 60-second poll buys everything the signal would have bought, with one fewer moving
part and no cross-unit dependency. **`cts-refresh.service` is left exactly as
`install-timer.sh` writes it today.**

#### Failure modes of the chosen design

- **A rebuild mid-refresh sees a half-updated corpus.** New artworks whose props are
  written but not yet embedded are simply absent from the index — `load_index` joins
  `props` to `embeddings` — and `missing_embeddings` reports how many. The next rebuild
  picks them up. Strictly more corpus than before, never wrong, and self-correcting.
- **Peak memory doubles for the 4–7 seconds of a rebuild.** The new index is constructed
  before the old one is dropped, so both matrices (523MB each, plus the `rank_bm25`
  structures over the same rows) are live at once. This is the right trade — the
  alternative is dropping first and having no index at all if the rebuild throws — but it
  is the number to watch, and it is why `MemoryMax=` is deliberately *not* set on the
  unit. An OOM kill mid-rebuild would be far worse than the spike it prevented.
- **If a rebuild raises**, the old index is kept and served, the traceback goes to
  journald, `/health` reports `index.stale: true`, and the next search tries again. The
  API is never left without an index. The realistic cause is the database being mid-write
  in a way that trips a lock, which the next attempt will not hit.
- **The 60-second poll can rebuild an index nobody then uses**, wasting ~5 seconds of CPU
  after a quiet ingest. Acceptable, and the debounce bounds it.
- **A search that began before a refresh committed can straddle it** — see the WAL section
  immediately below.

### What WAL does and does not give us

`db.connect()` already sets `PRAGMA journal_mode=WAL`, which buys the important half:
refresh's writes never block the API's reads and vice versa. Because this design lets
searches run *during* a refresh rather than excluding them, the details matter more here
than they would under a lock.

- **The API sees committed writes without reconnecting.** Python's `sqlite3` autocommits a
  bare `SELECT`, so each statement takes its own WAL snapshot rather than the connection
  pinning one at boot forever. That is what makes a fingerprint check on a
  connection opened days ago work at all.
- **A single search's statements can straddle a refresh commit.** `candidate_rows()` may
  return an artwork the in-memory index has never heard of, or fail to return one it has.
  Both are benign and both are already handled: `collapse()` explicitly skips rows that
  vanished ("art row vanished between index build and query"), and an artwork with no
  props in the index cannot be retrieved in the first place. The result is a search that
  saw a very slightly odd corpus, not a wrong one.
- **Writers still serialise, and the API is a writer.** `search.execute` writes `queries`
  and `retrievals`, `judge.log_judgments` writes `judgments`, `/feedback` writes too.
  Against refresh's long embed transactions, Python's default 5-second busy timeout is not
  enough, and a `database is locked` raised out of `_log_query` fails the **entire** search
  after all the model work has already been paid for. **`cts/db.py::connect()` sets
  `PRAGMA busy_timeout=30000` alongside the WAL and foreign-keys pragmas it already sets.**
  It is the single most important line in this section: without it, "searches during a
  refresh are slow" quietly becomes "searches during a refresh throw".

  This was originally specified as a serving-side pragma applied after `db.connect()`, on
  the principle that `cts/` stays untouched. That was wrong, and it is the one place this
  design edits `cts/`. The bug is not serving-specific: *any* two `cts` processes that
  write concurrently hit it — a `python -m cts search` in a terminal while the weekly
  refresh embeds, `evaluate` against a running `ingest`. Fixing it one caller at a time
  leaves the same landmine for every other caller and makes the correct pragma set
  something each new connection site has to remember. It belongs where the other two
  pragmas already live. The serving connection in `serve/api.py` opens its own
  `sqlite3.connect` (for `check_same_thread=False`) and therefore sets all three pragmas
  itself, mirroring `db.connect()`.

### Other accepted risks

- **A search issued during the describe stage may exceed the bot's read timeout.** The bot
  extends its timeout when `/health` reports a refresh running (see below), but a
  pathologically thrashing search can still outlast it. The results are not lost: they are
  already durably in `queries`, `retrievals` and `judgments`, recoverable by `query_id`,
  which is logged.
- **An in-flight search cannot be cancelled.** `search.execute` is synchronous blocking
  code on a worker thread; Python cannot interrupt it. Nothing in this design needs to,
  which is one more small benefit of dropping the drain protocol.
- **The eval-only Ollama at `127.0.0.1:11435` also holds VRAM.** Per the machine
  inventory, `nemotron-3.5-lightning:30b` lives there. If that instance has a model
  resident, the 92.8GB budget above is wrong by ~20GB and Ollama's eviction decisions
  change accordingly. This design does not manage it; `/health` reports what Ollama says
  is loaded, so the cause is at least visible.

---

## The API

### Binding and address guard

`127.0.0.1:8077`. Not configurable to anything else in v1, and this is enforced in code,
not by a default: the entry point reads `SCRYING_API_ADDR`, and **refuses to start** if
the host is anything other than `127.0.0.1` or `::1`, with an error naming why.

The reason is on this machine specifically. Ollama here is bound `0.0.0.0:11434` and is
flagged as a known risk in the machine inventory. A second network-exposed service that
proxies straight into that one, with no auth, is not a thing this project adds. If remote
access is ever wanted, Tailscale is already installed and is the answer.

### `POST /search`

Request:

| Field | Type | Default | Notes |
| :-- | :-- | :-- | :-- |
| `theme` | string, 1–300 chars | required | passed verbatim to `execute` |
| `k` | int 1–5 | 5 | capped at 5 so results fit one Discord message |
| `band` | int 1–5 or null | null | Popularity band |
| `colors` | string or null | null | subset of `WUBRG`, validated case-insensitively |

Handler: `cts.search.execute(cfg, theme, band=…, colors=…, k=…, kind="user", conn=<the
long-lived connection>, index=<the process-wide index>)`. `execute` already takes `conn`
and `index` precisely so a long-lived caller pays the build once; that is the whole
integration.

`kind="user"` and not `"discord"`. `export_training.py` filters on
`kind IN ('user','eval')`, so real Discord searches land in the training exports where
they belong. The Discord-ness of a search is recorded on the *feedback* row instead.

Response 200 is `execute`'s dict **passed through unchanged** — `query_id`, `plan`,
`relaxed`, `results`, `pool` — with one added key:

```json
"service": {
  "index_rebuilt": false,
  "index_built_at": "2026-08-17T03:41:09Z",
  "queued_seconds": 0.0,
  "refresh_running": false,
  "degraded": false
}
```

There is **no reshaping layer**. Every field the bot renders is already on each result:
`oracle_id`, `name`, `mana_cost`, `type_line`, `color_identity`, `band`, `fit`,
`rationale`, `verified`, `illustration_id`, `set_code`, `artist`, `prop_ids`, `links`
(built once in `cts/links.py`), `stretch`, `vision_rejected`, `verify_note`. Inventing a
DTO in front of that would be a second definition of the result contract and a second
place to forget a field.

`pool` (up to 100 judged candidates) is kept in the response even though the bot ignores
it. It costs a couple of hundred KB over loopback and it is what makes
`curl … | jq '.pool'` worth typing.

`degraded` is true when `plan.notes` is non-empty or `plan.vision_verified` is false — the
signal the bot uses to print a warning banner.

Errors:

| Status | Body | When |
| :-- | :-- | :-- |
| 422 | FastAPI's validation shape | bad `k`, bad `colors`, empty `theme` |
| 503 | `{"status":"busy","queued":4,"max_queued":4}` | queue full |
| 500 | `{"status":"error","detail":"…"}` | `execute` raised |

There is deliberately **no 503 for "refreshing"**. A refresh does not make searches
impossible, only slow, and refusing them would be inventing an outage.

### `POST /feedback`

Request: `{"query_id": int, "illustration_id": str, "accepted": bool, "discord_user_id":
str|null}`.

Writes one row into `judgments` in the same shape as `cts/evaluate.py::_write_mark`:
`fit` = 1.0 or 0.0, `model` = `""` (a human is not a model), `prop_ids` = the JSON list
carried on the result, `rationale` = prose naming what happened — and **`source =
'discord'`**, the one field that distinguishes these from `eval`'s `source='human'` marks.

Two deliberate divergences from `_write_mark`:

- **The vote is idempotent.** `judgments` has no unique constraint, so a double-tapped 👍
  would insert twice and a 👍-then-👎 would leave two contradictory training rows. The
  handler deletes any existing `(query_id, illustration_id, source='discord')` row before
  inserting, so the latest vote wins and changing your mind is not a data bug.
- **`query_id` is validated** against the `queries` table; an unknown id is a 404 rather
  than an orphan row. The buttons are persistent across bot restarts, so a stale button
  from three weeks ago is a normal event, not an anomaly.

The Discord user id has nowhere structured to go — the schema is SPEC.md verbatim and this
design does not add a column — so it is folded into the `rationale` text (`"discord user
1234… marked this result acceptable"`). Good enough to audit, not pretending to be
structured.

The write uses a **short-lived connection of its own**, not the serving connection, so a
👍 tapped 40 seconds into someone else's search does not sit behind the search lock.
`busy_timeout=30000` applies to it too.

Response: `{"ok": true, "replaced": false}`.

### `GET /health`

Everything needed to answer "why is it doing that", in one object:

```json
{
  "status": "ok",
  "uptime_seconds": 84213,
  "index":  {"props": 170487, "artworks": 5530, "dim": 768,
             "build_seconds": 5.1, "built_at": "2026-08-17T03:41:09Z",
             "age_seconds": 8104, "missing_embeddings": 0, "stale": false},
  "corpus": {"commanders": 3202, "last_refresh_at": "2026-08-17T03:39:55Z"},
  "refresh": {"running": false, "unit": "cts-refresh.service"},
  "ollama": {"url": "http://localhost:11434", "reachable": true,
             "missing_models": [], "loaded": ["qwen3.6:latest"]},
  "search": {"in_flight": 0, "queued": 0, "max_queued": 4,
             "last_search_seconds": 74.2, "searches_since_start": 61}
}
```

- `status` is `ok`, `refreshing`, or `degraded` (Ollama unreachable or models missing).
- `index.age_seconds` and `corpus.last_refresh_at` together answer "is this index current"
  at a glance; `index.stale` is true when the fingerprint has moved but a rebuild has not
  yet succeeded.
- `refresh.running` comes from `systemctl --user is-active cts-refresh.service`, cached
  for 10 seconds. If `systemctl` is unavailable the field is `null` rather than `false` —
  unknown and known-not-running are different answers.
- The Ollama probe reuses `cts.ollama.preflight` and is also cached for 10 seconds, so the
  bot polling `/health` before every search cannot become a hot loop.

**`/health` must answer during a search.** That is the whole reason the search runs on a
worker thread rather than on the event loop, and it gets a test of its own.

### `POST /admin/reload`

Force an index rebuild. Loopback-only like everything else; no token, because binding to
loopback on a single-user machine is already the boundary and a token in a curl line would
be a secret in a place this project has decided secrets do not go.

This exists for the human who just did something the fingerprint cannot see — cleared the
`embeddings` table, restored a database, edited props by hand. It is the escape hatch for
the blind spots named above, and it is the only admin endpoint. There is no pause, no
drain, and no lease.

### Concurrency and the single search lock

**One search at a time**, via an `asyncio.Lock`. This is not a throttle bolted on for
safety; it is an acknowledgement that the GPU serialises the work regardless. Two
concurrent searches would interleave `judge_model` calls into the same Ollama instance and
both finish later than if they had queued, while doubling peak memory on the Python side.

Mechanics, concretely:

- `execute` is synchronous blocking code. It runs via `asyncio.to_thread`, never on the
  event loop, so `/health`, `/feedback` and `/admin/reload` stay responsive through a 100s
  search.
- The threadpool may hand the work to a different thread each time, so the serving
  connection is opened with `check_same_thread=False`. That is safe **only** because the
  lock guarantees exactly one user of it at a time — which is also why feedback uses its
  own connection rather than borrowing this one, and why the background staleness poll
  takes the same lock.
- **Queue cap: 4** in-flight-plus-waiting. The fifth request gets a 503 `busy`. The
  arithmetic: 4 × 106.7s ≈ 7.1 minutes, comfortably inside Discord's 15-minute deferred
  token with room for the API to have been mid-search when the first arrived. A deeper
  queue would resolve into expired tokens.

**Queue-position messaging.** The API cannot report a position in the search response — by
the time the response exists, the wait is over. So the bot calls `GET /health` first (it
is cheap and cached) and writes the queue depth into the placeholder it is about to post:
`queued behind 2 searches · ~4 min`. There is a small race — someone may slip in between
the `/health` and the `/search` — and the consequence is a slightly optimistic estimate,
which is acceptable. The placeholder is corrected if `service.queued_seconds` in the
response disagrees materially.

---

## The Discord bot

### The command

`/scry theme:<text> [k:1-5] [band:1-5] [colors:<WUBRG>]`

Registered as a **guild** command against the single server id in `bot.env`, so a redeploy
is live instantly instead of waiting out global command propagation.

**No privileged intents.** Slash commands do not require Message Content, so the bot runs
with default intents and the Discord developer portal needs no special toggles. Worth
stating because it is the single most common setup snag.

### The wait

1. `interaction.response.defer(thinking=True)` **first, before anything else** — Discord
   kills an interaction that is not acknowledged within 3 seconds, and `/health` plus
   `/search` will take much longer than that. The deferred token is then valid for 15
   minutes against a 106.7s worst case. Comfortable.
2. `GET /health`, then edit the deferred response to a placeholder built from what it said:
   - normal: `🔮 scrying… ~80s`
   - queued: `🔮 scrying… queued behind 1 search, ~3 min`
   - refresh running: `🔮 scrying… ⚠️ the weekly corpus refresh is running, so this will
     be slow — several minutes rather than ~80s.`
   That last line is the *entire* user-facing consequence of GPU contention, and it is
   information, not a refusal.
3. `POST /search`. Read timeout **300s** normally; **780s** when `/health` reported a
   refresh running. 13 minutes sits inside the 15-minute token with margin, and it is the
   real bound on how long a thrashing search is allowed to be waited on.
4. Edit the same message in place with the results. One message, edited twice. No
   follow-up spam.

### The result message

The edited message carries a content line and up to five embeds.

The content line mirrors the CLI's own honesty, because the CLI already got this right:

```
🔮 "commanders that look lonely" · 30% literal / 70% interpretive · band 3
3 of 5 results clear the 0.5 fit bar; the rest are stretches.
```

One embed per result:

| Element | Source |
| :-- | :-- |
| Title | `name` + ` ` + `mana_cost`, then `· STRETCH` when `stretch` is true |
| Colour | green when `verified`, blue when passing but unverified, grey when `stretch` |
| Field: Colours | `color_identity` as WUBRG letters, `C` when empty |
| Field: **Popularity band** | `Popularity band 3/5` |
| Field: Fit | `fit` to two decimals, always shown — this is a chat client, not a terminal, and the number is small |
| Description | `rationale`, the judge's one-line justification |
| Thumbnail | `links.art_crop` |
| Footer | `set_code · artist`, plus `1 of 4 arts` when `art_count > 1` |
| Links | one markdown line: `[EDHREC](…) · [theme](…) · [Scryfall](…) · [TCGplayer](…)`, each omitted entirely when its key is absent |

**"Popularity band", never "Power level."** The composite is 0.4 × deck count + 0.25 ×
price + 0.2 × cmc + 0.15 × a cEDH flag that saturates across nearly the whole corpus.
`config.toml` says in its own comments that deck count is "popularity, not power", and the
README lists the saturating flag under known limits. The label in the interface says what
the number measures. This is a decision, not a wording preference, and it gets a test.

**Stretches are labelled, never silently mixed in.** `select()` already appends below-bar
results to fill out `k`, and `_format_result` already prints `STRETCH (below the 0.5
bar)`. The bot carries that through: stretch results are ordered last, coloured
differently, titled `· STRETCH`, and counted in the content line. A user who asked for
five and got two real matches must see that at a glance. `vision_rejected` results never
reach the bot — `select()` excludes them — but `verify_note`, when present, is appended to
the description so "no local art crop to verify against" is visible rather than
mysterious.

When `service.degraded` is true, a warning line goes above the embeds carrying
`plan.notes` verbatim: `⚠️ vision verification unavailable — results are judge-ordered and
unverified`. Same text the CLI prints. Two renderings of one fact, one source.

When `results` is empty, the message says so with the counts from `plan.counts`, matching
the CLI: `no matches. 412 commanders retrieved, 0 survived the filters.`

### Feedback buttons

Ten buttons in two action rows: 👍1–👍5 and 👎1–👎5, matched to the embed positions. Each
carries `custom_id = sp:v1:<query_id>:<illustration_id>:<u|d>` — roughly 60 characters
against Discord's 100-character limit, with a UUID in the middle.

They are **persistent** components (`timeout=None`, registered at startup via discord.py's
dynamic-item matching), which matters more here than it would elsewhere: the entire
rationale for the two-process split is that the bot restarts constantly, and a
conventional `View` dies with its process, leaving every button in the channel silently
dead. Encoding the identity in `custom_id` means a button tapped after a restart — or
three weeks later — still resolves, and the API's `query_id` validation covers the case
where it should not.

On tap: `POST /feedback`, then an ephemeral confirmation (`recorded 👍 for Avacyn, Angel
of Hope`) so the channel does not fill with acknowledgements.

This is not a nicety. `judgments` with `source='discord'` is exactly the human-marked data
the eval harness collects interactively and mostly never gets, arriving here as a side
effect of people using the thing. `export_training.py` already reads that table.

### Discord limits that shape the design

- 3s to acknowledge; 15 min deferred token. Drives defer-first, the queue cap, and the
  780s ceiling on the refresh-window timeout.
- 10 embeds and 6,000 total characters per message. `k ≤ 5` keeps this uncrowded.
- 5 buttons per action row, 5 rows. Ten buttons is two rows.
- 4,096 characters per embed description. A `rationale` is one sentence, but the renderer
  truncates at 1,000 with an ellipsis anyway rather than letting a pathological judge
  response 400 the whole message.

---

## Error handling

The design principle: **the bot always says something specific.** A spinner that never
resolves is the worst outcome available, and every branch below exists to avoid it.

| Failure | What actually happens | What the user sees |
| :-- | :-- | :-- |
| **API not running** | `httpx.ConnectError` on the first call | ``The search service isn't running on the host. Someone needs to check `systemctl --user status scrying-api`.`` |
| **Refresh running** | Nothing fails. Searches are slow. | The placeholder says so up front, and the results arrive normally when they arrive. |
| **Queue full** | 503 `{"status":"busy"}` | `Four searches are already queued (~7 min of work). Try again shortly.` |
| **Ollama down** | Nothing raises. `route()` falls back to 0.5/0.5, both expansions fall back to the raw query, `_query_vectors` returns None so ranking is BM25-only, `judge_batch` returns fallback entries with no `fit`, `verify_finalists` reports unavailable. A full result set comes back, fast, and entirely unjudged — every result flagged `stretch`. | The results, under a prominent banner: `⚠️ Ollama is unreachable — these are keyword-ranked only, nothing was judged or verified.` Built from `plan.notes`, which already says exactly this. Deliberately **not** an error: degraded output plus an honest label beats a refusal. |
| **Index rebuild fails** | Old index kept and served, traceback to journald, `/health` reports `index.stale: true`, next search retries | Nothing, unless it persists — this is a `/health` concern, not a chat concern. Results are correct-but-possibly-missing-this-week's-cards, which is the pre-existing state, not a new failure. |
| **Search raises** | 500 with the exception string; traceback to journald | ``The search failed: <detail>. This is logged — check `journalctl --user -u scrying-api`.`` |
| **Bot timeout (300s / 780s)** | httpx read timeout; the search may still be running server-side | `Gave up waiting after 5 minutes. The search may still be running — check /health.` |
| **Discord edit fails after a long search** | Token expired, message deleted, or a 5xx from Discord | Retry the edit once, then fall back to a fresh channel message mentioning the user. **In every case log `query_id` at WARNING.** The search is not lost: results are already durably in `queries`, `retrievals` and `judgments`, recoverable with one SQL query. A real benefit of logging first and rendering second, worth saying out loud. |
| **Feedback write fails** | 404 (unknown `query_id`) or a lock timeout | Ephemeral `Couldn't record that.` — never a silent no-op, and never a failure that touches the results message. |

The API **does not** preflight Ollama on the search path. It would add a round trip to
every search to prevent a failure mode that already degrades gracefully. `/health`
preflights; `/search` reports what happened.

---

## The weekly refresh, wired in

`python -m cts refresh` already does the entire job — ingest (skipping an unchanged
Scryfall bulk), EDHREC across the whole corpus, power recompute, describe for new
`illustration_id`s only, embed — and `install-timer.sh` already installs it as a weekly
user timer with `Persistent=true` and oneshot overlap protection. **This is wiring, not
new code, and after the staleness decision above it is barely even wiring.**

Two things to do, neither of which touches the refresh itself:

1. **Enable the timer.** It has never been run. `./install-timer.sh`, then
   `loginctl enable-linger "$USER"` so it fires without a login session, then one manual
   `systemctl --user start cts-refresh.service` to confirm before trusting the schedule —
   all of which the script's own closing output already tells you to do.
2. **Say two things in the README.** That searches during a refresh are slow and why, and
   that the API notices new corpus data by itself within about a minute, so nothing needs
   restarting after a refresh.

`cts-refresh.service` gets **no drop-in, no `ExecStartPre`, no `ExecStopPost`, and no
`Conflicts=`.** It runs exactly as it does today, on a machine with the serving layer or
without it.

One note on `Persistent=true` in this context: a machine that was off on Sunday runs the
refresh shortly after boot on Monday, which is a *daytime* window and a much more visible
slowdown than 03:00. That is correct behaviour, and the placeholder line about the refresh
running is what makes it comprehensible instead of alarming.

---

## Secrets

`~/.config/scrying-pool/bot.env`, mode 600, owned by the user, loaded by
`EnvironmentFile=`:

```
DISCORD_TOKEN=…
DISCORD_GUILD_ID=…
SCRYING_API_URL=http://127.0.0.1:8077
```

Never in the repo, never in a unit file, never in a shell script, never a default in
Python. `.gitignore` gains `*.env` as a second layer.

This is not boilerplate caution. The `cloudflare-ddns` service on this same machine has a
Cloudflare API token as a literal in `main.py`, it is flagged as a standing risk in the
machine inventory, and it is exactly the shape of mistake being avoided here. `bot.env` is
also outside the repo entirely, so a token can never ride along in a `git add -A`.

Note that `EnvironmentFile=` is not a shell: no `export`, no comments after values, no
quoting subtleties. Plain `KEY=value` lines.

---

## Files, packages and dependencies

```
serve/
  __init__.py
  api.py          FastAPI app, lifespan, the lock, the fingerprint check, the poll
  bot.py          discord.py client, the /scry command, the persistent button view
  render.py       result dict -> Discord embed JSON. Pure functions, no discord import.
  install-services.sh   writes both units; --dry-run supported
serve-requirements.txt
```

`serve/render.py` renders to **plain dicts matching Discord's embed JSON**, not to
`discord.Embed` objects, and `bot.py` calls `Embed.from_dict`. That keeps the rendering
layer testable with nothing installed but pytest, which is a property the existing suite
has and must not lose.

`serve-requirements.txt`: `fastapi`, `uvicorn[standard]`, `discord.py`, `httpx`.
**`requirements.txt` stays at its three lines** — `requests`, `numpy`, `rank_bm25`. The
README's "no cloud, no vector database, no framework" claim is about the search engine and
it stays true: the framework is in the serving layer, which is optional, separate, and
named as such.

`cts/` is modified in exactly **one place**: `db.py::connect()` gains
`PRAGMA busy_timeout=30000` next to the WAL and foreign-keys pragmas it already sets, for
the reasons argued in *What WAL does and does not give us* above. Nothing else in `cts/`
changes, and no `cts` behaviour changes for any existing caller beyond waiting for a lock
instead of immediately raising `database is locked`. Everything else the API needs is
already public: `config.load_config`, `db.connect`, `db.init_schema`, `index.load_index`,
`search.execute`, `ollama.preflight`.

`serve/install-services.sh` mirrors `install-timer.sh`'s existing style deliberately —
derive the repo root from the script's own location, prefer `.venv/bin/python`, bake
absolute paths into `ExecStart` because a user unit inherits almost no environment,
support `--dry-run`, and print the follow-up commands at the end. Someone who has read one
of these scripts can read the other.

---

## The two systemd units

Described, not written; `install-services.sh` emits them.

### `scrying-api.service`

- `[Unit]` — description; `After=network.target`. Notably **no dependency on Ollama**: it
  is a *system* unit and this is a *user* unit, so the ordering is not expressible, and
  more importantly the API must tolerate Ollama being down at boot. It builds its index
  and serves `/health` with `status: degraded` rather than exiting, because a
  preflight-and-die would crash-loop through every reboot where Ollama is slow to come up.
- `[Service]` — `Type=simple`; `WorkingDirectory=` the repo (config.toml and the relative
  `data/` paths resolve from there); `ExecStart=<repo>/.venv/bin/python -m serve.api` with
  the absolute interpreter path; `Environment=PYTHONUNBUFFERED=1` so `index.py`'s stderr
  diagnostics reach journald as they happen rather than at exit; `Restart=on-failure` with
  `RestartSec=5`.
- `Restart=on-failure`, not `always`, so a deliberate `systemctl --user stop` stays
  stopped.
- No `EnvironmentFile=` — the API holds no secrets.
- No `MemoryMax=`, deliberately: an OOM kill during a 523MB index rebuild is worse than the
  spike it would prevent.
- Minimal hardening. `PrivateTmp=true` is safe and included; `ProtectSystem=strict` and
  friends are **not**, because the service legitimately writes `data/commanders.db` and the
  WAL beside it, and a sandbox that silently breaks the write path buys nothing on a
  single-user desktop.
- `[Install] WantedBy=default.target`.

### `scrying-bot.service`

- `[Unit]` — `Wants=scrying-api.service` and `After=scrying-api.service`. **`Wants` and not
  `Requires`**: if the API fails, the bot must stay up to *say* the API is down. `Requires`
  would stop the bot along with it and turn a legible error message into silence.
- `[Service]` — `Type=simple`; `EnvironmentFile=%h/.config/scrying-pool/bot.env`;
  `ExecStart=<repo>/.venv/bin/python -m serve.bot`; `Restart=always` with `RestartSec=10`,
  because a dropped gateway connection is normal and should self-heal.
- `StartLimitIntervalSec=300`, `StartLimitBurst=5` — a bad or revoked token would otherwise
  crash-loop against Discord's login endpoint forever. Five tries in five minutes, then
  stop and stay stopped, which is both polite and diagnosable.
- `[Install] WantedBy=default.target`.

Both units need `loginctl enable-linger "$USER"` to survive logout — the same requirement
`install-timer.sh` already documents for the refresh timer, and the same trap.

### Machine inventory

`~/.claude/CLAUDE.md` carries a standing rule: any new service, container or daemon on this
machine gets added to the inventory, from whatever project directory it was set up in. This
design adds **three** user units (`scrying-api`, `scrying-bot`, and the `cts-refresh` timer
that was written but never enabled), one new listening port (**8077**, loopback only), and
one new secret location (`~/.config/scrying-pool/bot.env`).

The entry should also record the VRAM interaction, because it is the non-obvious
operational fact: **the serving API holds ~27GB of VRAM continuously; the weekly refresh's
describe stage needs 81GB; Ollama evicts between them, so searches during a refresh are
slow by design and nothing is broken.** Anyone later wondering why a Sunday-morning search
took four minutes should find the answer there. Worth noting alongside it that the
eval-only Ollama on 11435 competes for the same card.

Updating the inventory is part of the deployment step, not an afterthought.

---

## Testing

The existing suite is **62 tests, no Ollama, no network, no fixtures that require the
prebuilt database** (the one that does, `real_conn`, skips cleanly when it is absent). That
property is the reason the suite is worth running, and every test added here keeps it.

- **`tests/test_api.py`** — FastAPI `TestClient` against the app with the search callable
  **injected**, not monkeypatched at import time: `api.py` holds its engine in an object
  created by the lifespan, and the tests construct one with a stub `execute` returning a
  canned dict. Covers the happy-path shape, `k`/`colors`/`band` validation, 503 when the
  queue is full, that two concurrent searches serialise, and that `/health` answers *while*
  a stub search is sleeping — the test that actually proves the threadpool decision.
  Begins with `pytest.importorskip("fastapi")`, so a checkout with only
  `dev-requirements.txt` installed still runs the original 62 green rather than erroring on
  collection.
- **`tests/test_render.py`** — pure functions over canned result dicts, nothing imported
  beyond the stdlib. A passing verified result, a stretch, a result with `links` missing
  every optional key, one with `art_count > 1`, an empty result set, and one with a
  4,000-character rationale. Asserts the STRETCH label is present, and that the band field
  reads `Popularity band n/5` and **never contains the word "power"** — that string is a
  decision, so it gets a test.
- **`tests/test_staleness.py`** — the heaviest new file, matching where the risk is. The
  fingerprint is a pure function of a connection: build a small database, snapshot, insert
  props, assert it moved; insert an embedding, assert it moved again; change nothing,
  assert it did not; set `meta.last_refresh_at`, assert it moved. Then, against a stub
  index builder: a search with a moved fingerprint rebuilds before running and reports
  `index_rebuilt: true`; a search with an unchanged fingerprint does not rebuild; a builder
  that raises leaves the old index in place, marks `stale`, and still serves the search;
  the background poll skips its tick while the lock is held and respects the 5-minute
  debounce.
- **`tests/test_feedback.py`** — an in-memory database via `db.init_schema` (the pattern
  `conftest.py` already uses), asserting the inserted row matches `_write_mark`'s columns
  exactly except `source='discord'`, that a repeat vote replaces rather than duplicates,
  and that an unknown `query_id` writes nothing.
- **`tests/test_bind_guard.py`** — the address validator accepts `127.0.0.1` and `::1` and
  rejects `0.0.0.0`, `::`, and a LAN address. Three lines, and it is the guard that keeps a
  second service off this machine's network.

`dev-requirements.txt` gains `-r serve-requirements.txt` so the full suite runs after one
install, while the `importorskip` keeps the minimal install honest. Expected total:
**~80 tests, still zero network calls.**

Not tested, and deliberately: discord.py's gateway behaviour, real Ollama responses, and
the systemd units themselves. Those are verified by hand once — `systemctl --user start`,
one `/scry`, one 👍, `journalctl --user -u scrying-api` — and that manual pass is the
acceptance criterion for the deployment step.

---

## Not building

Listed so nobody re-derives them as gaps:

- **Authentication and authorisation.** Discord is the identity; loopback is the boundary.
- **Rate limiting, quotas, abuse handling, moderation.** Audience is under ten people who
  know each other.
- **Public exposure, TLS, nginx proxying, a tunnel.** Tailscale is already on this machine
  if remote access is ever wanted.
- **Docker.** Everything else here is systemd user units; a container adds GPU passthrough
  and volume plumbing to solve nothing.
- **Any GPU mutual-exclusion mechanism** — pause endpoints, drain protocols, serving
  leases, `Conflicts=`. Ollama arbitrates VRAM already, the contention is a slowdown rather
  than a failure, and the window is usually zero-length.
- **A refresh→API completion signal.** Evaluated and rejected in favour of the
  self-sufficient fingerprint check.
- **A web UI.** The API makes it cheap later; that is not a reason to build it now.
- **Result caching.** The same theme twice is rare, and a cache would serve pre-refresh
  results after a refresh — a staleness bug traded for an optimisation nobody asked for.
- **Streaming progress** ("routing… expanding… judging 40 candidates…"). Genuinely nicer,
  and it means threading progress callbacks through `search.execute`, which is a change to
  `cts/`. The `~80s` estimate carries the load instead.
- **A queue that survives an API restart.** A restart drops in-flight searches. Persisting
  the queue means a durable job store, and restarts are rare enough that "run it again" is
  the answer.
- **Cancelling an in-flight search.** Not possible against blocking synchronous code
  without process-level machinery, and nothing in this design needs it.
- **Per-user history, favourites, pagination past `k=5`, theme autocomplete.**
- **Prometheus metrics, structured log shipping.** `/health` and journald cover it.
- **An ANN index, or incremental index updates.** The README is explicit that brute force
  over 170k vectors is the right answer at this size, and a full rebuild is 4–7s.
- **Upgrading or pinning Ollama.** The system instance is 0.31.1 and out of date per the
  machine inventory; that is a separate job with its own risks and it is not this one.

---

## Order of work

1. `serve/render.py` and its tests. Pure functions, no dependencies, and it forces every
   result-shape question to be answered before anything is wired up.
2. `serve/api.py`: lifespan, index build, `/search`, `/health`, the lock. Verify with
   `curl` against real Ollama and the real corpus, before Discord exists at all.
3. The fingerprint check, the synchronous rebuild, the background poll, and
   `tests/test_staleness.py`. Verify by running `python -m cts ingest` in one terminal and
   watching `/health`'s `index.built_at` move in another within a minute.
4. `/feedback` and `/admin/reload`.
5. `serve/bot.py`: defer, placeholder, results, buttons.
6. `serve/install-services.sh`, then enable `cts-refresh.timer`, then
   `loginctl enable-linger`.
7. One full manual pass: start the refresh by hand, confirm the bot's placeholder says the
   refresh is running, confirm a search still completes, confirm the index picks up new
   data without a restart.
8. Update `~/.claude/CLAUDE.md`'s machine inventory. Update the README with a serving
   section.
