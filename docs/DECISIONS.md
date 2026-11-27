# Design & Decision Log

A running record of every significant design decision, experiment, and result in this project — written so you can learn RL from the *why*, not just the *what*. Newest entries at the bottom. Each entry: **Decision → Why → RL concept → Result** (when known).

---

## Phase 0 — Foundations

### D1. Format: `gen9randombattle`
- **Decision:** Train on Gen 9 Random Battles only.
- **Why:** Random teams remove teambuilding (a separate, huge search problem). The bot only has to learn *play*. It also gives enormous variety for free — no two battles have the same teams, which prevents memorizing.
- **RL concept:** *Environment design / task scoping.* Pick the smallest environment that still contains the skill you care about.

### D2. Local server, never the public one
- **Decision:** Run Pokémon Showdown locally with `--no-security`; set `exports.repl = false` in `pokemon-showdown/config/config.js`.
- **Why:** RL needs millions of steps. A local server does ~60 battles in 2 seconds; the public server would rate-limit us and it would be rude. The REPL change was needed because macOS limits Unix socket paths to 104 chars and the project path (with spaces) exceeded it.
- **RL concept:** *Sample throughput is the bottleneck in RL.* Unlike supervised learning, you generate your own data — simulator speed directly bounds how much you can learn.

### D3. Write a heuristic bot before any RL (`bots/heuristic_bot.py`)
- **Decision:** Hand-code a bot first: score each move by base power × STAB × type effectiveness × accuracy × attack/defense ratio; switch when the matchup is bad.
- **Why:** (1) Learn the battle API; (2) it becomes a baseline opponent and a sanity check. Result: 100% vs random, 80% vs max-power, ~45–55% vs poke-env's `SimpleHeuristicsPlayer`.
- **RL concept:** *Always have baselines.* An RL number means nothing without a reference point. Many "RL successes" lose to a 50-line heuristic.

### D4. Library stack: poke-env 0.17 + sb3-contrib MaskablePPO
- **Decision:** poke-env for the protocol/battle-state parsing and its Gymnasium env; Stable-Baselines3's `MaskablePPO` for learning.
- **Why:** Don't reinvent websocket parsing or PPO. MaskablePPO adds *action masking* to PPO out of the box.
- **RL concept:** *PPO (Proximal Policy Optimization)* — collect a batch of experience with the current policy, compute *advantages* (was this action better or worse than the value network expected?), then nudge the policy toward good actions with a *clipped* update so it can't change too much at once. Same algorithm family used in RLHF for LLMs.

---

## Phase 1 — Environment design (built by the env subagent)

### D5. Observation: 226 hand-engineered features, normalized to [-1, 1]
- **Decision:** `pokebot/features.py` turns a battle into 226 numbers: 4 move slots × 15 features (power, effectiveness, STAB, accuracy, category, priority, PP, estimated damage…), both active Pokémon × 40 (HP, status, boosts, types, base stats), speed comparison, our 6 team slots × 7 (HP, fainted, can-switch, matchup score), opponent summary, weather/terrain, hazards/screens per side, misc (can Tera, forced switch, turn).
- **Why:** The network only "knows" what's in this vector. Hand-crafted features (like type effectiveness already computed) make learning far easier than raw data — the network doesn't have to re-discover the type chart.
- **Key detail:** Move slot *i* in the observation is exactly move *i* in the action space. If those orders disagree, the agent learns garbage.
- **RL concept:** *Observation/feature engineering* and *normalization*. Neural nets train badly when inputs have wildly different scales (e.g. base power 120 next to HP fraction 0.5).

### D6. Action space: 26 discrete actions with masking
- **Decision:** poke-env's layout: 0–5 switch to team slot, 6–9 use move, 22–25 use move + Terastallize (10–21 are mega/z/dynamax, always illegal in Gen 9). Illegal actions get masked (logit → −∞).
- **Why:** At any moment most of the 26 actions are illegal. Without masking the agent wastes millions of steps learning "don't pick nonexistent actions."
- **RL concept:** *Action masking.* Cheap, huge sample-efficiency win whenever legality is known.

### D7. Reward: potential-based shaping + ±30 for win/loss
- **Decision:** `Φ(state) = Σ our HP − Σ their HP + 2·(their faints − our faints) + 0.5·(their statuses − ours)`. Per-step reward = `shaping_scale × (Φ_now − Φ_prev)`, plus ±30 at the end.
- **Why:** Pure win/loss is a very sparse signal (≈30 turns of zeros, then ±1). Shaping gives feedback every turn. Using the *difference* of a potential function provably doesn't change which policy is optimal (Ng, Harada & Russell, 1999) — it only speeds learning — which protects against *reward hacking* (e.g. stalling to farm "HP lead" reward).
- **RL concept:** *Reward shaping, potential-based shaping, reward hacking, credit assignment.*

### D8. `strict=False` in the env
- **Decision:** An illegal action or unconvertible order becomes a random legal move (with a warning) rather than crashing.
- **Why:** A 1-hour run shouldn't die on a rare edge case (e.g. Struggle). Warnings let us notice if it happens often.
- **RL concept:** *Robust environments.* Long training runs amplify rare bugs.

---

## Phase 2 — Training design (built by the training subagent)

### D9. Curriculum: random → max-power → heuristics → self-play
- **Decision:** Four stages; advance early when rolling win rate over 200 battles passes 85% / 70% / 60%, or forcibly at 10% / 25% / 50% of training.
- **Why:** A brand-new agent vs a strong opponent loses every game → every reward looks the same → nothing to learn from. Start easy so wins (signal) happen, then raise difficulty.
- **RL concept:** *Curriculum learning*; *opponent selection in multi-agent RL.*

### D10. Self-play with a pool of past checkpoints (+ heuristic opponents mixed in)
- **Decision:** Stage 4 samples 65% past checkpoints (last 20, favoring the 3 newest), 30% heuristic bots, 5% max-power.
- **Why:** Playing only your latest self can *cycle* (A beats B, B beats C, C beats A) or overfit to your own quirks. A pool of past selves (like AlphaStar's "league") and fixed heuristic opponents keep the agent robust and stop it forgetting how to beat simple strategies (*catastrophic forgetting*).
- **RL concept:** *Self-play, fictitious play, non-transitivity, catastrophic forgetting.*

### D11. Anneal shaping 1.0 → 0.1 over training
- **Decision:** Linearly shrink the HP/faint/status reward weight; the ±30 win term stays.
- **Why:** Shaping helps early but is a *proxy*. Late in training we want the agent optimizing the real objective (winning) — e.g. sacrificing a low-HP mon is good play even though it "loses Φ."
- **RL concept:** *Proxy objectives vs true objectives.*

### D12. Hyperparameters
- **Decision:** MLP [256, 256] for both policy and value heads; lr 3e-4 decaying to 3e-5; γ = 0.99; GAE λ = 0.95; clip 0.2; entropy coef 0.01; 1024 steps/env per rollout; batch 256; 10 epochs.
- **Why:** Standard PPO defaults that work across many tasks. γ = 0.99 → effective horizon ≈ 100 steps, longer than a battle (~30 turns), so the agent cares about winning, not just this turn. Entropy bonus keeps exploration alive.
- **RL concept:** *Discount factor γ, GAE (bias/variance trade-off in advantage estimation), entropy regularization.*

### D13. Bug: username collisions from RNG reseeding
- **Decision:** Never seed Python's global `random`; load models with `custom_objects={"seed": None}`.
- **Why:** poke-env generates usernames from `random`; SB3 reseeds `random` when loading a seeded model → two players got the same name → login failed ("nametaken"). Fix applied in train.py, callbacks.py, env.py, play.py.
- **Lesson:** Global RNG state is shared, invisible coupling between libraries.

---

## Phase 3 — Results & experiments

### E1. First real run (v1): 1M steps, 4 envs, 32 minutes
- Stage 1 (random) cleared at 39k steps (85% threshold); stage 2 (max-power) at 117k (70%); stage 3 (heuristics) ran until the forced switch at 500k; then self-play.
- Eval vs SimpleHeuristics (50 battles each) bounced 22% → 58%.
- **Proper eval (300 battles/opponent):** `final.zip`: 99% vs random, 86% vs max-power, 45% vs our HeuristicBot, 49% vs SimpleHeuristics. `best.zip` (the "58%" checkpoint) scored only 46% on 300 battles.
- **Lesson — the winner's curse:** picking the max of 20 noisy 50-battle evals selects for *luck*. 95% CI for 50 battles at 50% is ±14%! Fix: evals of 200 battles for selecting `best.zip` (±7%), 300+ for reporting.
- **RL concept:** *Evaluation noise, selection bias, confidence intervals.*

### D14. Resume without re-inflating shaping
- **Decision:** When resuming v1 for 2M more steps, pass `--shaping-start 0.1 --shaping-end 0.1`.
- **Why:** The anneal is computed from progress within the *current* call (1M done / 3M total = 33%), so resuming would have bumped shaping from 0.1 back to ~0.7 — silently changing the reward mid-training.
- **Lesson:** Schedules (lr, shaping, entropy) must be checked when resuming.

### D15. Parallelism: tried 8 subprocess envs, reverted to 4 in-process envs
- Tried `--subproc --n-envs 8`: ~1.5× faster (≈850 vs 580 steps/s). Reverted at your request to keep CPU/fans at the original level (one process ≈ 65% of one core).
- **Why it wasn't faster still:** PPO alternates *rollout* (battles; parallelizable) and *update* (gradient steps in one process). Speeding up only rollouts hits Amdahl's law.
- **Rule from now on:** one training job at a time, 4 envs, single process.

### D16. Feature set v2
- **Decision:** Added a second observation version instead of editing the first. `pokebot/features.py` (v1, 226 numbers) is frozen; `pokebot/features_v2.py` builds v2 = the exact v1 vector + 120 new numbers (346 total). `pokebot/feature_sets.py` maps `"v1"`/`"v2"` → (size, embed function). `train.py --features v1|v2` (default v2) records the choice in `train_state.json`, and `--resume` reuses it. `PolicyPlayer` reads the version off the model's input size, so v1 checkpoints still play correctly in `evaluate.py`, `play.py`, and as self-play opponents inside a v2 training run.
- **Why versioning:** A trained network is tied to its input layout. Feature *i* means one specific thing to it, so changing the vector under an old checkpoint quietly turns it into a random player. Keeping v1 byte-for-byte means the 2.2M-step v1 agent stays usable as a baseline and as a sparring partner.
- **What was added, and why each part should help:**
  - **Damage as a % of the target's current HP (12 numbers: 4 move slots × damage, likely-KO, priority-KO).** v1 only had a relative "power score". v2 runs a simplified Gen 9 damage formula (level, real or estimated stats, boosts, STAB/Tera STAB, type chart, burn, sun/rain, average roll, multi-hit), with accuracy handled separately. "Can I KO it this turn?" and "can I KO it before it moves?" (priority) become single yes/no inputs. Before, the network had to work these out by multiplying a dozen raw inputs together.
  - **Opponent threat from its possible moves (31).** Random battles are *partially observable*: we only see moves the opponent has used. Showdown's random-battle set file lists every move each species can carry (per role), so we narrow it to the roles that fit the moves already seen. From that list we compute: the most damage it can do to our active, whether it can KO, a super-effective move, a priority attack, setup / recovery / pivot / hazard / status moves, whether it is faster than us (real speed estimates, boosts, paralysis, Tailwind, Trick Room), and its possible Tera types. This is like giving the agent an experienced player's knowledge of common sets. It cuts down the hidden information without any memory.
  - **Per switch-in matchup (30: 6 team slots × level, speed vs their active, best damage we deal, worst damage we take, net).** Switching is the hardest part of Pokémon. v1 only gave a type-matchup score per slot. Now the switch action for slot *i* sits next to "how hard does their active hit this mon, and how hard does it hit back". That is the information a good player uses to pick a safe switch-in, and it lines up with the action index (D5's alignment rule).
  - **Opponent team slots (42: 6 × revealed, HP, fainted, status, active, its threat to our active, our damage to it).** This shows which of their benched mons we have to respect and which we can clean up later. It helps the agent plan beyond the current pair.
  - **Short history (5): opponent switched last turn, we switched last turn, how effective the opponent's last move on us was, turns our active / their active has been in (÷10).** All read from the battle log by a bounded backward scan. A feedforward policy has no memory (no RNN). These few bits are a cheap stand-in, a *hand-made state summary*: is the opponent switching a lot, has it just hit us super-effectively, has our mon been sitting there long.
- **Simplifications (by design):** items, most abilities, crits, screens and terrain are ignored. Opponent stats assume the random-battle spread (31 IVs / 84 EVs / neutral nature). A few ability immunities (Levitate, Flash Fire, Water/Volt Absorb, …) are applied when the ability is known or every random-battle ability of that species grants the immunity. The estimates only need to rank options well, not match the official calculator exactly. Cost is about 0.2 ms per observation (v1 alone is about 0.1 ms), which is small next to a battle step.
- **RL concept:** *Feature engineering vs representation learning*, and *partial observability (POMDP)*. A small MLP trained for a few million steps learns much faster when the inputs already line up with the quantities that decide good actions (KO thresholds, speed order, switch-in safety). Adding prior knowledge (likely movepools) shrinks the hidden state. Hand-made history features are a lightweight alternative to recurrent policies. The version check guards against *train/serve skew*: a model must see the same feature layout when it plays as when it trained.

### D17. Behavior cloning warm start (`pokebot/bc.py`) — "SFT before RL"
- **Decision:** Record ~3,000 battles of a *teacher* (poke-env's `SimpleHeuristicsPlayer`, the strongest scripted bot we have) and train the policy network to imitate it with supervised learning *before* PPO. Then `pokebot.train --init-from bc.zip` fine-tunes with RL.
  - Policy head: cross-entropy toward the teacher's action, using the same action mask PPO uses.
  - Value head: regression onto the discounted *shaped return* the teacher actually obtained from each state (same reward formula as the env, γ = 0.99). A pre-trained critic matters: PPO's advantages come from the value function; a random critic produces noisy advantages that can wreck the cloned policy in the first few updates.
  - Validation split is **by battle**, not by sample — states from the same battle are highly correlated, so a per-sample split would leak and overstate accuracy.
  - Early stopping on validation NLL.
- **Why:** PPO from scratch spends hundreds of thousands of steps rediscovering "click the super-effective move." Imitation gets that for free, so RL can spend its budget on the hard part (switching, setup timing, Tera). This is exactly the LLM recipe: pretrain/SFT on demonstrations, then RL to exceed the demonstrator.
- **Known limitation — distribution shift / compounding errors:** a cloned policy only saw states the *teacher* reached. Its own small mistakes lead to unfamiliar states where it errs more. RL fixes this because it trains on the policy's *own* state distribution (DAgger is the supervised-learning fix).
- **Known risk — forgetting during fine-tuning:** early PPO updates can destroy the cloned behaviour ("unlearning"). Mitigations used: pre-trained value head, skip the trivial curriculum stages (start at stage 3), and a lower learning rate. (A KL penalty toward the BC policy — exactly RLHF's KL-to-reference term — is the stronger fix if needed.)
- **RL concept:** *Imitation learning, behavior cloning, distribution shift, warm-starting actor and critic.*

### E2. v1 continued to 3M steps (self-play), evaluated over 500 battles
- 200-battle evals during 1M→3M hovered 41–64% vs SimpleHeuristics with no obvious trend (noise ±7%), yet the **500-battle eval shows real improvement**: `v1/final.zip` (3M) = 99.2% random, 86.6% max-power, **59.0% HeuristicBot, 54.8% SimpleHeuristics**, and it beats the 1M model head-to-head **59.8%** (CI 55–64%).
- **Lesson:** slow improvement is invisible under eval noise. A 5-point gain needs ~500+ battles per side to detect. Always confirm "plateaus" with a big eval before concluding.
- **Lesson:** `best.zip` (63.5% on a 200-battle eval) scored 54.2% on 500 → the winner's curse again, smaller with bigger evals.

### E3. Behavior cloning results
- Data: 10,000 teacher battles (`SimpleHeuristicsPlayer` vs a mix of opponents) → 216,981 decisions (forced moves with a single legal action dropped). Collection took ~3 minutes.
- Training (20 epochs, early-stopped on held-out battles): **90.7% validation agreement** with the teacher. Train accuracy kept rising to 97% while validation flattened at ~90% → mild overfitting, handled by early stopping.
- Value head: validation MSE ≈ 590 vs return variance ≈ 896 → explains ~34% of variance. The rest is largely *irreducible*: from turn 1 of an even battle, the outcome is close to a coin flip, so the ideal value is ~0 with ±30 noise.
- **Playing strength:** 50.6% vs SimpleHeuristics (it *is* a copy, so ≈50% is expected), 49.0% vs HeuristicBot, 89.4% vs max-power, and **44.4% vs v1-3M**. Imitation reaches the teacher's level but not beyond — which is exactly why RL comes next.
- **RL concept:** *the imitation ceiling.* BC can at best match its demonstrator; RL can exceed it because it optimizes outcomes rather than agreement.

### E4. BC warm start + PPO, v2 features
- `--init-from checkpoints/bc_v2/bc.zip --start-stage 3 --lr 1e-4 --timesteps 2M`. Skips random/max-power stages (the clone already beats them), lower learning rate to avoid unlearning the cloned behaviour.
- First 12k steps: 46% → 60% rolling win rate vs heuristic opponents — i.e. it starts where v1 reached only after ~500k steps.
- Progress (200-battle evals vs SimpleHeuristics): 51% @200k → 57.5% @300k → 61.5% @350k → 63% @450k. **The fine-tuned agent has passed its teacher** (the pure clone scored 50.6%). This is the core promise of "imitate, then RL."
- **Result (500 battles/opponent):** `v2bc/final.zip` = 91.0% max-power, **66.0% HeuristicBot, 62.6% SimpleHeuristics**, and **56.6% head-to-head vs v1-3M** (CI 52–61%). `best.zip` (73% on one 200-battle eval) ≈ `final.zip` (49.4% head-to-head) — winner's curse again; the eval curve after 1M was a noisy 56–73% band.
- 2M steps of warm-started training beat 3M steps from scratch. Took 75.6 min (442 steps/s — v2 features are slightly slower to compute and battles vs heuristic opponents are slower than vs random).

### E5. Ablation: v2 features from scratch (no BC), 1M steps
- **Why:** E4 changed two things at once (v2 features *and* BC warm start). To learn which one mattered, change one variable at a time. Compare against v1 at 1M (≈49% vs SimpleHeuristics on 300 battles).
- **RL concept:** *ablation study* — the standard way to attribute gains in ML research.
- **Result (500 battles/opponent):** `v2scratch/final.zip` @1M = 91.2% max-power, **68.8% HeuristicBot, 64.2% SimpleHeuristics**, **59.0% vs v1-3M**, and **49.0% vs v2bc-2M (a tie)**. Curriculum stages cleared faster than v1 (stage 3 → self-play by win-rate threshold at 392k; v1 never hit that threshold and was force-advanced at 500k).
- **Conclusion:** the **features** were the big win. v2 from scratch at 1M ≈ BC+v2 at 2M ≫ v1 at 3M. BC bought a fast start (it began near 50–60%) but not a higher ceiling in this budget.
- **Why BC didn't add more here:** (1) the teacher (SimpleHeuristics) is only ~as strong as what PPO reaches on its own in ~400k steps with good features, so cloning it saves little; (2) the clone is near-deterministic, so early PPO explores less; (3) we lowered the learning rate to protect the clone, which also slows improvement. BC shines when the demonstrator is much stronger than what RL finds quickly (e.g. human expert replays) or when RL from scratch can't get off the ground.
- **RL concept:** *representation matters more than algorithm tweaks* — giving the agent the right information (damage as % of HP, KO flags, opponent's possible moves, switch-in threat) changed the ceiling; warm-starting only changed the speed.

### D18. Final overnight run: continue v2 from scratch for +3M steps
- **Decision:** Resume `v2scratch` (not `v2bc`) for 3M more steps with shaping held at 0.1.
- **Why:** both were tied, but v2scratch's eval curve was still rising at 1M (61→64→68→63.5→72%) while v2bc had been flat at ~62% for its last 1M steps. Its learning-rate schedule re-warms on resume (≈2.3e-4 decaying to 3e-5), which is fine for a run that is still improving.
- Backed up the 1M models as `final_1M.zip` / `best_1M.zip`.
- Took 114 minutes (439 steps/s average) at the agreed CPU budget (4 in-process envs).

### E6. Final results: v2 from scratch, 4M steps → `checkpoints/champion.zip`
- 200-battle evals vs SimpleHeuristics during 1M→4M sat in a noisy 57–72% band and drifted up slowly (mostly 60–64% before 2.5M, mostly 66–72% after 3M).
- **500-battle results:**

  | model | random | max-power | HeuristicBot | SimpleHeuristics |
  |---|---|---|---|---|
  | v1 @3M | 99.2% | 86.6% | 59.0% | 54.8% |
  | BC clone | — | 89.4% | 49.0% | 50.6% |
  | v2bc @2M | — | 91.0% | 66.0% | 62.6% |
  | v2scratch @1M | — | 91.2% | 68.8% | 64.2% |
  | v2scratch `best.zip` (3.0M, 72.5% on a 200-battle eval) | — | 89.4% | 70.0% | 63.8% |
  | **v2scratch `final.zip` @4M (champion)** | **98.6%** | **92.0%** | **73.2%** | **70.6%** |

- **Champion head-to-head (500 battles each):** 61.2% vs itself at 1M, 53.8% vs `best.zip`, 59.6% vs v2bc-2M, **65.2% vs v1-3M**.
- **Champion choice:** `final.zip` over `best.zip`. `best.zip` won its spot on a single lucky 200-battle eval (72.5%) and only measured 63.8% when re-tested on 500 battles, the **winner's curse** for the third time. Choosing a checkpoint by its maximum score on a noisy eval biases you toward lucky checkpoints. Re-test the candidates on a large fresh sample before picking (this is the reason for separate validation and test sets).
- **Lesson: self-play kept paying off slowly.** The +3M steps added about 6 points vs SimpleHeuristics and 4.5 vs HeuristicBot. In head-to-head the 4M model beats its 1M self 61%, which shows real improvement and not just over-fitting to the scripted bots. Diminishing returns are visible, though: roughly +6 points for 3× the compute.
- **Rough ranking of what mattered, by measured win-rate gain vs SimpleHeuristics:** better features (+15 points at equal steps) > more training/self-play (+6 from 1M→4M) > BC warm start (≈0 at the ceiling, but a much faster start).

## Phase 4 — v3: items, crits, memory (built by three subagents in parallel; not trained yet)

### D19. Feature set v3 (635 numbers) = v2 + items (71) + crits (22) + history (196)
- **Same versioning rule as D16:** `features_v3.embed_battle_v3` is the exact v2 vector followed by three new blocks, so v1/v2 checkpoints keep working and the champion can be widened (D20). About 0.35 ms per turn, versus 0.2 ms for v2.
- **Held items (`pokebot/v3_items.py`, 71):**
  - **What it includes:**
    - Our item and the opponent's item, as a one-hot over the 22 items that cover 96% of random-battle items.
    - Choice-lock state for both sides, and whether their locked move hits us super-effectively.
    - Item-aware damage and KO flags per move slot, and the opponent's item-aware threat.
    - Scarf-aware speed, and Heavy-Duty Boots on our bench.
  - **Hidden information, handled with a prior:** poke-env only knows the opponent's item once it shows (Leftovers healing, Life Orb recoil, Air Balloon, Knock Off). Until then the bot sees a *probability distribution* over likely items. It is built from the same rules Showdown's team generator uses, and narrowed as moves are revealed:
    - A status move rules out Choice items and Assault Vest.
    - Using two different moves without switching rules out every Choice item.
  - **How good the guess is** (checked against 1,500 generated teams): the top guess is right 73% of the time with no moves seen and 82% after two moves.
  - **RL concept:** *belief state.* In a partially observable game, the right input is your *belief* about the hidden variable, not a single guess. "40% Choice Band" lets the policy weigh both worlds.
- **Critical hits (`v3_items.py`, 22):** crits are random, so they can't be predicted, only *priced in*. The block gives:
  - Crit chance per move, using Showdown's stage rules: base 1/24, high-crit moves, Scope Lens, Super Luck, Focus Energy and always-crit moves.
  - A "KOs only if it crits" flag.
  - Expected damage, (1−p)·normal + p·crit.
  - The opponent's crit-KO risk against us.
  - **RL concept:** *expected value under stochastic transitions.* The environment is random, and the policy should optimise the average outcome while knowing the variance.
- **Memory (`pokebot/v3_history.py`, 196):**
  - **Last 6 turns:** each one records our decision (with move slot, so "the move I just used failed" lines up with the action that used it), the opponent's decision, effectiveness, damage both ways, KOs, crits, who moved first, and status.
  - **Whole-battle tallies:** switch rates; how often *this* opponent switches out when we threaten it, using a count with a prior, (k+1)/(n+2); damage and KO totals; hazards; tera usage; uses of each move.
  - **Our previous action**, as a one-hot.
  - **Why not an LSTM:** sb3-contrib has no recurrent PPO that supports action masks. Hand-built summaries are cheaper and more stable, but only remember what we chose to record.
  - **RL concept:** *POMDPs and memoryless policies.* An MLP maps one observation to one action, so anything it should remember has to be written into the observation. Feeding back the previous action is the classic trick.

### D20. Warm start by network surgery (`pokebot/surgery.py`)
- **Decision:** don't retrain from scratch. Widen the champion's two input layers from 346 to 635 inputs:
  - Copy the old weights.
  - Set every weight on a new input to **zero**.
  - Copy all the other layers unchanged.
- **Result:** the new network computes *exactly* the same function. On 2,048 random inputs the output difference is 0. On 100 real battle turns it chose the same action as the champion every time. On a quick 20-battle check it scored 14/20 against SimpleHeuristics, the champion's level.
- **Why zero weights still learn:** the gradient for a weight is (its input) × (the gradient flowing into its neuron). Neither depends on the weight's current value, so PPO starts adjusting a new feature's weights as soon as that feature carries a useful signal.
- **Risk:** the policy sits in a comfortable spot built around the old inputs and may learn to use the new ones only slowly. Remedies, if needed: a little random noise on the new weights (keep it ≤ 1e-3; 0.01 already changed 4% of decisions), or a higher learning rate just for the new weights.
- **RL concept:** *function-preserving expansion (Net2Net)*, the same idea as growing a model mid-training instead of starting over.

### E7. Does v3 actually help? (run 2026-11-20, 17:00–19:55)
- **Treatment:** fine-tune the widened champion with v3 features for 2M steps.
  ```bash
  .venv/bin/python -m pokebot.train --run-name v3ft --features v3 --init-from checkpoints/v3_init/init.zip \
      --start-stage 4 --lr 1e-4 --shaping-start 0.1 --shaping-end 0.1 --timesteps 2000000 --n-envs 4 --eval-battles 200
  ```
- **Control:** fine-tune the *v2* champion with the same settings, so "more training" isn't mistaken for "better features".
  ```bash
  .venv/bin/python -m pokebot.train --run-name v2ft --features v2 --init-from checkpoints/champion.zip \
      --start-stage 4 --lr 1e-4 --shaping-start 0.1 --shaping-end 0.1 --timesteps 2000000 --n-envs 4 --eval-battles 200
  ```
- **How to judge:** 500-battle evals for each against the scripted bots, plus a 500-battle head-to-head `v3ft/final.zip` vs `v2ft/final.zip`. Each run is about 75–85 min at the usual 4-env CPU budget; v3 is slightly slower per step. Run them one after the other, not at the same time.
- **Ran:** v3ft took 92 min (362 sps; the v3 features cost ~0.35 ms/turn) and v2ft took 78 min. Script: `scripts/run_e7.sh`; logs: `logs/v3ft-train.log`, `logs/v2ft-train.log`, `logs/e7-*.log`.

| 500 battles each | v3ft (v3, +2M) | v2ft (v2, +2M) | champion (for reference, E6) |
|---|---|---|---|
| random | 98.8% | 98.4% | 98.6% |
| maxpower | 92.6% | 94.2% | 92.0% |
| heuristic | 76.8% [72.9, 80.3] | 77.6% [73.7, 81.0] | 73.2% |
| simple_heuristic | 74.2% [70.2, 77.8] | 71.4% [67.3, 75.2] | 70.6% |

| head-to-head (500) | win rate of the first model |
|---|---|
| v3ft vs v2ft | **49.4%** [45.0, 53.8] |
| v3ft vs champion | 49.8% [45.4, 54.2] |
| v2ft vs champion | 48.4% [44.0, 52.8] |

- **Result: a null result.** Neither the new features nor 2M more steps produced a bot that beats the champion. The scripted-bot numbers all moved by +0–4 points, but no pair differs by more than noise (v3ft vs v2ft on simple_heuristic is +2.8 points, z ≈ 1.0). The head-to-heads, which are the direct test, are all a coin flip. **The champion stays `checkpoints/champion.zip`.**
- **Why it might not have helped** (hypotheses, not yet tested):
  1. **Plateau, not a feature gap.** v2ft didn't beat the champion either, so self-play at lr 1e-4 has stopped improving at this network size. Any new feature has to compete with that.
  2. **Zero-init is slow to wake.** New columns start at 0 and get gradients only from the PPO signal, which is very noisy (one ±30 reward per ~30-turn battle). 2M steps may be too few to find small effects. A run with a higher lr for just the new layer, or 5M+ steps, would test this.
  3. **The information is mostly redundant.** v2 already priced damage; items shift those estimates a little, and crit odds (~4%) rarely flip a decision. Memory pays off mainly against *adaptive* opponents, but the scripted bots are memoryless and the self-play pool is a stateless policy too, so there is little to "remember" about them.
  4. **A flat MLP over 635 inputs.** The history block is 6×27 structured records; an MLP has to learn that slot 3 of record 2 means the same thing as slot 3 of record 5. A small shared encoder (or attention) per record would be more sample-efficient.
- **RL lesson:** a feature that helps a human think doesn't automatically help a policy learn. A new input must change *which action is best* often enough for the noisy return signal to notice it. And the matched control mattered: without v2ft, the +3.6-point simple_heuristic gain over the champion (70.6 → 74.2) would have looked like a v3 win, but v2ft got +0.8 with no new features, and the head-to-heads show no difference at all.

## Phase 5 — v4: abilities

### D21. v4 = v3 + an ability block (117 inputs → 752 total)
- **Trigger:** a human game (2026-11-24). Our Eiscue fainted with their Gyarados on ~2% HP and no weather up. Sending Tyranitar would have set sand and KO'd Gyarados at the end of the turn, whatever Gyarados did. The bot sent Espeon (policy 56% Espeon vs 41% Tyranitar) and lost the game. Nothing in v1–v3 encoded "this switch-in's ability changes the weather, and that weather finishes their active".
- **What the bot now sees** (`pokebot/v4_abilities.py`):
  - What each active's ability *does*, as 26 effect categories (weather setter, Intimidate, immunity, attack doubler, defense, survive-a-hit, weather speed, snowball, Unaware, Magic Guard, Regenerator, Mold Breaker, …). Categories rather than a one-hot over 200+ abilities, so "Swift Swim" and "Chlorophyll" share what they mean.
  - The opponent's ability as a **posterior**: one-hot once revealed; before that, uniform over the random-battle roles consistent with its revealed moves (same idea as the item prior in D19).
  - Ability-aware damage, P(nullified) and P(KO) for each of our 4 moves (Levitate, Flash Fire, Water Absorb, Sturdy, Multiscale, Disguise, Ice Face, Huge Power, Adaptability, Technician, …), and the opponent's ability-aware threat to us.
  - **End-of-turn residual** for both actives: sand, burn, poison/toxic, Leech Seed, Salt Cure, Leftovers, Black Sludge, Grassy Terrain, Rain Dish/Dry Skin/Ice Body, Poison Heal, Magic Guard; Cloud Nine/Air Lock cancel weather. Plus "dies at end of turn" flags.
  - Ability-aware speed (Swift Swim/Chlorophyll/Sand Rush/Slush Rush/Surge Surfer, Quick Feet, Protosynthesis/Quark Drive, Trick Room), and Regenerator's heal on switching out.
  - **Per team slot (aligned with switch actions 0–5):** the weather after that mon switches in, the end-of-turn damage it then deals to their active and whether it KOs, the value of Intimidate (negative into Defiant/Competitive), how many of their likely moves its ability nullifies, ability-aware threat and speed. In the lost game the Tyranitar slot now reads [chip KO = 1, changes weather = 1] and the Espeon slot reads 0 (a test reproduces it).
- **Cost:** about 0.1 ms per turn (v3 itself is about 0.35 ms).
- **Warm start:** `checkpoints/v4_init/init.zip` is `v3ft/final.zip` widened 635 → 752 with zero-initialised new columns (D20). Parity is exact: it agreed on 86 of 86 real turns and scored 39/50 vs SimpleHeuristics.
- **RL lesson / expectation:** E7 says a feature only helps if it changes the best action often enough for the win/loss signal to notice it. Weather-chip KOs on switch-in are rare. Immunities (switching a Levitate mon into Earthquake), Intimidate and residual damage come up far more often, and those are where any measurable gain should come from.

### E8 (planned, not run). Does v4 help?
- **Treatment:** `--run-name v4ft --features v4 --init-from checkpoints/v4_init/init.zip --start-stage 4 --lr 1e-4 --shaping-start 0.1 --shaping-end 0.1 --timesteps 2000000 --n-envs 4 --eval-battles 200`
- **Control:** the same settings from `checkpoints/v3ft/final.zip` with `--features v3` (run name `v3ft2`), so "more training" isn't mistaken for "abilities".
- **Judge:** 500-battle evals of each against the scripted bots, plus 500-battle head-to-heads v4ft vs v3ft2 and v4ft vs champion. Each run is about 90 min on the usual 4-env budget; run them one after the other.

## Ideas for next time (not done)
- **Stronger opponents:** the agent now beats every scripted bot ~70%+. Further progress will come from self-play and from league-style training (keep "exploiter" agents that only try to beat the current champion).
- ~~Memory~~ and ~~items in the damage model~~: built as v3 (D19), trained in E7, no measurable gain. Next to try: a per-turn shared encoder for the history block, a longer v3 run, or a bigger network. Abilities: built as v4 (D21).
- **Human replays for BC:** imitation helps most when the demonstrator is far stronger than what RL finds alone. Ladder replays would be a much better teacher than SimpleHeuristics.
- **Bigger evals during training** (500 battles) or a moving average, to make `best.zip` trustworthy.
- **Testing on the real ladder** would need an account and is against the spirit of the local-only rule (D2). Do it only with a clearly-marked bot account and the server's bot rules in mind.
