# four_in_a_roll_tree_v1.py
# Build & render the game-tree up to N turns with:
# - pruning below no-win alternatives
# - upstream red/yellow colouring (only if all downstream terminal wins are that colour and no no-win branches below)
# - rows driven by edges (prevents missing plotted nodes like 57)
# - hard assertions that ID 33 → ID 57 is present and plotted
#
# Output: /mnt/data/state_graph_turns1to8_full_logic_edges_rows_assert_33_57.pdf  (change TURNS / OUTFILE below)
#
# Requires: matplotlib

import json
import math
import os
import sys
from collections import defaultdict
from typing import Optional
from tqdm import tqdm

SUCCESSOR_CACHE = {}
WINNER_CACHE = {}
DISC_RESOLUTION = 22
OUTDIR = "outputs"

# ----------------------------
# Game rules & board utilities
# ----------------------------
# Board is bottom-indexed 6x7 initially: board[r][c], r=0 is BOTTOM row, r=ROWS-1 is TOP
INIT_ROWS, INIT_COLS = 6, 7
P1, P2 = 1, 2

def make_board(rows=None, cols=None):
    if rows is None:
        rows = INIT_ROWS
    if cols is None:
        cols = INIT_COLS
    return [[0 for _ in range(cols)] for _ in range(rows)]

def deep_copy(b):
    return [row[:] for row in b]

def in_bounds(b, r, c):
    return 0 <= r < len(b) and 0 <= c < len(b[0])

def get_next_open_row(b, col):
    if col < 0 or col >= len(b[0]): return -1
    for r in range(len(b)):
        if b[r][col] == 0:
            return r
    return -1

def drop_disc(b, col, player):
    """Place a disc in column (if possible) and return (ok, row). Does not rotate/settle."""
    r = get_next_open_row(b, col)
    if r == -1:
        return False, -1
    b[r][col] = player
    return True, r

def apply_gravity(b):
    rows, cols = len(b), len(b[0])
    out = [[0]*cols for _ in range(rows)]
    for c in range(cols):
        rr = 0
        for r in range(rows):
            if b[r][c] != 0:
                out[rr][c] = b[r][c]
                rr += 1
    return out

def rotate_cw_world(b):
    """Rotate 90 CW in the 'world'; since board is bottom-indexed, we:
       1) flip to top-index
       2) rotate CW in top space
       3) flip back to bottom-index
    """
    rows, cols = len(b), len(b[0])
    # bottom -> top
    top = [[0]*cols for _ in range(rows)]
    for rb in range(rows):
        rt = rows - 1 - rb
        for c in range(cols):
            top[rt][c] = b[rb][c]
    # rotate CW in top
    newTopRows, newTopCols = cols, rows
    rot = [[0]*newTopCols for _ in range(newTopRows)]
    for r in range(newTopRows):
        for c in range(newTopCols):
            rot[r][c] = top[rows - 1 - c][r]
    # back to bottom
    out = [[0]*newTopCols for _ in range(newTopRows)]
    for rt in range(newTopRows):
        rb = newTopRows - 1 - rt
        for c in range(newTopCols):
            out[rb][c] = rot[rt][c]
    return out

def line_winner(b, r, c, player):
    """Check from (r,c) if this player has 4 in a row."""
    if b[r][c] != player: return False
    dirs = [(1,0),(0,1),(1,1),(1,-1)]
    rows, cols = len(b), len(b[0])
    for dr, dc in dirs:
        cnt = 1
        rr, cc = r+dr, c+dc
        while 0 <= rr < rows and 0 <= cc < cols and b[rr][cc] == player:
            cnt += 1
            rr += dr; cc += dc
        rr, cc = r-dr, c-dc
        while 0 <= rr < rows and 0 <= cc < cols and b[rr][cc] == player:
            cnt += 1
            rr -= dr; cc -= dc
        if cnt >= 4: return True
    return False

def any_winner(b, key=None):
    if key is None:
        key = board_key(b)
    hit = WINNER_CACHE.get(key)
    if hit is not None:
        return hit
    rows, cols = len(b), len(b[0])
    for r in range(rows):
        for c in range(cols):
            v = b[r][c]
            if v in (P1, P2) and line_winner(b, r, c, v):
                WINNER_CACHE[key] = v
                return v
    WINNER_CACHE[key] = 0
    return 0

def board_key(b):
    # Immutable key for dedup (include size and cells)
    return (len(b), len(b[0]), tuple(tuple(row) for row in b))

# ---------------------------------
# Successor generation (one "turn")
# ---------------------------------
def successors_endstate_dedup(board, mover):
    """All unique end-of-turn states after the mover places one disc:
       - place
       - check pre-rotate win (flag 'prewin')
       - if no prewin: rotate + gravity
       - return child dicts with fields:
         { 'board', 'key', 'cols' (sequence), 'winner', 'prewin' }
       Deduplicate on final board state key.
    """
    rows, cols = len(board), len(board[0])
    uniq = {}
    for c in range(cols):
        if get_next_open_row(board, c) == -1:
            continue
        b2 = deep_copy(board)
        ok, _row = drop_disc(b2, c, mover)
        if not ok:  # should not happen with check above
            continue
        pre_key = board_key(b2)
        prewin = (any_winner(b2, pre_key) == mover)
        if not prewin:
            b2 = rotate_cw_world(b2)
            b2 = apply_gravity(b2)
            k = board_key(b2)
        else:
            k = pre_key
        winner = any_winner(b2, k)
        # dedup by final board only; consolidate column choices that yield same effect
        if k not in uniq:
            uniq[k] = {
                "board": b2,
                "key": k,
                "cols": [c],
                "winner": winner,
                "prewin": prewin
            }
        else:
            uniq[k]["cols"].append(c)
    # Turn dicts
    out = []
    for k, v in uniq.items():
        out.append(v)
    return out


def _prune_children(children, mover):
    """Return child states that survive the pruning policy for the parent.
    The policy is intentionally asymmetric:
    * If any successor ends the turn with the mover winning, we keep *only*
      the winning children. Exploring losing or neutral lines is pointless if a
      direct win is already available.
    * Otherwise (no immediate win for the mover), we discard successors where
      the opponent wins immediately. Those moves are dominated: the mover would
      never choose a line that hands the opponent a win on the spot, so the
      graph stays focused on neutral / still-fighting continuations.
    """
    if any(ch["winner"] == mover for ch in children):
        return [ch for ch in children if ch["winner"] == mover]
    opp = P1 if mover == P2 else P2
    return [ch for ch in children if ch["winner"] != opp]

def cached_successors(board, mover, key=None):
    """Memoized wrapper around successors_endstate_dedup keyed by immutable board."""
    if key is None:
        key = board_key(board)
    cache_key = (key, mover)
    hit = SUCCESSOR_CACHE.get(cache_key)
    if hit is not None:
        return hit
    result = successors_endstate_dedup(board, mover)
    SUCCESSOR_CACHE[cache_key] = result
    return result

# -----------------------------------------
# Build layers with pruning & deterministic IDs
# -----------------------------------------
def build_layers_pruned_with_ids(turns: int, *, disable_progress: bool = False):
    start = make_board()
    start_key = board_key(start)

    nodes = {
        0: {
            "id": 0,
            "key": start_key,
            "board": start,
            "layer": 0,
            "parents": [],
            "children": [],
            "child_layers": {},
            "child_columns": {},
            "incoming_columns": {},
            "player": None,
            "prewin": False,
            "winner": 0,
        }
    }
    key_to_id = {start_key: 0}
    layers = {0: [0]}
    next_id = 1

    progress = tqdm(
        total=0,
        desc="Expanding tree",
        unit="parent",
        disable=disable_progress or not sys.stderr.isatty(),
    )

    try:
        for layer in range(1, turns + 1):
            layers[layer] = []
            mover = P1 if (layer % 2 == 1) else P2
            parents = list(layers[layer - 1])

            if parents:
                progress.total = progress.n + len(parents)
                progress.refresh()

            for parent_id in parents:
                parent_node = nodes[parent_id]
                if parent_node["winner"]:
                    progress.update(1)
                    continue

                cache_key = (parent_node["key"], mover)
                kids = SUCCESSOR_CACHE.get(cache_key)
                if kids is None:
                    kids = successors_endstate_dedup(parent_node["board"], mover)
                    SUCCESSOR_CACHE[cache_key] = kids

                pruned_kids = _prune_children(kids, mover)

                for child in pruned_kids:
                    child_key = child["key"]
                    child_cols = list(child["cols"])

                    child_id = key_to_id.get(child_key)
                    if child_id is None:
                        child_id = next_id
                        next_id += 1
                        key_to_id[child_key] = child_id
                        nodes[child_id] = {
                            "id": child_id,
                            "key": child_key,
                            "board": child["board"],
                            "layer": layer,
                            "parents": [parent_id],
                            "children": [],
                            "child_layers": {},
                            "child_columns": {},
                            "incoming_columns": {parent_id: child_cols},
                            "player": mover,
                            "prewin": child["prewin"],
                            "winner": child["winner"],
                        }
                    else:
                        child_node = nodes[child_id]
                        if parent_id not in child_node["parents"]:
                            child_node["parents"].append(parent_id)
                        child_node["incoming_columns"][parent_id] = child_cols
                        child_node["layer"] = min(child_node["layer"], layer)

                    if child_id not in parent_node["children"]:
                        parent_node["children"].append(child_id)
                    parent_node["child_layers"][child_id] = layer
                    parent_node["child_columns"][child_id] = child_cols

                    if child_id not in layers[layer]:
                        layers[layer].append(child_id)

                progress.update(1)

            progress.set_postfix({"layer": layer, "frontier": len(layers[layer])})
    finally:
        progress.close()

    return nodes, layers

def filter_to_winning_paths(nodes, layers):
    """
    Returns:
      filtered_nodes: dict[node_id -> node_data] on any winning path + one-level children of those nodes
      added_children_nodes: dict[node_id -> node_data] included ONLY as one-level children (not on backbone)
      layers_keep: dict[layer -> list[node_id]] for kept nodes (layer 0 omitted)
    """
    winners = {nid for nid, node in nodes.items() if node["winner"] in (P1, P2)}

    keep_backbone = set()
    stack = list(winners)
    while stack:
        nid = stack.pop()
        if nid in keep_backbone:
            continue
        keep_backbone.add(nid)
        stack.extend(nodes[nid]["parents"])

    one_level_children = set()
    for parent_id in keep_backbone:
        one_level_children.update(nodes[parent_id]["children"])

    one_level_children.intersection_update(nodes.keys())
    added_children_ids = one_level_children - keep_backbone
    keep = keep_backbone | one_level_children

    def copy_node(node_id):
        node = nodes[node_id]
        kept_parents = [pid for pid in node["parents"] if pid in keep]
        kept_children = [cid for cid in node["children"] if cid in keep]
        return {
            **node,
            "parents": kept_parents,
            "children": kept_children,
            "child_layers": {cid: node["child_layers"][cid] for cid in kept_children},
            "child_columns": {cid: node["child_columns"][cid] for cid in kept_children},
            "incoming_columns": {pid: node["incoming_columns"][pid] for pid in kept_parents},
        }

    filtered_nodes = {nid: copy_node(nid) for nid in keep}
    added_children_nodes = {nid: filtered_nodes[nid] for nid in added_children_ids}

    layers_keep = {}
    for L, lst in layers.items():
        kept_layer_nodes = [nid for nid in lst if nid in keep]
        if kept_layer_nodes:
            layers_keep[L] = kept_layer_nodes

    return filtered_nodes, added_children_nodes, layers_keep

# -----------------------------------------
# Rendering with edges-driven rows & colouring
# -----------------------------------------
def render_full_logic(
    turns: int,
    *,
    plot_pdf: bool = False,
    exclude_wins: bool = False,
    max_lookahead: Optional[int] = None,
    disable_progress: bool = False,
):
    SUCCESSOR_CACHE.clear()
    WINNER_CACHE.clear()

    base = f"state_graph_{INIT_COLS}x{INIT_ROWS}_{turns}t{'_exclude_wins' if exclude_wins else ''}{f'_lookahead_{max_lookahead}' if max_lookahead else ''}"
    out_pdf = os.path.join(OUTDIR, f"{base}.pdf") if plot_pdf else None
    out_json = os.path.join(OUTDIR, f"{base}.json")

    for path in (out_pdf, out_json):
        if path is None:
            continue
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    nodes_full, layers_full = build_layers_pruned_with_ids(turns, disable_progress=disable_progress)
    nodes, truncated_nodes, layers = filter_to_winning_paths(nodes_full, layers_full)
    truncated_ids = set(truncated_nodes.keys())

    if 0 not in layers and 0 in nodes:
        layers[0] = [0]
    ordered_layers = sorted(layers.keys())

    base_child_map = defaultdict(list)
    for parent_id, node in nodes.items():
        for child_id in node["children"]:
            if child_id in nodes:
                base_child_map[parent_id].append(child_id)

    def make_evaluators(child_lookup):
        downstream_depth_cache = {}
        downstream_winner_cache = {}
        forced_cache = {}

        def downstream_min_depths(node_id):
            cached = downstream_depth_cache.get(node_id)
            if cached is not None:
                return cached
            if node_id in truncated_ids:
                downstream_depth_cache[node_id] = {}
                return {}
            node = nodes[node_id]
            if node["winner"] in (P1, P2):
                result = {node["winner"]: 0}
                downstream_depth_cache[node_id] = result
                return result
            best = {}
            for child_id in child_lookup.get(node_id, []):
                for player, depth in downstream_min_depths(child_id).items():
                    if depth is None:
                        continue
                    child_depth = depth + 1
                    current = best.get(player)
                    if current is None or child_depth < current:
                        best[player] = child_depth
            downstream_depth_cache[node_id] = best
            return best

        def downstream_winners(node_id):
            cached = downstream_winner_cache.get(node_id)
            if cached is not None:
                return cached
            depth_map = downstream_min_depths(node_id)
            if max_lookahead is None:
                winners = set(depth_map.keys())
            else:
                winners = {
                    player for player, depth in depth_map.items()
                    if depth is not None and depth <= max_lookahead
                }
            downstream_winner_cache[node_id] = winners
            return winners

        def player_to_move(node_id):
            last_player = nodes[node_id]["player"]
            if last_player is None:
                return P1
            return P1 if last_player == P2 else P2

        def forced_outcome(node_id):
            cached = forced_cache.get(node_id)
            if cached is not None:
                return cached
            if node_id in truncated_ids:
                forced_cache[node_id] = (0, None)
                return forced_cache[node_id]
            node = nodes[node_id]
            if node["winner"] in (P1, P2):
                forced_cache[node_id] = (node["winner"], 0)
                return forced_cache[node_id]
            cached_outcome = node.get("forced_outcome_cached")
            if isinstance(cached_outcome, tuple) and len(cached_outcome) == 2:
                forced_cache[node_id] = cached_outcome
                return cached_outcome
            if cached_outcome in (P1, P2):
                result = (cached_outcome, 0)
                forced_cache[node_id] = result
                return result
            children = child_lookup.get(node_id, [])
            if not children:
                forced_cache[node_id] = (0, None)
                return forced_cache[node_id]

            mover = player_to_move(node_id)
            child_results = [forced_outcome(child_id) for child_id in children]

            def bumped(depth):
                return None if depth is None else depth + 1

            if mover == P1:
                win_depths = [bumped(depth) for winner, depth in child_results if winner == P1]
                win_depths = [d for d in win_depths if d is not None]
                if win_depths:
                    result = (P1, min(win_depths))
                    forced_cache[node_id] = result
                    return result
                if any(winner == 0 for winner, _ in child_results):
                    result = (0, None)
                    forced_cache[node_id] = result
                    return result
                loss_depths = [bumped(depth) for winner, depth in child_results if winner == P2]
                loss_depths = [d for d in loss_depths if d is not None]
                if loss_depths:
                    result = (P2, max(loss_depths))
                    forced_cache[node_id] = result
                    return result
                result = (0, None)
                forced_cache[node_id] = result
                return result

            win_depths = [bumped(depth) for winner, depth in child_results if winner == P2]
            win_depths = [d for d in win_depths if d is not None]
            if win_depths:
                result = (P2, min(win_depths))
                forced_cache[node_id] = result
                return result
            if any(winner == 0 for winner, _ in child_results):
                result = (0, None)
                forced_cache[node_id] = result
                return result
            loss_depths = [bumped(depth) for winner, depth in child_results if winner == P1]
            loss_depths = [d for d in loss_depths if d is not None]
            if loss_depths:
                result = (P1, max(loss_depths))
                forced_cache[node_id] = result
                return result
            result = (0, None)
            forced_cache[node_id] = result
            return result

        def forced_winner(node_id):
            winner, depth = forced_outcome(node_id)
            if winner in (P1, P2):
                if depth is None:
                    return 0
                if max_lookahead is not None and depth > max_lookahead:
                    return 0
            return winner

        def forced_distance(node_id):
            _, depth = forced_outcome(node_id)
            return depth

        return downstream_winners, forced_winner, forced_distance, forced_outcome

    if exclude_wins:
        _downstream_base, forced_base, forced_distance_base, _forced_outcome_base = make_evaluators(base_child_map)

        guaranteed_nodes = set()
        for node_id, node in nodes.items():
            node.pop("forced_outcome_cached", None)
            if node["winner"] in (P1, P2):
                guaranteed_nodes.add(node_id)
                node["forced_outcome_cached"] = (node["winner"], 0)
                continue
            outcome = forced_base(node_id)
            if outcome in (P1, P2):
                guaranteed_nodes.add(node_id)
                depth = forced_distance_base(node_id)
                node["forced_outcome_cached"] = (outcome, depth if depth is not None else 0)

        nodes_to_remove = set()

        def mark_descendants(node_id):
            for child_id in base_child_map.get(node_id, []):
                if child_id not in nodes:
                    continue
                if child_id in nodes_to_remove:
                    continue
                nodes_to_remove.add(child_id)
                mark_descendants(child_id)

        for node_id in guaranteed_nodes:
            mark_descendants(node_id)

        if nodes_to_remove:
            for node_id in nodes_to_remove:
                nodes.pop(node_id, None)
                truncated_nodes.pop(node_id, None)

            truncated_ids = set(truncated_nodes.keys())

            for node in nodes.values():
                node["children"] = [cid for cid in node["children"] if cid not in nodes_to_remove]
                node["child_layers"] = {
                    cid: lay for cid, lay in node["child_layers"].items() if cid not in nodes_to_remove
                }
                node["child_columns"] = {
                    cid: cols for cid, cols in node["child_columns"].items() if cid not in nodes_to_remove
                }
                node["incoming_columns"] = {
                    pid: cols for pid, cols in node["incoming_columns"].items() if pid not in nodes_to_remove
                }
                node["parents"] = [pid for pid in node["parents"] if pid not in nodes_to_remove]

            for layer, row in list(layers.items()):
                filtered_row = [nid for nid in row if nid not in nodes_to_remove]
                if filtered_row:
                    layers[layer] = filtered_row
                else:
                    del layers[layer]

            base_child_map = defaultdict(list)
            for parent_id, node in nodes.items():
                for child_id in node["children"]:
                    if child_id in nodes:
                        base_child_map[parent_id].append(child_id)

            if 0 in nodes and 0 not in layers:
                layers[0] = [0]

            ordered_layers = sorted(layers.keys())

    _, _, _, forced_outcome_base = make_evaluators(base_child_map)

    forced_outcomes = {nid: forced_outcome_base(nid) for nid in nodes.keys()}

    def player_to_move_base(node_id):
        node = nodes.get(node_id)
        if node is None:
            return None
        last_player = node.get("player")
        if last_player is None:
            return P1
        return P1 if last_player == P2 else P2

    def detach_child(parent_id, child_id):
        parent = nodes.get(parent_id)
        if parent is None:
            return
        if child_id in parent["children"]:
            parent["children"] = [cid for cid in parent["children"] if cid != child_id]
        parent["child_layers"].pop(child_id, None)
        parent["child_columns"].pop(child_id, None)
        if parent_id in base_child_map:
            base_child_map[parent_id] = [cid for cid in base_child_map[parent_id] if cid != child_id]
            if not base_child_map[parent_id]:
                base_child_map.pop(parent_id, None)
        child = nodes.get(child_id)
        if child is None:
            return
        child["parents"] = [pid for pid in child["parents"] if pid != parent_id]
        child["incoming_columns"].pop(parent_id, None)

    def prune_orphan(start_id):
        stack = [start_id]
        while stack:
            node_id = stack.pop()
            node = nodes.get(node_id)
            if node is None or node_id == 0 or node["parents"]:
                continue
            layer = node.get("layer")
            if layer in layers:
                layers[layer] = [nid for nid in layers[layer] if nid != node_id]
                if not layers[layer]:
                    del layers[layer]
            truncated_nodes.pop(node_id, None)
            base_child_map.pop(node_id, None)
            children = list(node["children"])
            for child_id in children:
                child = nodes.get(child_id)
                if child is None:
                    continue
                child["parents"] = [pid for pid in child["parents"] if pid != node_id]
                child["incoming_columns"].pop(node_id, None)
            nodes.pop(node_id, None)
            for child_id in children:
                stack.append(child_id)

    visited_shortest = set()

    def prune_shortest_win(node_id):
        if node_id in visited_shortest:
            return
        visited_shortest.add(node_id)
        node = nodes.get(node_id)
        if node is None or node_id in truncated_nodes:
            return
        outcome = forced_outcomes.get(node_id)
        if not outcome:
            return
        winner, depth = outcome
        if winner not in (P1, P2) or depth is None:
            return
        children = list(node["children"])
        if not children:
            return
        mover = player_to_move_base(node_id)
        if mover != winner:
            for child_id in children:
                prune_shortest_win(child_id)
            return
        best_children = []
        best_total = None
        for child_id in children:
            child_outcome = forced_outcomes.get(child_id)
            if not child_outcome:
                continue
            child_winner, child_depth = child_outcome
            if child_winner != winner or child_depth is None:
                continue
            total_depth = child_depth + 1
            if best_total is None or total_depth < best_total:
                best_total = total_depth
                best_children = [child_id]
            elif total_depth == best_total:
                best_children.append(child_id)
        if not best_children:
            return
        to_remove = [cid for cid in children if cid not in best_children]
        for child_id in to_remove:
            detach_child(node_id, child_id)
            prune_orphan(child_id)
        for child_id in best_children:
            prune_shortest_win(child_id)

    for node_id, (winner, depth) in list(forced_outcomes.items()):
        if winner in (P1, P2) and depth is not None:
            prune_shortest_win(node_id)

    truncated_ids = set(truncated_nodes.keys())
    ordered_layers = sorted(layers.keys())

    edges = defaultdict(set)

    for layer in ordered_layers:
        if layer == 0 or layer == 1:
            continue
        prev_nodes = layers.get(layer - 1, [])
        current_nodes = set(layers.get(layer, []))
        for parent_id in prev_nodes:
            if parent_id in truncated_ids:
                continue
            parent_node = nodes[parent_id]
            for child_id in parent_node["children"]:
                if parent_node["child_layers"].get(child_id) != layer:
                    continue
                if child_id not in current_nodes:
                    continue
                child_node = nodes.get(child_id)
                if child_node is None:
                    continue
                mover = child_node["player"]
                edges[layer].add((parent_id, child_id, mover))

    child_map = defaultdict(list)
    for layer, e_set in edges.items():
        for parent_id, child_id, _ in e_set:
            child_map[parent_id].append(child_id)

    downstream_initial, forced_initial, _forced_distance_initial, _forced_outcome_initial = make_evaluators(child_map)

    def mover_for(node_id):
        last = nodes[node_id]["player"]
        if last is None:
            return P1
        return P1 if last == P2 else P2

    forced_children_map = defaultdict(list)
    for parent_id, children in child_map.items():
        mover = mover_for(parent_id)
        if parent_id in truncated_ids:
            forced_children_map[parent_id] = list(children)
            continue
        if nodes[parent_id]["winner"] in (P1, P2):
            forced_children_map[parent_id] = list(children)
            continue
        if forced_initial(parent_id) == mover:
            kept = [cid for cid in children if forced_initial(cid) == mover]
            if kept:
                forced_children_map[parent_id] = kept
                continue
        forced_children_map[parent_id] = list(children)

    child_map = defaultdict(list, {pid: list(children) for pid, children in forced_children_map.items()})

    filtered_edges = defaultdict(set)
    for layer, e_set in edges.items():
        for parent_id, child_id, mover in e_set:
            if child_id in child_map.get(parent_id, []):
                filtered_edges[layer].add((parent_id, child_id, mover))
    edges = filtered_edges

    downstream_winners, forced_winner, forced_distance, _forced_outcome = make_evaluators(child_map)

    edge_layers = sorted(edges.keys())
    if edge_layers:
        first_edge_layer = min(edge_layers)
        if first_edge_layer > 0:
            parent_row = sorted({p for (p, _, _) in edges[first_edge_layer]})
            layers[first_edge_layer - 1] = parent_row
        for layer in edge_layers:
            prev_set = set(layers.get(layer - 1, []))
            ordered_children = []
            for parent_id, child_id, _ in sorted(edges[layer], key=lambda e: (e[0], e[1])):
                if parent_id in prev_set:
                    ordered_children.append(child_id)
            seen = set()
            unique_children = []
            for child_id in ordered_children:
                if child_id not in seen:
                    seen.add(child_id)
                    unique_children.append(child_id)
            layers[layer] = unique_children

    ordered_layers = sorted(layers.keys())
    ordered_layers = ordered_layers[1:]

    if not ordered_layers:
        positions = {}
    else:
        positions = {}
        x_gap = 6.5
        y_gap = 7.0
        first_layer = ordered_layers[0]
        for layer in ordered_layers:
            row = layers[layer]
            row_width = (len(row) - 1) * x_gap
            for idx, node_id in enumerate(row):
                positions[node_id] = (idx * x_gap - row_width / 2, -(layer - first_layer) * y_gap)

    def has_orange_below(node_id, memo=None):
        if memo is None:
            memo = {}
        if node_id in memo:
            return memo[node_id]
        if node_id in truncated_ids:
            memo[node_id] = True
            return True
        for child_id in child_map.get(node_id, []):
            if has_orange_below(child_id, memo):
                memo[node_id] = True
                return True
        memo[node_id] = False
        return False

    RED = '#e63946'
    YEL = '#f4d03f'
    ORG = '#ff8c3a'
    DEF = '#AAB4CE'

    def border_color(node_id):
        if node_id in truncated_ids:
            return ORG
        node = nodes[node_id]
        if node["winner"] in (P1, P2):
            return RED if node["winner"] == P1 else YEL
        forced = forced_winner(node_id)
        if forced == P1:
            return RED
        if forced == P2:
            return YEL
        if not has_orange_below(node_id):
            wins = downstream_winners(node_id)
            if wins == {P1}:
                return RED
            if wins == {P2}:
                return YEL
        return DEF

    def background_color(node_id):
        node = nodes[node_id]
        if node["winner"] == P1:
            return '#500000'
        if node["winner"] == P2:
            return '#5c5a00'
        return '#1f2a44'

    def state_label(node_id):
        node = nodes[node_id]
        if node["winner"] == P1:
            return "terminal win for Player 1"
        if node["winner"] == P2:
            return "terminal win for Player 2"
        if node_id in truncated_ids:
            return "trimmed alternative branch"
        forced = forced_winner(node_id)
        if forced == P1:
            return "forced path leading to Player 1 victory"
        if forced == P2:
            return "forced path leading to Player 2 victory"
        if not has_orange_below(node_id):
            wins = downstream_winners(node_id)
            if wins == {P1}:
                return "forced path leading to Player 1 victory"
            if wins == {P2}:
                return "forced path leading to Player 2 victory"
        if child_map.get(node_id):
            return "ongoing position"
        return "dead end after pruning"

    drawn_children_by_parent = defaultdict(list)
    for layer, e_set in edges.items():
        plotted_children = set(layers.get(layer, []))
        for parent_id, child_id, _ in e_set:
            if child_id in plotted_children:
                drawn_children_by_parent[parent_id].append(child_id)
    for parent_id, kids in drawn_children_by_parent.items():
        kids.sort()

    json_payload = {
        "turns": turns,
        "max_lookahead": max_lookahead,
        "pdf_path": out_pdf,
        "nodes": [],
        "edges": [],
    }

    printed = False
    for layer in ordered_layers:
        for node_id in layers[layer]:
            node = nodes[node_id]
            drawn_children = sorted(drawn_children_by_parent.get(node_id, []))
            forced_outcome = forced_winner(node_id)
            forced_turns = forced_distance(node_id) if forced_outcome in (P1, P2) else None
            json_payload["nodes"].append({
                "id": node_id,
                "layer": layer,
                "board": node["board"],
                "status": state_label(node_id),
                "border_color": border_color(node_id),
                "background_color": background_color(node_id),
                "forced_winner": forced_outcome,
                "forced_turns": forced_turns,
                "parent_id": node["parents"][0] if node["parents"] else None,
                "parent_ids": node["parents"],
                "entered_by_player": node["player"],
                "columns_from_parents": node["incoming_columns"],
                "children_ids": sorted(set(child_map.get(node_id, []))),
                "drawn_children_ids": drawn_children,
                "position": {"x": positions[node_id][0], "y": positions[node_id][1]},
                "has_orange_below": has_orange_below(node_id),
                "downstream_winners": sorted(list(downstream_winners(node_id))),
            })
            if not printed and not disable_progress:
                print(f"\nfirst node state: {state_label(node_id)}\n")
                printed = True

    for layer, e_set in edges.items():
        for parent_id, child_id, mover in sorted(e_set, key=lambda e: (e[0], e[1])):
            if parent_id not in positions or child_id not in positions:
                continue
            json_payload["edges"].append({
                "from_id": parent_id,
                "to_id": child_id,
                "layer": layer,
                "mover": mover,
                "columns": nodes[parent_id]["child_columns"].get(child_id, []),
            })

    with open(out_json, "w") as fh:
        json.dump(json_payload, fh, indent=2)

    if plot_pdf:
        print("Rendering PDF...")
        assert out_pdf is not None
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, RegularPolygon

        n_rows = len(ordered_layers)
        max_row_nodes = max((len(layers[layer]) for layer in ordered_layers), default=1)
        fig_w = max(12, 2 + max_row_nodes * 1.6)
        fig_h = max(10, 3 + n_rows * 2.8)
        dpi = 110 + 28 * n_rows
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        ax = plt.gca()
        ax.set_facecolor('#0b1020')
        fig.set_facecolor('#0b1020')

        node_box = {}
        minx = miny = float('inf')
        maxx = maxy = float('-inf')
        cell_scale = 0.45

        for layer, row in layers.items():
            for node_id in row:
                if border_color(node_id) != DEF:
                    continue
                next_layer = layer + 1
                drawn_children = {
                    child_id for (parent_id, child_id, _) in edges.get(next_layer, set())
                    if parent_id == node_id
                }
                if not drawn_children:
                    continue
                expected_children = {
                    child_id for child_id in child_map.get(node_id, [])
                    if nodes[node_id]["child_layers"].get(child_id) == next_layer
                }
                assert drawn_children == expected_children, (
                    f"Node {node_id} has {len(drawn_children)} children drawn but "
                    f"{len(expected_children)} possible."
                )

        for layer in ordered_layers:
            for node_id in layers[layer]:
                node = nodes[node_id]
                x, y = positions[node_id]
                board = node["board"]
                rows, cols = len(board), len(board[0])

                width = cols * cell_scale + 0.8
                height = rows * cell_scale + 0.8
                rect = Rectangle(
                    (x - width / 2, y - height / 2),
                    width,
                    height,
                    facecolor=background_color(node_id),
                    edgecolor=border_color(node_id),
                    linewidth=2.2,
                    zorder=2,
                )
                ax.add_patch(rect)

                pad = 0.25
                gx0, gy0 = x - width / 2 + pad, y - height / 2 + pad
                gw, gh = width - 2 * pad, height - 2 * pad
                cell_w, cell_h = gw / cols, gh / rows
                disc_radius = 0.35 * min(cell_w, cell_h)
                for row_idx in range(rows):
                    for col_idx in range(cols):
                        value = board[row_idx][col_idx]
                        face = (0.23, 0.28, 0.43) if value == 0 else (
                            (0.90, 0.15, 0.2) if value == 1 else (0.96, 0.82, 0.25)
                        )
                        cx = gx0 + col_idx * cell_w + cell_w / 2
                        cy = gy0 + row_idx * cell_h + cell_h / 2
                        ax.add_patch(
                            RegularPolygon(
                                (cx, cy),
                                numVertices=DISC_RESOLUTION,
                                radius=disc_radius,
                                orientation=0,
                                facecolor=face,
                                edgecolor='black',
                                linewidth=0.6,
                                zorder=3,
                            )
                        )

                ax.text(
                    x,
                    y,
                    str(node_id),
                    color='white',
                    fontsize=14,
                    ha='center',
                    va='center',
                    alpha=0.4,
                    zorder=6,
                )

                node_box[node_id] = (width, height)
                minx = min(minx, x - width / 2 - 1)
                maxx = max(maxx, x + width / 2 + 1)
                miny = min(miny, y - height / 2 - 1)
                maxy = max(maxy, y + height / 2 + 1)

        for layer, e_set in edges.items():
            for parent_id, child_id, mover in e_set:
                if (
                    layer not in layers
                    or parent_id not in layers.get(layer - 1, [])
                    or child_id not in layers.get(layer, [])
                ):
                    continue
                x1, y1 = positions[parent_id]
                width1, height1 = node_box[parent_id]
                x2, y2 = positions[child_id]
                width2, height2 = node_box[child_id]
                parent_bottom = (x1, y1 - height1 / 2)
                child_top = (x2, y2 + height2 / 2)
                colour = RED if mover == P1 else YEL
                ax.plot(
                    [parent_bottom[0], child_top[0]],
                    [parent_bottom[1], child_top[1]],
                    linewidth=1.6,
                    color=colour,
                    alpha=0.95,
                    zorder=1,
                )

        if (
            not math.isfinite(minx)
            or not math.isfinite(maxx)
            or not math.isfinite(miny)
            or not math.isfinite(maxy)
        ):
            ax.autoscale()
        else:
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        if exclude_wins:
            if max_lookahead is None:
                title = f"{turns} turns — truncate paths to no wins within {turns} turns or guaranteed wins"
            else:
                title = (
                    f"{turns} turns — truncate paths to no wins within {turns} turns; "
                    f"forced pruning ≤ {max_lookahead} turns"
                )
        else:
            if max_lookahead is None:
                title = f"{turns} turns — truncate paths to no wins within {turns} turns"
            else:
                title = (
                    f"{turns} turns — truncate paths to no wins within {turns} turns; "
                    f"forced colouring ≤ {max_lookahead} turns"
                )
        ax.set_title(title, color='white', pad=12)

        fig.savefig(out_pdf, bbox_inches='tight')
        plt.close(fig)

    return out_json, out_pdf

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    import argparse

    def non_negative_int(value: str) -> int:
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError("--max-lookahead must be a non-negative integer")
        return ivalue

    parser = argparse.ArgumentParser(description="Generate pruned Four-in-a-Roll state graphs.")
    default_rows, default_cols = INIT_ROWS, INIT_COLS
    parser.add_argument("turns", type=int, help="number of turns to explore")
    parser.add_argument(
        "--plot-pdf",
        action="store_true",
        help="render the matplotlib visualization alongside the JSON payload",
    )
    parser.add_argument(
        "--exclude-wins",
        action="store_true",
        help="omit nodes that are already wins or have a forced win outcome from the outputs",
    )
    parser.add_argument(
        "--max-lookahead",
        type=non_negative_int,
        default=None,
        help="only apply forced-win colouring and pruning to nodes within this many turns of a guaranteed win",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help=f"override the starting board row count (requires --cols as well; default: {default_rows})",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help=f"override the starting board column count (requires --rows as well; default: {default_cols})",
    )
    args = parser.parse_args()

    if (args.rows is None) != (args.cols is None):
        parser.error("You must supply both --rows and --cols, or neither.")

    if args.rows is not None and args.cols is not None:
        INIT_ROWS, INIT_COLS = args.rows, args.cols
    else:
        INIT_ROWS, INIT_COLS = default_rows, default_cols

    json_path, pdf_path = render_full_logic(
        args.turns,
        plot_pdf=args.plot_pdf,
        exclude_wins=args.exclude_wins,
        max_lookahead=args.max_lookahead,
    )
    if args.plot_pdf:
        print("PDF:", pdf_path)
    else:
        print("PDF: (skipped, pass --plot-pdf to generate)")
    print("JSON:", json_path)
