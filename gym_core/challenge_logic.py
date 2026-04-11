from __future__ import annotations
import random
import pandas as pd
from gym_core.player import PlayerID, PlayerTable
from gym_core.cards import Card
from gym_core.challenge_table import ChallengeTable
from gym_core.matchup_table import MatchupTable




# preference gen ============================
# all players passed in through PlayerTable MUST be eligible to play (cards remaining)


def _rank_opponents(table: PlayerTable, pid: PlayerID) -> list[PlayerID]:
   candidates = [oid for oid in table if oid != pid]
   return random.sample(candidates, min(3, len(candidates)))

#TODO
def _agent_rank_opponents(table: PlayerTable, pid: PlayerID) -> list[PlayerID]:
   candidates = [oid for oid in table if oid != pid]
   return random.sample(candidates, min(3, len(candidates)))


def _select_move(pid: PlayerID, table: PlayerTable) -> Card:
   available = [card for card in Card if table[pid][card.value] > 0]
   return random.choice(available)

#TODO
def _agent_select_move(pid: PlayerID, table: PlayerTable) -> Card:
   available = [card for card in Card if table[pid][card.value] > 0]
   return random.choice(available)


# table making ============================


def build_challenge_table(table: PlayerTable) -> ChallengeTable:
   rows: list[dict] = []


   for pid in table:
       card = _select_move(pid, table) if pid != 0 else _agent_select_move(pid, table)
       targets = _rank_opponents(pid, table) if pid != 0 else _agent_rank_opponents(table, pid)
       for rank, target_id in enumerate(targets):
           rows.append({
               "player_id": pid,
               "card": str(card),
               "target_id": target_id,
               "priority": rank,
           })


   df = pd.DataFrame(rows, columns=["player_id", "card", "priority", "target_id"])
   return ChallengeTable.validate(df)

def _get_card(pid: PlayerID, challenge_table: ChallengeTable) -> Card:
    row = challenge_table.loc[challenge_table["player_id"] == pid].iloc[0]
    return Card(row["card"])

def resolve_matchups(table: PlayerTable, challenge_table: ChallengeTable) -> MatchupTable:
    prefs: dict[PlayerID, list[PlayerID]] = (
        challenge_table
        .sort_values(["player_id", "priority"])   # priority col makes order guaranteed
        .groupby("player_id", sort=False)["target_id"]
        .apply(list)
        .to_dict()
    )

    all_ids = list(table.keys())
    unmatched: set[PlayerID] = set(all_ids)
    matchups: MatchupTable = {}

    # first pass - match mutual top picks
    #shuffled to avoid priority
    shuffled = all_ids.copy()
    random.shuffle(shuffled)

    for pid in shuffled:
        if pid not in unmatched:
            continue
        for candidate in prefs.get(pid, []):
            if candidate not in unmatched:
                continue
            if pid in prefs.get(candidate, []):
                matchups[(pid, candidate)] = (_get_card(pid, challenge_table), _get_card(candidate, challenge_table))
                unmatched -= {pid, candidate}
                break

    # second pass - one-sided interest
    for pid in shuffled:
        if pid not in unmatched:
            continue
        for candidate in prefs.get(pid, []):
            if candidate in unmatched:
                matchups[(pid, candidate)] = (_get_card(pid, challenge_table), _get_card(candidate, challenge_table))
                unmatched -= {pid, candidate}
                break

    # third pass - random fallback
    leftover = list(unmatched)
    random.shuffle(leftover)
    while len(leftover) >= 2:
        a = leftover.pop()
        b = leftover.pop()
        matchups[(a, b)] = (_get_card(a, challenge_table), _get_card(b, challenge_table))

    return matchups
