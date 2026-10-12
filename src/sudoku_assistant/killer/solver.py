"""Killer Sudoku: exact cage sums, no cage repeats, and unique-solution search."""
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import time

FULL = 0x3FE
HOUSES = tuple(tuple(r*9+c for c in range(9)) for r in range(9)) + \
         tuple(tuple(r*9+c for r in range(9)) for c in range(9)) + \
         tuple(tuple((br+r)*9+bc+c for r in range(3) for c in range(3))
               for br in (0,3,6) for bc in (0,3,6))


class PuzzleError(ValueError):
    pass


@dataclass(frozen=True)
class Cage:
    total: int
    cells: tuple[int, ...]


@lru_cache(maxsize=512)
def combination_masks(size, total):
    return tuple(sum(1 << n for n in digits) for digits in combinations(range(1,10), size)
                 if sum(digits) == total)


def validate_cages(cages):
    coverage = []
    for cage in cages:
        if type(cage.total) is not int or not 1 <= len(cage.cells) <= 9:
            raise PuzzleError("每个虚线框需要整数总和及 1–9 个格子。")
        if any(type(i) is not int or not 0 <= i < 81 for i in cage.cells):
            raise PuzzleError("虚线框的格子编号超出棋盘。")
        if len(set(cage.cells)) != len(cage.cells):
            raise PuzzleError("虚线框内存在重复格子。")
        cells, visited, pending = set(cage.cells), set(), [cage.cells[0]]
        while pending:
            i = pending.pop()
            if i in visited:
                continue
            visited.add(i)
            r,c = divmod(i,9)
            pending.extend(rr*9+cc for rr,cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
                           if 0 <= rr < 9 and 0 <= cc < 9 and rr*9+cc in cells)
        if visited != cells:
            raise PuzzleError("虚线框的格子不连通，请检查边界识别。")
        if not combination_masks(len(cage.cells), cage.total):
            raise PuzzleError(f"总和 {cage.total} 无法由 {len(cage.cells)} 个不重复的数字组成。")
        coverage.extend(cage.cells)
    if sorted(coverage) != list(range(81)):
        raise PuzzleError("所有虚线框必须恰好覆盖 81 格，不能遗漏或重叠。")
    if sum(c.total for c in cages) != 405:
        raise PuzzleError("总和角标之和必须为 405，请检查总和及虚线边界识别。")


@lru_cache(maxsize=48000)
def cage_support(domains, total):
    """Supported cell digits using subset matching, without enumerating permutations."""
    support = [0]*len(domains)
    for combo in combination_masks(len(domains), total):
        choices = tuple(d & combo for d in domains)
        if not all(choices):
            continue
        layers = [{0}]
        for allowed in choices:
            following = set()
            for used in layers[-1]:
                available = allowed & ~used
                while available:
                    bit = available & -available
                    available -= bit
                    following.add(used | bit)
            layers.append(following)
            if not following:
                break
        if len(layers) != len(domains)+1 or combo not in layers[-1]:
            continue
        viable = {combo}
        for pos in range(len(domains)-1,-1,-1):
            previous = set()
            for used in layers[pos]:
                available = choices[pos] & ~used
                while available:
                    bit = available & -available
                    available -= bit
                    if used | bit in viable:
                        support[pos] |= bit
                        previous.add(used)
            viable = previous
    return tuple(support)


def validate_solution(board, cages):
    if len(board) != 81 or any(type(n) is not int or not 1 <= n <= 9 for n in board):
        raise PuzzleError("解必须包含 81 个 1–9 的数字。")
    if any({board[i] for i in h} != set(range(1,10)) for h in HOUSES):
        raise PuzzleError("解不满足标准数独的行、列及九宫约束。")
    for cage in cages:
        values = [board[i] for i in cage.cells]
        if sum(values) != cage.total or len(set(values)) != len(values):
            raise PuzzleError("解不满足虚线框的总和或不重复约束。")


@lru_cache(maxsize=48000)
def sum_support(domains, total):
    """Exact reachable sums as integer bitsets, allowing repeated digits."""
    values=tuple(tuple(n for n in range(1,10) if d & (1 << n)) for d in domains)
    if any(not digits for digits in values) or total<sum(min(digits) for digits in values) or total>sum(max(digits) for digits in values):
        return (0,)*len(domains)
    limit=(1 << (total+1))-1
    prefix=[1]
    for digits in values:
        reachable=0
        for n in digits:
            reachable |= prefix[-1] << n
        prefix.append(reachable & limit)
    if not prefix[-1] & (1 << total):
        return (0,)*len(domains)
    suffix=[0]*(len(domains)+1)
    suffix[-1]=1
    for i in range(len(domains)-1,-1,-1):
        reachable=0
        for n in values[i]:
            reachable |= suffix[i+1] << n
        suffix[i]=reachable & limit
    support=[]
    for i,digits in enumerate(values):
        # Reverse suffix sums around the target. Shift by a candidate digit,
        # then one intersection checks every compatible prefix sum at once.
        reflected=int(bin(suffix[i+1])[2:].zfill(total+1)[::-1],2)
        support.append(sum(1 << n for n in digits if prefix[i] & (reflected >> n)))
    return tuple(support)


def house_sum_constraints(cages):
    constraints=set()
    for house in HOUSES:
        cells=set(house)
        contained=[c for c in cages if set(c.cells) <= cells]
        remainder=cells-set(i for c in contained for i in c.cells)
        if 1 <= len(remainder) <= 7:
            constraints.add((tuple(sorted(remainder)),45-sum(c.total for c in contained)))
        touching=[c for c in cages if cells.intersection(c.cells)]
        outside=set(i for c in touching for i in c.cells)-cells
        if 1 <= len(outside) <= 7:
            constraints.add((tuple(sorted(outside)),sum(c.total for c in touching)-45))
    return tuple(sorted(constraints))


def solve_unique(board, cages, check=None, node_limit=200000, timeout=45, stats=None):
    validate_cages(cages)
    if len(board) != 81 or any(type(n) is not int or not 0 <= n <= 9 for n in board):
        raise PuzzleError("棋盘必须包含 81 个 0–9 的数字。")
    peers = [set() for _ in range(81)]
    for house in HOUSES + tuple(c.cells for c in cages):
        for i in house:
            peers[i].update(j for j in house if j != i)
    for i,n in enumerate(board):
        if n and any(board[j] == n for j in peers[i]):
            r,c = divmod(i,9)
            raise PuzzleError(f"第 {r+1} 行第 {c+1} 列在行、列、九宫或虚线框内发生数字冲突。")
    nodes, answers, started = 0, [], time.monotonic()
    derived_sums=house_sum_constraints(cages)

    def guard():
        if check:
            check()
        if nodes > node_limit or time.monotonic()-started >= timeout:
            raise PuzzleError("求解超过上限，已停止点击；请检查识别结果。")

    def propagate(domains):
        propagated=[0]*81
        while True:
            guard()
            changed = False
            for i,d in enumerate(domains):
                if not d:
                    return False
                if d.bit_count() == 1 and propagated[i]!=d:
                    propagated[i]=d
                    for j in peers[i]:
                        if domains[j] & d:
                            domains[j] &= ~d
                            changed = True
                            if not domains[j]:
                                return False
            for house in HOUSES:
                seen,repeated=0,0
                for i in house:
                    repeated |= seen & domains[i]
                    seen |= domains[i]
                if seen!=FULL:
                    return False
                unique=seen & ~repeated
                if not unique:
                    continue
                for i in house:
                    forced=domains[i] & unique
                    if forced and forced.bit_count()!=1:
                        return False
                    if forced and domains[i]!=forced:
                        domains[i]=forced
                        changed = True
            for cage in cages:
                supported = cage_support(tuple(domains[i] for i in cage.cells), cage.total)
                for i,d in zip(cage.cells, supported):
                    if not d:
                        return False
                    if domains[i] != d:
                        domains[i] = d
                        changed = True
            for cells,total in derived_sums:
                supported=sum_support(tuple(domains[i] for i in cells),total)
                for i,d in zip(cells,supported):
                    if not d:
                        return False
                    if domains[i] != d:
                        domains[i] = d
                        changed = True
            if not changed:
                for house in HOUSES:
                    counts={}
                    for i in house:
                        d=domains[i]
                        if 2<=d.bit_count()<=3:
                            counts[d]=counts.get(d,0)+1
                    for mask,count in counts.items():
                        width=mask.bit_count()
                        if count>width:
                            return False
                        if count!=width:
                            continue
                        # Identical pairs/triples occupy these digits in
                        # exactly these cells, so every other cell excludes them.
                        for i in house:
                            d=domains[i]
                            if d==mask:
                                continue
                            restricted=d & ~mask
                            if not restricted:
                                return False
                            if restricted!=d:
                                domains[i]=restricted
                                changed=True
                if not changed:
                    return True

    def search(domains):
        nonlocal nodes
        nodes += 1
        guard()
        if not propagate(domains):
            return
        undecided = [i for i,d in enumerate(domains) if d.bit_count() > 1]
        if not undecided:
            answer = [d.bit_length()-1 for d in domains]
            validate_solution(answer,cages)
            answers.append(answer)
            return
        i = min(undecided,key=lambda i:(domains[i].bit_count(),-len(peers[i])))
        allowed = domains[i]
        while allowed and len(answers) < 2:
            bit = allowed & -allowed
            allowed -= bit
            following = domains.copy()
            following[i] = bit
            search(following)

    search([1 << n if n else FULL for n in board])
    if stats is not None:
        stats.update(nodes=nodes,solutions=len(answers),seconds=time.monotonic()-started)
    if not answers:
        raise PuzzleError("棋盘无解，已停止点击；请检查数字、虚线框和总和角标。")
    if len(answers) != 1:
        raise PuzzleError("棋盘存在多个解，已停止点击；请检查识别是否遗漏约束。")
    return answers[0]
