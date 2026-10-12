"""Unique Jigsaw Sudoku solver; regions come from the screenshot, never 3x3 boxes."""
from collections import Counter


class PuzzleError(ValueError):
    pass


def validate_regions(regions):
    if len(regions) != 81 or any(type(v) is not int or not 0 <= v < 9 for v in regions):
        raise PuzzleError("异形宫必须包含 81 个 0–8 的宫编号。")
    if Counter(regions) != Counter({n: 9 for n in range(9)}):
        raise PuzzleError("必须识别出 9 个异形宫，每宫恰好 9 格。")
    for region in range(9):
        cells = {i for i, n in enumerate(regions) if n == region}
        visited, pending = set(), [min(cells)]
        while pending:
            i = pending.pop()
            if i in visited:
                continue
            visited.add(i)
            r, c = divmod(i, 9)
            neighbors = [rr * 9 + cc for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
                         if 0 <= rr < 9 and 0 <= cc < 9]
            pending.extend(j for j in neighbors if j in cells and j not in visited)
        if visited != cells:
            raise PuzzleError(f"第 {region + 1} 个异形宫不连通，请检查边界识别。")


def solve_unique(board, regions, check=None, node_limit=1000000, stats=None):
    validate_regions(regions)
    if len(board) != 81 or any(type(v) is not int or not 0 <= v <= 9 for v in board):
        raise PuzzleError("棋盘必须包含 81 个 0–9 的数字。")
    rows, cols, groups = [0]*9, [0]*9, [0]*9
    answers, nodes = [], 0
    for i, value in enumerate(board):
        if not value:
            continue
        r, c = divmod(i, 9)
        group, bit = regions[i], 1 << value
        if (rows[r] | cols[c] | groups[group]) & bit:
            raise PuzzleError(f"第 {r+1} 行第 {c+1} 列在行、列或异形宫内发生数字冲突。")
        rows[r] |= bit
        cols[c] |= bit
        groups[group] |= bit

    # Exact cover: each candidate satisfies cell, row-digit, column-digit and
    # irregular-palace-digit constraints. Choosing the most constrained column
    # propagates hidden singles too, unlike choosing only the emptiest cell.
    column_rows=[0]*324
    candidate_columns=[]
    candidate_column_masks=[]
    for i in range(81):
        r,c=divmod(i,9)
        for digit in range(9):
            candidate=i*9+digit
            columns=(i,81+r*9+digit,162+c*9+digit,243+regions[i]*9+digit)
            candidate_columns.append(columns)
            candidate_column_masks.append(sum(1<<column for column in columns))
            for column in columns:
                column_rows[column]|=1<<candidate
    conflicts=[]
    for columns in candidate_columns:
        conflicts.append(column_rows[columns[0]]|column_rows[columns[1]]|
                         column_rows[columns[2]]|column_rows[columns[3]])
    active_rows=(1<<729)-1
    active_columns=(1<<324)-1
    selected=[]
    for i,value in enumerate(board):
        if value:
            candidate=i*9+value-1
            selected.append(candidate)
            active_rows &= ~conflicts[candidate]
            active_columns &= ~candidate_column_masks[candidate]

    def search(remaining_rows,remaining_columns):
        nonlocal nodes
        nodes += 1
        if check and nodes % 256 == 1:
            check()
        if nodes > node_limit:
            raise PuzzleError("求解搜索达到上限，题目可能过于复杂，已停止点击。")
        if not remaining_columns:
            answer=[0]*81
            for candidate in selected:
                i,digit=divmod(candidate,9)
                answer[i]=digit+1
            answers.append(answer)
            return
        choices,count=0,10
        pending=remaining_columns
        while pending:
            bit=pending & -pending
            pending-=bit
            column=bit.bit_length()-1
            available=column_rows[column] & remaining_rows
            size=available.bit_count()
            if not size:
                return
            if size < count:
                choices,count=available,size
                if count == 1:
                    break
        while choices and len(answers) < 2:
            bit = choices & -choices
            choices -= bit
            candidate=bit.bit_length()-1
            selected.append(candidate)
            search(remaining_rows & ~conflicts[candidate],
                   remaining_columns & ~candidate_column_masks[candidate])
            selected.pop()

    search(active_rows,active_columns)
    if stats is not None:
        stats.update(nodes=nodes,solutions=len(answers))
    if not answers:
        raise PuzzleError("棋盘无解，已停止点击；请检查数字及异形宫识别。")
    if len(answers) != 1:
        raise PuzzleError("棋盘存在多个解，已停止点击；可能有数字漏识别。")
    return answers[0]
