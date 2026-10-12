"""Sudoku validation and unique solving. No game state is changed here."""


class PuzzleError(ValueError):
    pass


def solve_unique(board, check=None, node_limit=300000):
    if len(board) != 81 or any(type(v) is not int or not 0 <= v <= 9 for v in board):
        raise PuzzleError("棋盘必须包含 81 个 0–9 的数字。")
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    work = list(board)
    for i, n in enumerate(work):
        if not n:
            continue
        r, c = divmod(i, 9)
        b, bit = r // 3 * 3 + c // 3, 1 << n
        if (rows[r] | cols[c] | boxes[b]) & bit:
            raise PuzzleError(f"第 {r + 1} 行第 {c + 1} 列与其他数字冲突；请检查截图或已有填写。")
        rows[r] |= bit
        cols[c] |= bit
        boxes[b] |= bit
    answers = []
    nodes = 0

    def search():
        nonlocal nodes
        nodes += 1
        if check and nodes % 256 == 1:
            check()
        if nodes > node_limit:
            raise PuzzleError("求解超过上限，请确认棋盘识别是否完整。")
        best, candidates, count = -1, 0, 10
        for i, value in enumerate(work):
            if value:
                continue
            r, c = divmod(i, 9)
            allowed = 0x3FE & ~(rows[r] | cols[c] | boxes[r // 3 * 3 + c // 3])
            size = allowed.bit_count()
            if not size:
                return
            if size < count:
                best, candidates, count = i, allowed, size
                if size == 1:
                    break
        if best < 0:
            answers.append(work.copy())
            return
        r, c = divmod(best, 9)
        b = r // 3 * 3 + c // 3
        while candidates and len(answers) < 2:
            bit = candidates & -candidates
            candidates -= bit
            work[best] = bit.bit_length() - 1
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            search()
            rows[r] ^= bit
            cols[c] ^= bit
            boxes[b] ^= bit
            work[best] = 0

    search()
    if not answers:
        raise PuzzleError("棋盘无解，已停止点击；请检查识别结果或错误填写。")
    if len(answers) > 1:
        raise PuzzleError("棋盘存在多个解，已停止点击；可能有数字漏识别。")
    return answers[0]
