"""Side-by-side picker that pairs a list of target files with a list of reference files.

Both panes list the files. Each pick is numbered in the order it was made; pair N is the Nth pick on the left with the Nth
pick on the right. For a season whose files sort by episode, `A` (select all) in both panes pairs everything in a handful of key presses (a filter such as ".pt." narrows a pane first).
"""
import curses
import re


def natural_key(name):
    """Sort key that orders E2 before E10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def shorten(text, width):
    """Cuts a long name in the middle so both ends (a title and an episode code) stay visible."""
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    keep = width - 3
    head = (keep + 1) // 2
    return text[:head] + "..." + text[len(text) - (keep - head):]


class PairingState:
    """Selection logic of the picker, kept apart from curses so it can be tested on its own."""

    def __init__(self, left_names, right_names):
        self.names = (list(left_names), list(right_names))
        self.picks = ([], [])           # per pane: item indices in the order they were picked
        self.filters = ["", ""]         # per pane: only names containing this text are shown
        self.cursor = [0, 0]            # per pane: position in the shown list
        self.active = 0

    def visible(self, pane):
        """Indices of the items shown in a pane (those matching its filter), in listing order."""
        needle = self.filters[pane].lower()
        return [i for i, name in enumerate(self.names[pane]) if needle in name.lower()]

    def set_filter(self, pane, text):
        self.filters[pane] = text
        self.cursor[pane] = 0

    def number(self, pane, index):
        """1-based pick number of an item, or None when it is not picked."""
        try:
            return self.picks[pane].index(index) + 1
        except ValueError:
            return None

    def toggle(self, pane=None, index=None):
        """Picks or unpicks an item: `index` is an item index, by default the one under the cursor."""
        pane = self.active if pane is None else pane
        if index is None:
            shown = self.visible(pane)
            if not shown:
                return
            index = shown[self.cursor[pane]]
        if index in self.picks[pane]:
            self.picks[pane].remove(index)          # later picks move up one number
        else:
            self.picks[pane].append(index)

    def select_all(self, pane=None):
        """Picks every shown item in listing order; when all of them are already picked, unpicks them instead."""
        pane = self.active if pane is None else pane
        shown = self.visible(pane)
        if shown and all(i in self.picks[pane] for i in shown):
            self.picks[pane][:] = [i for i in self.picks[pane] if i not in shown]
        else:
            self.picks[pane].extend(i for i in shown if i not in self.picks[pane])

    def clear(self, pane=None):
        pane = self.active if pane is None else pane
        self.picks[pane].clear()

    def move(self, step):
        pane = self.active
        self.cursor[pane] = max(0, min(len(self.visible(pane)) - 1, self.cursor[pane] + step))

    def switch(self, pane):
        self.active = pane

    def problem(self):
        """Why the current picks cannot be confirmed, or None."""
        left, right = self.picks
        if not left or not right:
            return "Pick at least one file on each side"
        if len(left) != len(right):
            return f"{len(left)} on the left but {len(right)} on the right: the counts must match"
        for n, (a, b) in enumerate(zip(left, right), 1):
            if self.names[0][a] == self.names[1][b]:
                return f"Pair {n} would use {self.names[0][a]} as both target and reference"
        return None

    def pairs(self):
        """[(left_index, right_index)] in pick order."""
        return list(zip(*self.picks))


def _picker(stdscr, state, titles, files_title):
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)    # cursor
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_GREEN)   # picked
    curses.init_pair(3, curses.COLOR_WHITE, -1)                   # normal
    curses.init_pair(4, curses.COLOR_YELLOW, -1)                  # title
    curses.init_pair(5, curses.COLOR_CYAN, -1)                    # headers
    curses.init_pair(6, curses.COLOR_RED, -1)                     # problem
    offsets = [0, 0]
    message = ""
    editing = False         # typing a filter for the active pane

    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        col_w = max(10, (width - 3) // 2)
        list_h = max(1, height - 7)

        def put(y, x, text, style):
            try:
                stdscr.addstr(y, x, text[:max(0, width - x - 1)], style)
            except curses.error:
                pass

        put(0, 0, files_title, curses.color_pair(4) | curses.A_BOLD)
        put(1, 0, "[Up/Down] Move  [Left/Right] Switch side  [Space] Pick (numbered in pick order)  [/] Filter  [A] All shown  [C] Clear  [Enter] Confirm  [Q] Cancel",
            curses.color_pair(3) | curses.A_DIM)
        heads = []
        for pane in (0, 1):
            text = f"{titles[pane]} ({len(state.picks[pane])} picked)"
            if state.filters[pane] or (editing and state.active == pane):
                text += f"  filter: {state.filters[pane]}{'_' if editing and state.active == pane else ''}"
            heads.append(shorten(text, col_w))
        put(3, 0, f"{heads[0]:<{col_w}} | {heads[1]:<{col_w}}", curses.color_pair(5) | curses.A_BOLD)
        put(4, 0, "-" * (width - 1), curses.color_pair(3) | curses.A_DIM)

        for pane in (0, 1):
            shown = state.visible(pane)
            state.cursor[pane] = max(0, min(len(shown) - 1, state.cursor[pane]))
            cursor = state.cursor[pane]
            if cursor < offsets[pane]:
                offsets[pane] = cursor
            elif cursor >= offsets[pane] + list_h:
                offsets[pane] = cursor - list_h + 1
            x = 0 if pane == 0 else col_w + 3
            for row in range(list_h):
                position = offsets[pane] + row
                if position >= len(shown):
                    break
                index = shown[position]
                n = state.number(pane, index)
                badge = f"[{n:>3}] " if n else "[   ] "
                text = badge + shorten(state.names[pane][index], col_w - len(badge))
                style = curses.color_pair(2) if n else curses.color_pair(3)
                if state.active == pane and cursor == position:
                    style = curses.color_pair(1)
                put(5 + row, x, f"{text:<{col_w}}", style)
        for row in range(list_h):
            put(5 + row, col_w, " | ", curses.color_pair(3) | curses.A_DIM)

        problem = state.problem()
        status = f"{len(state.picks[0])} target(s), {len(state.picks[1])} reference(s)  ->  {len(state.pairs()) if not problem else 0} pair(s)"
        put(height - 2, 0, status, curses.color_pair(3) | curses.A_BOLD)
        put(height - 1, 0, message or (problem or "Ready: press Enter to confirm"),
            curses.color_pair(6) if (message or problem) else curses.color_pair(2))
        stdscr.refresh()

        key = stdscr.getch()
        message = ""
        if editing:
            text = state.filters[state.active]
            if key in (10, 13, curses.KEY_ENTER, 27):
                editing = False
            elif key in (curses.KEY_BACKSPACE, 127, 8):
                state.set_filter(state.active, text[:-1])
            elif 32 <= key < 127:
                state.set_filter(state.active, text + chr(key))
            continue
        if key == ord("/"):
            editing = True
        elif key == curses.KEY_UP:
            state.move(-1)
        elif key == curses.KEY_DOWN:
            state.move(1)
        elif key == curses.KEY_PPAGE:
            state.move(-list_h)
        elif key == curses.KEY_NPAGE:
            state.move(list_h)
        elif key == curses.KEY_HOME:
            state.cursor[state.active] = 0
        elif key == curses.KEY_END:
            state.cursor[state.active] = max(0, len(state.visible(state.active)) - 1)
        elif key == curses.KEY_LEFT:
            state.switch(0)
        elif key == curses.KEY_RIGHT:
            state.switch(1)
        elif key == ord(" "):
            state.toggle()
            state.move(1)           # ready for the next pick
        elif key in (ord("a"), ord("A")):
            state.select_all()
        elif key in (ord("c"), ord("C")):
            state.clear()
        elif key in (10, 13, curses.KEY_ENTER):
            problem = state.problem()
            if problem:
                message = problem
            else:
                return state.pairs()
        elif key in (ord("q"), ord("Q"), 27):
            return None


def pick_pairs(left_files, right_files, titles=("TARGETS (to be fixed)", "REFERENCES (already synced)"),
               files_title="Pick targets on the left and references on the right; the numbers show which belongs to which."):
    """Shows the picker. `left_files` / `right_files` are Paths. Returns [(left_path, right_path)] in pick order, or None."""
    left = sorted(left_files, key=lambda f: natural_key(f.name))
    right = sorted(right_files, key=lambda f: natural_key(f.name))
    state = PairingState([f.name for f in left], [f.name for f in right])
    pairs = curses.wrapper(_picker, state, titles, files_title)
    if pairs is None:
        return None
    return [(left[a], right[b]) for a, b in pairs]
