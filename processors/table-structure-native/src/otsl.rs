//! OTSL (Docling's table structure language) as TableFormerV2 emits it: the token vocabulary, the data-cell set, and the parse of a
//! generated token sequence into a grid of cells with spans. Pure code, no model needed.
//!
//! Vocabulary (the model's `tokenizer.json` added tokens, id order): 0 `<pad>`, 1 `[UNK]`, 2 `<start>`, 3 `<end>`, 4 `<ecel>` (empty cell),
//! 5 `<fcel>` (cell with content), 6 `<lcel>` (continues the cell to its left), 7 `<ucel>` (continues the cell above), 8 `<xcel>` (both),
//! 9 `<nl>` (end of row), 10 `<ched>` (column header cell), 11 `<rhed>` (row header cell), 12 `<srow>` (section row cell).

pub const BOS: i64 = 2;
pub const EOS: i64 = 3;
pub const NL: i64 = 9;
const LCEL: i64 = 6;
const UCEL: i64 = 7;
const XCEL: i64 = 8;

/// Tokens that start a cell and get one predicted box each (the model's `bbox_head` runs on exactly these positions, in order).
pub fn is_data_cell(t: i64) -> bool {
    matches!(t, 4 | 5 | 10 | 11 | 12)
}

pub fn name(t: i64) -> &'static str {
    match t {
        0 => "<pad>",
        1 => "[UNK]",
        2 => "<start>",
        3 => "<end>",
        4 => "<ecel>",
        5 => "<fcel>",
        6 => "<lcel>",
        7 => "<ucel>",
        8 => "<xcel>",
        9 => "<nl>",
        10 => "<ched>",
        11 => "<rhed>",
        12 => "<srow>",
        _ => "<?>",
    }
}

/// One table cell after the spans are resolved.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Cell {
    /// The cell's own token (a data-cell token).
    pub token: i64,
    pub row: usize,
    pub col: usize,
    pub rowspan: usize,
    pub colspan: usize,
    /// Index of this cell among the data-cell tokens in generation order = the index of its predicted box.
    pub box_index: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Grid {
    pub rows: usize,
    pub cols: usize,
    pub cells: Vec<Cell>,
}

/// Parse a generated sequence (the leading `<start>` and trailing `<end>`/`<pad>` are skipped) into rows split at `<nl>`.
/// A `<lcel>` extends the colspan of the nearest data cell to its left in the same row, a `<ucel>` the rowspan of the cell above in the same
/// column, an `<xcel>` both (it sits under a `<lcel>` or to the right of a `<ucel>`; the cell it belongs to is found through those). A
/// continuation with nothing to continue (a ragged or broken sequence) is dropped: the parse never fails, because the model is greedy and
/// a half-valid table is more useful to the caller than none.
pub fn parse(tokens: &[i64]) -> Grid {
    // grid[r][c] = (index into cells of the cell that covers this position)
    let mut cells: Vec<Cell> = Vec::new();
    let mut grid: Vec<Vec<Option<usize>>> = vec![Vec::new()];
    let mut box_index = 0usize;
    for &t in tokens {
        if t == BOS || t == 0 {
            continue;
        }
        if t == EOS {
            break;
        }
        if t == NL {
            grid.push(Vec::new());
            continue;
        }
        let r = grid.len() - 1;
        let c = grid[r].len();
        if is_data_cell(t) {
            cells.push(Cell { token: t, row: r, col: c, rowspan: 1, colspan: 1, box_index });
            box_index += 1;
            grid[r].push(Some(cells.len() - 1));
        } else if t == LCEL || t == UCEL || t == XCEL {
            let left = if c > 0 { grid[r][c - 1] } else { None };
            let above = if r > 0 { grid[r - 1].get(c).copied().flatten() } else { None };
            let owner = match t {
                LCEL => left,
                UCEL => above,
                _ => above.or(left),
            };
            if let Some(o) = owner {
                if t == LCEL || t == XCEL {
                    cells[o].colspan = cells[o].colspan.max(c + 1 - cells[o].col);
                }
                if t == UCEL || t == XCEL {
                    cells[o].rowspan = cells[o].rowspan.max(r + 1 - cells[o].row);
                }
            }
            grid[r].push(owner);
        }
        // any other token (pad inside the sequence, unknown) is ignored
    }
    while grid.last().is_some_and(|r| r.is_empty()) && grid.len() > 1 {
        grid.pop();
    }
    let cols = grid.iter().map(Vec::len).max().unwrap_or(0);
    Grid { rows: if cells.is_empty() { 0 } else { grid.len() }, cols, cells }
}

#[cfg(test)]
mod tests {
    use super::*;

    // helper: names to ids
    fn ids(s: &str) -> Vec<i64> {
        s.split_whitespace().map(|n| (0..13).find(|&i| name(i) == n).unwrap_or_else(|| panic!("no token {n}"))).collect()
    }

    #[test]
    fn the_data_cell_set_is_the_five_cell_starting_tokens() {
        let d: Vec<i64> = (0..13).filter(|&t| is_data_cell(t)).collect();
        assert_eq!(d, vec![4, 5, 10, 11, 12]);
        assert_eq!(name(5), "<fcel>");
        assert_eq!(name(12), "<srow>");
    }

    #[test]
    fn a_plain_two_by_three_grid() {
        let g = parse(&ids("<start> <ched> <ched> <ched> <nl> <fcel> <fcel> <ecel> <nl> <end>"));
        assert_eq!((g.rows, g.cols, g.cells.len()), (2, 3, 6));
        assert_eq!((g.cells[0].row, g.cells[0].col, g.cells[0].token), (0, 0, 10));
        assert_eq!((g.cells[5].row, g.cells[5].col, g.cells[5].token), (1, 2, 4));
        assert!(g.cells.iter().all(|c| c.rowspan == 1 && c.colspan == 1));
        assert_eq!(g.cells.iter().map(|c| c.box_index).collect::<Vec<_>>(), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn lcel_widens_the_cell_on_its_left() {
        let g = parse(&ids("<start> <ched> <lcel> <ched> <nl> <fcel> <fcel> <fcel> <nl> <end>"));
        assert_eq!(g.cells.len(), 5, "an <lcel> is not a cell");
        assert_eq!((g.cells[0].colspan, g.cells[0].rowspan), (2, 1));
        assert_eq!((g.cells[1].col, g.cells[1].colspan), (2, 1));
        assert_eq!(g.cols, 3);
    }

    #[test]
    fn ucel_makes_the_cell_above_taller() {
        let g = parse(&ids("<start> <rhed> <fcel> <nl> <ucel> <fcel> <nl> <end>"));
        assert_eq!(g.cells.len(), 3);
        assert_eq!((g.cells[0].rowspan, g.cells[0].colspan), (2, 1));
        assert_eq!((g.rows, g.cols), (2, 2));
    }

    #[test]
    fn xcel_in_the_corner_of_a_two_by_two_span() {
        let g = parse(&ids("<start> <fcel> <lcel> <fcel> <nl> <ucel> <xcel> <fcel> <nl> <end>"));
        assert_eq!(g.cells.len(), 3, "the big cell and the two cells beside it");
        assert_eq!((g.cells[0].rowspan, g.cells[0].colspan), (2, 2));
        assert_eq!((g.cells[2].row, g.cells[2].col), (1, 2));
        assert_eq!((g.rows, g.cols), (2, 3));
    }

    #[test]
    fn a_broken_sequence_never_panics_and_keeps_what_it_can() {
        // continuations with nothing to continue, no <end>, a stray pad, an unknown id
        let g = parse(&[2, 6, 7, 8, 5, 0, 99, 9, 7, 5]);
        assert_eq!(g.cells.len(), 2);
        assert_eq!(g.cells[0].col, 3, "the three dropped continuations still occupy their columns");
        assert!(parse(&[]).cells.is_empty());
        assert_eq!(parse(&[2, 3]).rows, 0);
    }

    #[test]
    fn a_trailing_empty_row_is_not_counted() {
        let g = parse(&ids("<start> <fcel> <nl> <fcel> <nl> <end>"));
        assert_eq!((g.rows, g.cols), (2, 1));
    }
}
