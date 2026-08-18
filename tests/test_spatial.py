from app.models.document import BBox, TextBlock
from app.stages.spatial import group_into_rows


def _block(block_id: str, text: str, x0: float, y0: float, x1: float, y1: float) -> TextBlock:
    return TextBlock(
        block_id=block_id,
        text=text,
        bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
        reading_order=0,
    )


def test_group_into_rows_handles_baseline_shift_and_columns():
    blocks = [
        _block("p1-b3", "159/-", 40, 114, 70, 124),
        _block("p1-b4", "129/-", 80, 114, 110, 124),
        _block("p1-b0", "Veg noodles", 140, 100, 260, 112),
        _block("p1-b1", "139/-", 40, 102, 70, 112),
        _block("p1-b2", "109/-", 80, 102, 110, 112),
        _block("p1-b5", "Egg noodles", 140, 112, 260, 124),
    ]

    grouped = group_into_rows(blocks)

    first_row = [b.text for b in grouped if b.row_id == 0]
    second_row = [b.text for b in grouped if b.row_id == 1]

    assert first_row == ["139/-", "109/-", "Veg noodles"]
    assert second_row == ["159/-", "129/-", "Egg noodles"]
    assert [b.reading_order for b in grouped] == list(range(len(grouped)))
