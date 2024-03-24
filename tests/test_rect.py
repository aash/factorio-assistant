

from common import Rect, Segment

def test_rect_repr_from_xyxy():
    r = Rect(0, 1, 2, 3)
    r1 = Rect.from_xyxy(0, 1, 2, 3)
    assert repr(r) == 'Rect(x0=0, y0=1, w=2, h=3)'
    assert repr(r1) == 'Rect(x0=0, y0=1, w=2, h=2)'

def test_rect_basic_api():
    r = Rect(0, 1, 2, 3)
    assert r.wh() == (2, 3)
    assert r.xy() == (0, 1)
    assert r.width() == 2
    assert r.height() == 3
    assert r.top_segment() == Segment(0, 2)
    assert r.left_segment() == Segment(1, 4)

def test_rect_ctors():
    a = (0, 1)
    b = (7, 9)
    r0 = Rect.from_xyxy(*a, *b)
    r1 = Rect.from_top_left(*a, *r0.wh())
    r2 = Rect(*a, *r0.wh())
    r3 = Rect.from_bottom_left(*r1.bottom_left(), *r1.wh())
    r4 = Rect.from_bottom_right(*r1.bottom_right(), *r1.wh())
    r5 = Rect.from_top_right(*r1.top_right(), *r1.wh())
    assert r0 == r1 == r2 == r3 == r4 == r5

def test_rect_dynamic_call():
    a = (0, 1)
    b = (7, 9)
    r0 = Rect.from_xyxy(*a, *b)
    bottom_top, right_left = False, True
    d = {
        (True, True): 'top_left',
        (True, False): 'top_right',
        (False, True): 'bottom_left',
        (False, False): 'bottom_right',
    }
    m = getattr(Rect, f'from_{d[(bottom_top, right_left)]}')
    r1 = m(*r0.bottom_left(), *r0.wh())
    assert r0 == r1
