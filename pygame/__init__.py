# Minimal pygame stub for headless test environment
# Provides just enough to allow imports in tests without requiring the real pygame library.

# Dummy module to return placeholders for any attribute
class _DummyModule:
    def __getattr__(self, name):
        def _dummy(*args, **kwargs):
            return None
        return _dummy

def init():
    return None

def quit():
    return None

# Dummy Surface class
class Surface:
    def __init__(self, *args, **kwargs):
        pass

# Dummy draw module with rect and line functions
class _Draw:
    @staticmethod
    def rect(surface, color, rect, width=0):
        return None
    @staticmethod
    def line(surface, color, start_pos, end_pos, width=1):
        return None

# Dummy Rect placeholder
class Rect(_DummyModule):
    pass

# Dummy font module
class font:
    @staticmethod
    def get_init():
        return False
    @staticmethod
    def init():
        return None
    @staticmethod
    def SysFont(name=None, size=12):
        class _Font:
            def render(self, *args, **kwargs):
                return None
        return _Font()
    class Font:
        def __init__(self, *args, **kwargs):
            pass
        def render(self, *args, **kwargs):
            return None

# Dummy display module
class display:
    @staticmethod
    def set_mode(*args, **kwargs):
        return Surface()
    @staticmethod
    def flip():
        return None

# Dummy event module
class event:
    @staticmethod
    def get():
        return []

# Exported names
draw = _Draw()
__all__ = ["init", "quit", "Surface", "Rect", "font", "display", "event", "draw"]
