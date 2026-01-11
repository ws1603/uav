# ui/widgets/__init__.py

"""UI组件模块"""

try:
    from ui.widgets.simple_3d_view import Simple3DView
except ImportError:
    Simple3DView = None

__all__ = ['Simple3DView']
