"""Explicit plotting-only compatibility for unmodified YOLOv5 v7.0 on Python 3.12."""


def enable_legacy_font_size():
    # YOLOv5 v7.0 plots call getsize, removed by Pillow 10 (Python 3.12 wheels).
    # Preserve old height semantics (bottom includes the top bearing).
    from PIL import ImageFont
    def getsize(self,text,*args,**kwargs):
        left,top,right,bottom=self.getbbox(text,*args,**kwargs)
        return right-left,bottom
    for cls in (ImageFont.FreeTypeFont,ImageFont.ImageFont):
        if not hasattr(cls,'getsize'):
            cls.getsize=getsize
