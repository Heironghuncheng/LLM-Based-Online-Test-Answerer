"""
OCR module for screen capture and text recognition using Tesseract.
Handles region capture and text extraction from images.
"""

import os
import mss
import pytesseract
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from i18n import init_i18n

console = Console()


class OCRProcessor:
    """Handles OCR operations including screen capture and text recognition."""
    
    def __init__(self, config: dict):
        """
        Initialize OCR processor with configuration.
        
        Args:
            config: Configuration dictionary containing OCR settings
        """
        self.i18n = init_i18n(config)
        
        # Set OCR language
        self.lang = config.get("tesseract_lang") or "chi_sim+eng"
        console.print(self.i18n.t("ocr.lang", lang=self.lang))
        
        # Set Tesseract command path if specified
        t_cmd = config.get("tesseract_cmd")
        if t_cmd:
            pytesseract.pytesseract.tesseract_cmd = t_cmd
            console.print(self.i18n.t("ocr.tesseract_custom_path", path=t_cmd))
        else:
            console.print(self.i18n.t("ocr.tesseract_default"))
    
    @staticmethod
    def capture_region(left: int, top: int, width: int, height: int) -> Image.Image | None:
        """
        Capture a specific region of the screen.
        
        Args:
            left: Left coordinate of the region
            top: Top coordinate of the region
            width: Width of the region
            height: Height of the region
            
        Returns:
            PIL Image object or None if capture fails
        """
        try:
            with mss.mss() as sct:
                bbox = {"left": left, "top": top, "width": width, "height": height}
                shot = sct.grab(bbox)
                # mss returns BGRA, convert to RGB using PIL
                img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
                return img
        except Exception as e:
            console.print(
                Panel(
                    f"Screenshot exception: {e}", 
                    title="[red]❌ Screenshot Error[/red]", 
                    border_style="red"
                )
            )
            return None
    
    def ocr_image(self, img: Image.Image) -> str:
        """
        Extract text from an image using OCR.
        
        Args:
            img: PIL Image object to process
            
        Returns:
            str: Extracted text or empty string if OCR fails
        """
        try:
            # For Chinese recognition, ensure chi_sim language pack is installed
            text = pytesseract.image_to_string(img, lang=self.lang)
            return text.strip()
        except Exception as e:
            console.print(
                Panel(
                    f"OCR recognition exception: {e}",
                    title="[red]❌ OCR Error[/red]",
                    border_style="red",
                )
            )
            return ""
    
    def process_region(self, left: int, top: int, width: int, height: int) -> str:
        """
        Capture a screen region and extract text from it.
        
        Args:
            left: Left coordinate of the region
            top: Top coordinate of the region
            width: Width of the region
            height: Height of the region
            
        Returns:
            str: Extracted text or empty string if processing fails
        """
        console.print(self.i18n.t("ocr.capture_prepare"))
        
        with console.status(f"[bold green]{self.i18n.t('ocr.capturing')}"):
            img = self.capture_region(left, top, width, height)
        
        if img is None:
            console.print(f"[red]{self.i18n.t('ocr.capture_failed')}[/red]")
            return ""
        
        console.print(self.i18n.t("ocr.capture_done"))
        console.print(self.i18n.t("ocr.ocr_prepare"))
        
        with console.status(f"[bold blue]{self.i18n.t('ocr.ocr_running')}"):
            text = self.ocr_image(img)
        
        if text.strip():
            console.print(
                Panel(text, title=f"[green]{self.i18n.t('ocr.recognition_complete')}[/green]", border_style="green")
            )
        else:
            console.print(self.i18n.t("ocr.no_text"))
        
        return text