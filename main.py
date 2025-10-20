"""
Main application entry point for OCR + DeepSeek Assistant.
Coordinates between configuration, OCR processing, AI analysis, and user input handling.
"""

import logging
import time
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.traceback import install

# Import custom modules
from config import load_config
from ocr import OCRProcessor
from deepseek import DeepSeekClient
from input_handler import InputHandler
from i18n import init_i18n

# Install rich traceback handler for beautiful exception display
install(show_locals=True)

# Use rich's RichHandler to replace default logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

console = Console()


class OCRAssistant:
    """Main application class that coordinates all components."""
    
    def __init__(self):
        """Initialize the OCR Assistant with all required components."""
        # Load configuration and initialize i18n
        self.config = load_config()
        self.i18n = init_i18n(self.config)
        console.print(self.i18n.t("app.init_assistant"))
        console.print(self.i18n.t("app.loading_config"))
        
        # Initialize OCR processor
        console.print(self.i18n.t("app.ocr_config"))
        self.ocr_processor = OCRProcessor(self.config)
        
        # Initialize DeepSeek client
        self.deepseek_client = DeepSeekClient(self.config)
        
        # Initialize input handler with processing callback
        self.input_handler = InputHandler(self.process_selection)
        
        console.print(self.i18n.t("app.init_complete"))

    def process_selection(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        """
        Process a selected screen region through OCR and AI analysis.
        
        Args:
            start: Starting coordinates (top-left)
            end: Ending coordinates (bottom-right)
        """
        console.print(self.i18n.t("app.selection_start"))
        
        x1, y1 = start
        x2, y2 = end
        left = min(x1, x2)
        top = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        if width < 5 or height < 5:
            console.print(self.i18n.t("app.selection_too_small", width=width, height=height))
            return

        console.print(self.i18n.t("app.selection_rect", left=left, top=top, width=width, height=height))

        # Perform OCR on the selected region
        text = self.ocr_processor.process_region(left, top, width, height)
        
        if not text.strip():
            console.print(self.i18n.t("app.no_text"))
            return

        # Send to DeepSeek for AI analysis using two-stage pipeline
        console.print(self.i18n.t("app.deepseek_prepare"))
        self.deepseek_client.send_to_deepseek_pipeline(text)
        console.print(self.i18n.t("app.selection_done"))

    def run(self):
        """Start the application and run the main loop."""
        panel_text_cn = (
            "[bold green]🚀 启动成功！[/bold green]\n\n"
            "📋 使用说明:\n"
            "• 按住 [bold cyan]Ctrl[/bold cyan]，第一次左键点击设置左上角\n"
            "• 第二次左键点击设置右下角\n"
            "• 自动识别文本并发送到 DeepSeek（默认推理模型）\n"
            "• 按 [bold red]Ctrl+C[/bold red] 退出"
        )
        panel_text_en = (
            "[bold green]🚀 Startup successful![/bold green]\n\n"
            "📋 Usage instructions:\n"
            "• Hold [bold cyan]Ctrl[/bold cyan], first left click sets top-left corner\n"
            "• Second left click sets bottom-right corner\n"
            "• Automatically recognizes text and sends to DeepSeek (default reasoning model)\n"
            "• Press [bold red]Ctrl+C[/bold red] to exit"
        )
        console.print(
            Panel(
                panel_text_cn if self.i18n.lang_code == "zh" else panel_text_en,
                title=f"[bold blue]{self.i18n.t('app.panel_title')}[/bold blue]",
                border_style="blue",
            )
        )

        console.print(self.i18n.t("app.initializing_app"))
        
        # Start input listeners
        self.input_handler.start_listeners()

        try:
            # Main application loop
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            console.print(self.i18n.t("app.exit"))
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up all resources before exit."""
        # Stop input listeners
        self.input_handler.stop_listeners()
        
        # Clean up input handler resources
        self.input_handler.cleanup()
        
        # Close DeepSeek client
        self.deepseek_client.close()
        
        console.print(self.i18n.t("app.exit_done"))


def main():
    """Main entry point of the application."""
    app = OCRAssistant()
    app.run()


if __name__ == "__main__":
    main()
