"""
Input handling module using pynput to capture Ctrl + two left clicks
and report the selected rectangle coordinates via a callback.
"""

from typing import Callable, Optional, Tuple
import threading

from pynput import keyboard
from pynput import mouse


class InputHandler:
    """Listen for Ctrl + two left-clicks and invoke a callback with points."""

    def __init__(self, selection_callback: Callable[[Tuple[int, int], Tuple[int, int]], None]):
        self.selection_callback = selection_callback
        self.key_listener: Optional[keyboard.Listener] = None
        self.mouse_listener: Optional[mouse.Listener] = None
        self.ctrl_down: bool = False
        self.first_point: Optional[Tuple[int, int]] = None
        self.running: bool = False
        self._processing: bool = False

    # --- Keyboard events ---
    def _on_key_press(self, key):
        try:
            if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                self.ctrl_down = True
        except Exception:
            pass

    def _on_key_release(self, key):
        try:
            if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                self.ctrl_down = False
                # If Ctrl released mid-selection, reset.
                self.first_point = None
        except Exception:
            pass

    # --- Mouse events ---
    def _on_click(self, x: int, y: int, button: mouse.Button, pressed: bool):
        if not pressed:
            return
        if button != mouse.Button.left:
            return
        if not self.ctrl_down:
            return
        if self._processing:
            return

        if self.first_point is None:
            # First click while Ctrl is held: set top-left anchor
            self.first_point = (x, y)
        else:
            # Second click while Ctrl is held: set bottom-right and trigger callback asynchronously
            second_point = (x, y)
            first_point = self.first_point
            self.first_point = None
            self._processing = True

            def worker():
                try:
                    self.selection_callback(first_point, second_point)
                finally:
                    self._processing = False

            threading.Thread(target=worker, daemon=True).start()

    # --- Public API ---
    def start_listeners(self):
        """Start keyboard and mouse listeners."""
        if self.running:
            return
        self.running = True

        self.key_listener = keyboard.Listener(
            on_press=self._on_key_press, on_release=self._on_key_release
        )
        self.key_listener.start()

        self.mouse_listener = mouse.Listener(on_click=self._on_click)
        self.mouse_listener.start()

    def stop_listeners(self):
        """Stop keyboard and mouse listeners."""
        if self.key_listener is not None:
            try:
                self.key_listener.stop()
            except Exception:
                pass
            self.key_listener = None
        if self.mouse_listener is not None:
            try:
                self.mouse_listener.stop()
            except Exception:
                pass
            self.mouse_listener = None
        self.running = False

    def cleanup(self):
        """Clean up resources. Stop listeners if running."""
        self.stop_listeners()