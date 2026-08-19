import threading
import queue
import sshkeyboard


class Keyboard:
    def __init__(self):
        self._keys = queue.Queue()
        self._pressed = set()
        self._lock = threading.Lock()

        self._thread = threading.Thread(
            target=self._listen,
            daemon=True,
        )
        self._thread.start()

    def _listen(self):
        def on_press(key):
            with self._lock:
                self._pressed.add(key)
            self._keys.put(key)

        def on_release(key):
            with self._lock:
                self._pressed.discard(key)

        sshkeyboard.listen_keyboard(
            on_press=on_press,
            on_release=on_release,
        )


    def is_pressed(self, key=None):
        """Non-blocking check.

        If key is None, returns True if any key is currently pressed.
        Otherwise checks whether the specified key is pressed.
        """
        with self._lock:
            if key is None:
                return bool(self._pressed)
            return key in self._pressed

    def get_key(self):
        """Return the next key press without blocking.

        Returns None if no key has been pressed.
        """
        try:
            k = self._keys.get_nowait()
            return k
        except queue.Empty:
            return None