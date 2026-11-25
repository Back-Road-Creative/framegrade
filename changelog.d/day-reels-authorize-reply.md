### Fixed

- `gomoveshift day-reels authorize` reads the token rclone v1.60 actually prints: base64-wrapped `{"token": ...}`, not raw JSON. It sends the scope blob URL-safe and unpadded, as rclone requires.
