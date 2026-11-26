### Fixed

- Within a platform, the drain now sends the delivery with the oldest slot first, not the one enqueued first, so a backlog after an outage goes out in slot order.
