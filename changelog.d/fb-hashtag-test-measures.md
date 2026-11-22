### Fixed

- The `fb_hashtags` caption experiment now measures what it assigns. A Short whose arm would ship
  no hashtags like arm A (no location tag to carry, or an interactive Short) is no longer tagged
  into arm B and keeps the normal hashtags. `queue edit-facebook-caption` keeps the Short's arm:
  arm A adds no hashtags, B and C reproduce their own set.
