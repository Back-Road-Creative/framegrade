### Added

- Add an unwired single-Page-photo delivery consumer with a required trusted
  authorization port, atomic write-ahead attempts and retained-ID reconciliation.
  Uncertain writes remain held across crashes, retry exhaustion and ordinary
  cancellation/recovery; exact captions, bytes and Page identity stay bound to
  the approved item. Production publishing remains disabled pending the real
  approval consumer and explicit route activation.
