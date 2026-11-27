### Fixed

- Moving a slot from the console now moves it in the publish plan too, so YouTube and Facebook publish at the new time. The time is stored as UTC, and a time with no zone is refused with a message saying so.
- A delivery whose slot has no zone no longer stops the whole drain pass when it is held.
