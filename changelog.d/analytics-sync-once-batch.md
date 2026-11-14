### Fixed

- The YouTube analytics sync saved each channel video's event one at a time (every save rewrote the whole event store and rebuilt its indexes) and ran again for every long-form. It now collects the events and saves them in one batch (falling back to one save per event when the store has no batch method), and runs once per Phase D run. A 1126-video channel took about 23 minutes in one sync before any upload.
