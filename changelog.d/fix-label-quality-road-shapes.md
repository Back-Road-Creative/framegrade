### Fixed

- The label-quality backstop no longer warns "county-only" on a settlement-led label such as "Goleen, County Cork"; road-led labels ("R591, County Cork", "NC Highway 28, Macon County, NC", "M50 Motorway, County Dublin", or any road the resolver recorded) still warn. The check stays warn-only.
- The shorts CSV grammar gate now checks each YouTube tag on its own, so a tag that starts with a bare preposition is caught anywhere in the cell, not only when it is first.
- Shorts tags strip a leading title word ("To", "Through") from each location component, so "To Goleen, County Cork" no longer yields a "To Goleen" tag.
