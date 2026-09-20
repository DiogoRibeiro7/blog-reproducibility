# Economic release vintages

`gdp_q1_2025_release_vintages.csv` holds three dated publications of one number:
US real GDP growth for 2025Q1, as the Bureau of Economic Analysis published it
in its advance, second, and third estimates.

| | |
| --- | --- |
| Source | Bureau of Economic Analysis news releases; each row carries its own URL |
| Retrieved | September 2026, transcribed by hand from the published releases |
| Series | US real GDP growth, 2025Q1 |
| Units | Percent change from the previous quarter at a seasonally adjusted annual rate |
| Rows | 3 |
| Licence | US government work, not subject to copyright in the United States |

## Columns

`series`, `reference_period`, `release_name`, `released_at`, `value`, `units`,
`source_url`. `released_at` is an ISO 8601 timestamp with its UTC offset, which
is the column the whole example turns on.

## Limitations

- **Three releases, not all of them.** BEA revises further in later annual and
  comprehensive updates. Those revisions exist and are outside this example,
  which is about the first three months rather than the full revision history.
- **Availability is treated as immediate.** The model assumes a number is usable
  the instant it is released. In practice a pipeline ingests it later, and a
  serious backtest would record that lag too.
- **Transcribed by hand.** The values and timestamps were read from the
  published releases rather than pulled from an API, so the file is small enough
  to check by eye and does not depend on a service staying available.

The website keeps its own copy at `/assets/data/gdp_q1_2025_release_vintages.csv`
because the article links to it as a download. This copy exists so the model is
reproducible without the website.
