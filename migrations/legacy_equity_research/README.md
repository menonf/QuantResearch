# The first migration history (EquityResearch, 0001-0011) - kept for reference, never run

These are the migrations of the database layout before the multi-asset refactor of November 2026: the schemas `ref`,
`market` and `fund`, shared by every asset class. `equity_research_local` (migration 0011) and the Azure database
`equity_research` (0010) are on this history. Alembic does not read this folder.

The current history (`migrations/versions`, from 0001) lays the database out as the code is: `core`, `ingest`, `ops`
and `research` for `research_core`, and one schema per asset class (`equity`, `fixed_income`). A database on this
history is not upgraded: it is rebuilt from the vendors on the new one (`quant_research_local`, November 2026).
